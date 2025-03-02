import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

################################################################################
# SteeringVector
################################################################################
class SteeringVector:
    def __init__(self, directions: dict[int, np.ndarray]):
        self.directions = directions

    def __add__(self, other: "SteeringVector") -> "SteeringVector":
        new_dirs = {}
        for k in set(self.directions.keys()).union(other.directions.keys()):
            new_dirs[k] = self.directions.get(k, 0) + other.directions.get(k, 0)
        return SteeringVector(new_dirs)

    def __sub__(self, other: "SteeringVector") -> "SteeringVector":
        return self + (-other)

    def __neg__(self) -> "SteeringVector":
        return SteeringVector({k: -v for k, v in self.directions.items()})

    def __mul__(self, scalar: float) -> "SteeringVector":
        return SteeringVector({k: scalar * v for k, v in self.directions.items()})

    def __rmul__(self, scalar: float) -> "SteeringVector":
        return self.__mul__(scalar)

################################################################################
# SteeringModel
################################################################################
class SteeringModel(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, layer_ids: list[int], device: str):
        super().__init__()
        self.base_model = base_model.to(device)
        self.steering_vector = None
        self.coeff = 1.0
        self.layer_ids = layer_ids
        self.operator = lambda h, c: h + c
        self.normalize = False

        if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            self.layers = base_model.transformer.h
        elif hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            self.layers = base_model.model.layers
        else:
            raise ValueError("Cannot find transformer layers in model.")

        for idx in layer_ids:
            self.layers[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx: int):
        def hook(module, inp, output):
            if self.steering_vector is None or idx not in self.steering_vector.directions:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            control = torch.tensor(self.steering_vector.directions[idx],
                                   dtype=hidden.dtype, device=hidden.device)
            control = control.view(1, 1, -1).expand_as(hidden)
            steered = self.operator(hidden, self.coeff * control)
            if self.normalize:
                steered = steered / (steered.norm(dim=-1, keepdim=True) + 1e-6) * hidden.norm(dim=-1, keepdim=True)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        return hook

    def set_steering(self, vector: "SteeringVector", coeff: float = 1.0, operator=None, normalize=False):
        self.steering_vector = vector
        self.coeff = coeff
        if operator is not None:
            self.operator = operator
        self.normalize = normalize

    def reset_steering(self):
        self.steering_vector = None
        self.coeff = 1.0
        self.operator = lambda h, c: h + c
        self.normalize = False

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def default_generate(self, input_ids: torch.LongTensor, max_new_tokens: int, **kwargs) -> torch.LongTensor:
        return self.base_model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

    @torch.no_grad()
    def steered_generate(self, input_ids: torch.LongTensor, max_new_tokens: int, **kwargs) -> torch.LongTensor:
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=input_ids, **kwargs)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

################################################################################
# Training: Contrastive
################################################################################
def train_steering_vector_contrastive(
    model: SteeringModel,
    tokenizer,
    prompt_pairs: list[tuple[str, str]],
    layer_ids: list[int],
    device: str,
    batch_size: int = 2,
    show_progress: bool = False
) -> "SteeringVector":
    all_prompts = [p for pair in prompt_pairs for p in pair]
    hidden_states = {layer: [] for layer in layer_ids}
    model.eval()
    with torch.no_grad():
        indices = range(0, len(all_prompts), batch_size)
        if show_progress:
            indices = tqdm.tqdm(indices, desc="Extracting Hiddens (Contrastive)")
        for start in indices:
            batch = all_prompts[start:start+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            for i in range(len(batch)):
                seq_len = int(enc["attention_mask"][i].sum() - 1)
                for layer in layer_ids:
                    vec = out.hidden_states[layer+1][i, seq_len, :].cpu().numpy()
                    hidden_states[layer].append(vec)

    import numpy as np
    from sklearn.decomposition import PCA
    directions = {}
    for layer in layer_ids:
        arr = np.array(hidden_states[layer])
        # pos - neg for each pair
        diffs = [arr[i] - arr[i+1] for i in range(0, len(arr), 2)]
        diffs = np.array(diffs)
        if diffs.shape[0] == 1:
            direction = diffs[0]
        else:
            pca = PCA(n_components=1)
            pca.fit(diffs)
            direction = pca.components_[0]
        direction = direction.astype(np.float32)

        # Flip sign if negative direction aligns better with positives
        proj = arr.dot(direction)
        pos_idx = list(range(0, len(arr), 2))
        neg_idx = list(range(1, len(arr), 2))
        if np.mean(proj[pos_idx]) < np.mean(proj[neg_idx]):
            direction = -direction
        directions[layer] = direction

    return SteeringVector(directions)

################################################################################
# Training: KTO
################################################################################
def train_steering_vector_kto(
    model: SteeringModel,
    tokenizer,
    examples: list[str],
    labels: list[int],
    layer_ids: list[int],
    device: str,
    batch_size: int = 2,
    show_progress: bool = False
) -> "SteeringVector":
    hidden_states = {layer: [] for layer in layer_ids}
    label_states = {layer: [] for layer in layer_ids}
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(examples), batch_size), desc="Extracting Hiddens (KTO)", disable=not show_progress):
            batch = examples[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            for j in range(len(batch)):
                seq_len = int(enc["attention_mask"][j].sum() - 1)
                for layer in layer_ids:
                    vec = out.hidden_states[layer+1][j, seq_len, :].cpu().numpy()
                    hidden_states[layer].append(vec)
                    label_states[layer].append(batch_labels[j])

    import numpy as np
    directions = {}
    for layer in layer_ids:
        arr = np.array(hidden_states[layer])
        labs = np.array(label_states[layer]).reshape(-1, 1)
        direction = (labs * arr).sum(axis=0) / np.sum(np.abs(labs))
        directions[layer] = direction.astype(np.float32)

    return SteeringVector(directions)
