# rast_app.py  ── Streamlit playground for Redundancy-Aware Steering Technique
import os, json, random, numpy as np, torch, torch.nn.functional as F
import streamlit as st
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from steerit.steerit import SteeringVector, SteeringModel     # your library

# ────────── Sidebar config ────────────────────────────────────────────────
st.sidebar.title("RAST Playground")

MODEL_NAME  = st.sidebar.text_input(
    "🤖 HF model name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
ALPHA_HI    = st.sidebar.slider("α (steering strength on)", 0.1, 2.0, 1.0, 0.1)
K_WIN       = st.sidebar.slider("K (ΔKL window)",            2, 12, 6)
DKL_THR     = st.sidebar.slider("ΔKL threshold τ",           0.01, 0.2, 0.05, 0.01)

st.sidebar.markdown("---")
data_repo   = st.sidebar.text_input("📚 HF dataset", "gsm8k")
data_split  = st.sidebar.text_input("split", "test")
N_TRAIN     = st.sidebar.number_input("#train items", 10, 500, 50, 10)
N_EVAL      = st.sidebar.number_input("#eval items",  10, 300, 50, 10)

MAX_TOKENS  = st.sidebar.number_input("max_new_tokens", 64, 4096, 256, 64)
STEER_LAY   = st.sidebar.number_input("steer layer idx", 0, 40, 20, 1)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float16 if DEVICE == "cuda" else torch.float32

# ────────── Main layout ───────────────────────────────────────────────────
st.title("RAST - Redundancy-Aware Steering Technique")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Concise example (high-gain)**")
    pos_example = st.text_area("Positive (concise)", "Question: 2+2=\nAnswer: 4.")
with col2:
    st.markdown("**Verbose examples (low-gain)**")
    neg_examples_raw = st.text_area(
        "Negative (verbose, JSON list)",
        '["Question: 2+2=\\nAnswer: First we observe 2 is two units, '
        'and adding another 2 gives 4."]'
    )

# ────────── Load / cache model ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    base= AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=DTYPE, device_map="auto" if DEVICE=="cuda" else None
    )
    mdl = SteeringModel(base, [STEER_LAY], DEVICE)
    return tok, mdl
tokenizer, model = load_model(MODEL_NAME)

# ────────── Helper: ΔKL & generation utils (trimmed cache) ───────────────
def kl_div(p, q):
    return F.kl_div(F.log_softmax(p,dim=-1), F.softmax(q,dim=-1),
                    reduction="batchmean").item()

@torch.no_grad()
def generate(tokens, past=None):
    """Stream one token; returns new_ids, new_past, logits"""
    out = model(input_ids=tokens[:, -1:], use_cache=True, past_key_values=past)
    past = tuple((k[..., -K_WIN:, :], v[..., -K_WIN:, :]) for k,v in out.past_key_values)
    return out.logits[:, -1, :], past

def rast_stream(prompt, vec):
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    past, lg_buf = None, []
    model.set_steering(vec, coeff=0.)
    for _ in range(MAX_TOKENS):
        logits, past = generate(ids, past)
        lg_buf.append(logits.detach())
        if len(lg_buf) > K_WIN and kl_div(logits, lg_buf[-K_WIN-1]) < DKL_THR:
            model.coeff = ALPHA_HI
        else:
            model.coeff = 0.0
        next_tok = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=-1)
        if next_tok.item() in {tokenizer.eos_token_id}:
            break
    model.reset_steering()
    return tokenizer.decode(ids.squeeze(), skip_special_tokens=True)

# ────────── Build vector button ───────────────────────────────────────────
build_clicked = st.button("🛠️  Generate RAST Vector")

if build_clicked:
    try:
        neg_examples = json.loads(neg_examples_raw)
        assert isinstance(neg_examples, list) and len(neg_examples) > 0
    except Exception as e:
        st.error("Negative examples must be a JSON list of strings.")
        st.stop()

    pos_list = [pos_example.strip()] * len(neg_examples)
    train_pairs = list(zip(pos_list, neg_examples))
    st.write(f"Training on {len(train_pairs)} positive/negative pairs…")

    # collect hidden states
    hi_vecs, lo_vecs = [], []
    for pos, neg in train_pairs:
        for lbl, text in [(1, pos), (0, neg)]:
            ids, _ = tokenizer(text, return_tensors="pt").to(DEVICE)["input_ids"], None
            out = model(input_ids=ids, output_hidden_states=True)
            h = out.hidden_states[STEER_LAY+1][0, -1, :].cpu().numpy()
            (hi_vecs if lbl else lo_vecs).append(h)
    vec_dir = (np.mean(hi_vecs,0) - np.mean(lo_vecs,0)).astype(np.float32)
    rast_vec = SteeringVector({STEER_LAY: vec_dir})
    st.success("RAST vector created and cached ✅")

    st.session_state["rast_vector"] = rast_vec  # cache in session

# ────────── Evaluate button ───────────────────────────────────────────────
if ("rast_vector" in st.session_state) and st.button("📊 Evaluate on Dataset"):
    st.write("Loading dataset slice…")
    ds = load_dataset(data_repo, split=data_split).shuffle(seed=SEED)
    eval_rows = ds.select(range(N_EVAL))
    rast_vec  = st.session_state["rast_vector"]

    base_tokens, rast_tokens = [], []
    for row in stqdm(eval_rows, "Evaluating"):
        prompt = row['question'] if 'question' in row else str(row)
        baseline = rast_stream(prompt, SteeringVector({}))      # α=0 vec
        steered  = rast_stream(prompt, rast_vec)
        base_tokens.append(len(tokenizer.encode(baseline)))
        rast_tokens.append(len(tokenizer.encode(steered)))

    save_pct = 100*(np.mean(base_tokens)-np.mean(rast_tokens))/np.mean(base_tokens)
    st.write(f"**Mean baseline tokens** : {np.mean(base_tokens):.1f}")
    st.write(f"**Mean steered tokens**  : {np.mean(rast_tokens):.1f}")
    st.write(f"**Token saving**         : {save_pct:.1f} %")

# ────────── Live prompt steering ───────────────────────────────────────────
st.markdown("---")
prompt_live = st.text_area("✏️  Enter a prompt to test RAST steering")
if st.button("🚀 Run"):
    if "rast_vector" not in st.session_state:
        st.error("Generate the RAST vector first!")
    else:
        baseline = rast_stream(prompt_live, SteeringVector({}))
        steered  = rast_stream(prompt_live, st.session_state["rast_vector"])
        st.write("**Baseline:**")
        st.code(baseline)
        st.write("**Steered:**")
        st.code(steered)

