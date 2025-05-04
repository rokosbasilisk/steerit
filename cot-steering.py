#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install git+https://github.com/vgel/repeng.git')
get_ipython().system('pip install datasets')


# In[2]:


import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = ''
os.environ["HF_TOKEN"] = hf_token

# Model configuration
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and tokenizer
print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token = hf_token).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)

# Wrap model with ControlModel
control_model = ControlModel(model, list(range(0,28)))


# In[7]:


def make_dataset(positive_examples, negative_examples):
    """ Create a dataset for training the control vector """
    dataset = [
        DatasetEntry(f"Question: {pos}\nAnswer:", f"Question: {neg}\nAnswer:")
        for pos, neg in zip(positive_examples, negative_examples)
    ]
    return dataset

# Load dataset
print("Loading reasoning dataset...")
reasoning_data = load_dataset("rb/aime_reasoning", "default")['train'].to_pandas()

# Split data: 900 for training, 10 for validation, rest for testing
train_data = reasoning_data.iloc[:950]
val_data = reasoning_data.iloc[950:]

print(f"Data splits: {len(train_data)} train, {len(val_data)} validation")

# Prepare positive and negative reasoning examples
positive_examples = [item['refined_reasoning'] for _, item in train_data.iterrows()]
negative_examples = [item['reasoning_content'] for _, item in train_data.iterrows()]

# Create dataset
cot_dataset = make_dataset(positive_examples, negative_examples)

# Train the control vector
print("Training CoT behavior control vector...")
cot_behavior_vector = ControlVector.train(control_model, tokenizer, cot_dataset, batch_size=1)

# Save control vector
vector_path = "cot_behavior_vector.pt"
torch.save(cot_behavior_vector, vector_path)
print(f"CoT behavior vector saved to {vector_path}")

# Load control vector
cot_behavior_vector = torch.load(vector_path)


# In[8]:


def evaluate_vector_strengths(model, tokenizer, prompts, vector, strengths, device='cuda', max_tokens=8192):
    """Evaluate different steering strengths and return results."""
    results = {}

    # Simple logging of available layers
    if hasattr(vector, 'directions'):
        available_layers = list(vector.directions.keys())
        print(f"Vector contains layers: {available_layers}")

    from tqdm import tqdm

    for strength in strengths:
        print(f"Evaluating steering strength {strength}...")

        # Create a new control model with just the available layers
        from repeng import ControlModel
        control_model = ControlModel(model.model, available_layers)
        control_model.set_control(vector, strength)

        responses = []
        token_lengths = []

        for prompt in tqdm(prompts, desc=f"Processing at strength {strength}"):
            # Generate the response
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = control_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                repetition_penalty=1.1,
            )

            response = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
            responses.append(response)
            token_lengths.append(len(tokenizer.encode(response)))

        results[strength] = {
            "responses": responses,
            "mean_length": token_lengths,
            "strength": strength
        }


    return results
# Prepare validation prompts
val_prompts = [f"Solve: {item['question']}\nAnswer: " for _, item in val_data.iterrows()]

# Evaluate different strengths
steered_results = evaluate_vector_strengths(control_model, tokenizer, val_prompts, cot_behavior_vector, [1.5])
original_results = evaluate_vector_strengths(control_model, tokenizer, val_prompts, cot_behavior_vector, [0.0])


# In[19]:


original_results[0.0]


# In[20]:


# Save results
results_df = pd.DataFrame({
    "question": [item['question'] for _, item in val_data.iterrows()],

    "original_response": original_results[0.0]['responses'],
    "original_length": original_results[0.0]['mean_length'],

    "steered_response": steered_results[1.5]['responses'],
    "steered_length": steered_results[1.5]['mean_length'],

})

results_path = "cot_steering_results.csv"
results_df.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")


# In[25]:


results_df['tokens_efficiency'] = (results_df['original_length'] - results_df['steered_length'])*100/results_df['original_length']


# In[39]:


len(results_df[results_df['original_length']==results_df['steered_length']])


# In[41]:


results_df[results_df['original_length']>results_df['steered_length']]['tokens_efficiency']


# In[42]:


results_df['tokens_efficiency']


# In[44]:


results_df_subset = results_df[results_df['tokens_efficiency']]


# In[52]:


results_df_subset = results_df[results_df['tokens_efficiency']>-100]


# In[64]:


results_df_subset = results_df_subset[results_df_subset['tokens_efficiency']!=0]


# In[70]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.figure(figsize=(10, 5))
results_df['original_length'].plot(kind='hist', bins=30, alpha=0.7, title='Distribution of Original Length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Distribution of Original Length')
plt.grid(True)
plt.show()

plt.figure()
results_df['generated_length'].plot(kind='hist', bins=30)
plt.xlabel('Generated Length')
plt.ylabel('Frequency')
plt.title('Distribution of Steered Length')
plt.grid(True)
plt.show()


# In[ ]:




