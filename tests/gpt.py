from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os

from huggingface_hub import login

class PDBDataset(Dataset):
    def __init__(self, pdb_dir, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        pdb_files = [f for f in os.listdir(pdb_dir) if f.startswith('monomer') and f.endswith('.pdb')]
        for file in pdb_files:
            full_path = os.path.join(pdb_dir, file)
            with open(full_path, 'r') as f:
                lines = f.readlines()
                prop_line = next((line for line in lines if line.startswith('REMARK')), None)
                if prop_line:
                    parts = prop_line.split()
                    prop_name = parts[1]
                    prop_value = parts[2]
                    prop = f"{prop_name}: {prop_value}"
                else:
                    prop = "Unknown: 0.0"  # Fallback
                pdb_text = ''.join(lines)
            text = f"Property: {prop}\n{pdb_text}"
            self.data.append(tokenizer(text, return_tensors='pt')['input_ids'])

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

gpt = 'meta-llama/Llama-3.1-8B'
tokenizer = AutoTokenizer.from_pretrained(gpt)
model = AutoModelForCausalLM.from_pretrained(gpt)

dataset = PDBDataset('xyz_files', tokenizer)
dataloader = DataLoader(dataset, batch_size=2)