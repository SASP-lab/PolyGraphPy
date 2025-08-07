import os
import pandas as pd
import selfies as sf
from sklearn.preprocessing import MinMaxScaler
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import rdkit.Chem as Chem

class GenerativePreprocess:
    def __init__(self, input_csv, output_path='polygraphpy/data/generative_data/'):
        self.input_csv = input_csv
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        df = pd.read_csv(self.input_csv)
        df = df[df['chain_size'] == 0].reset_index(drop=True)
        scaler = MinMaxScaler()
        target = scaler.fit_transform(df['static_polarizability'].values.reshape(-1,1)).flatten()

        smiles_list = df['smiles_A'].values
        selfies_list = [sf.encoder(sm) for sm in smiles_list]

        data_df = pd.DataFrame({'selfies': selfies_list, 'polarizability': target})
        data_df.to_csv(os.path.join(self.output_path, 'training_data.csv'), index=False)

        with open(os.path.join(self.output_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        return self.output_path

class SelfiesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

class GenerativeTrainer:
    def __init__(self, data_path, model_output_path, batch_size=4, learning_rate=5e-5, epochs=100):
        self.data_path = data_path
        self.model_output_path = model_output_path
        os.makedirs(self.model_output_path, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Training in:', self.device)

    def run(self):
        if os.path.exists(os.path.join(self.model_output_path, 'gpt_selfies.pt')):
            print("Existing model found, skipping training.")
            return self.model_output_path
        
        df = pd.read_csv(os.path.join(self.data_path, 'training_data.csv'))

        texts = [f"polarizability: {p} selfies: {i}" for i, p in zip(df['selfies'], df['polarizability'])]

        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model = AutoModelForCausalLM.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))

        dataset = SelfiesDataset(texts, tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        aux = float('inf')

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch['input_ids'].size(0)

            total_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch+1}, Loss: {total_loss:.5f}")

            if total_loss < aux:
                torch.save(model, os.path.join(self.model_output_path, 'gpt_selfies.pt'))
                aux = total_loss

class MoleculeGenerator:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = torch.load(os.path.join(self.model_path, 'gpt_selfies.pt'))
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_one(self, pol_val, max_len=759):
        try:
            prompt = f"polarizability: {pol_val} selfies:"
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(**inputs, max_length=max_len, num_beams=1, do_sample=True, temperature=0.9, top_p=0.9)
            gen_text = self.tokenizer.decode(outputs[0])

            if "selfies:" in gen_text:
                selfies = gen_text.split("selfies:")[1].strip()
            else:
                selfies = None

            if selfies:
                selfies = selfies.split('[PAD]')[0].strip()
                selfies = selfies.replace("\\", "/")

            smiles = sf.decoder(selfies) if selfies else None
            valid = smiles and Chem.MolFromSmiles(smiles) is not None
            print(f"{smiles} - Valid: {valid}")
            return smiles if valid else None
        
        except Exception as e:
            print(e)
            return None

    def run(self, targets):
        data = []

        for i in tqdm(targets):
            smiles = self.generate_one(i)
            if smiles:
                data.append({'smiles': smiles, 'static_polarizability': i})

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_path, 'generated_molecules.csv'), index=False)
        
        return df