"""
This module implements a generative model based on the GPT architecture for
designing novel molecules. The process is broken down into three main stages:
1.  **GenerativePreprocess**: Prepares the training data by converting SMILES strings
    to SELFIES and standardizing the target property (e.g., polarizability).
2.  **GenerativeTrainer**: Fine-tunes a pre-trained GPT-2 model on the prepared
    SELFIES data to learn the relationship between a target property and molecular structure.
3.  **MoleculeGenerator**: Uses the trained GPT model to generate new molecular
    structures (as SELFIES), converts them back to SMILES, and then validates
    them using a pre-trained GNN to filter out invalid or low-quality molecules.
"""

import os
import pandas as pd
import selfies as sf
import rdkit.Chem as Chem
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from polygraphpy.gnn.pre_processing import PreProcess
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder

class GenerativePreprocess:
    """Prepares data for training a GPT-based generative model."""
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
    """A PyTorch Dataset for SELFIES strings."""
    def __init__(self, texts, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

class GenerativeTrainer:
    """Fine-tunes a GPT-2 model for molecular generation."""
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
                torch.save(model, f'{self.model_output_path}gpt_selfies.pt')
                aux = total_loss

class MoleculeGenerator:
    """Generates and validates new molecules using a trained GPT model and a GNN."""
    def __init__(self, model_path, output_path, monomers_number_per_target, threshold):
        self.model_path = model_path
        self.output_path = output_path
        self.monomers_number_per_target = monomers_number_per_target
        self.threshold = threshold
        print(f'Generating {self.monomers_number_per_target} monomers per target...')
        os.makedirs(self.output_path, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = torch.load(f'{self.model_path}gpt_selfies.pt', weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the broad Acryloyl pattern to block bifunctional monomers
        self.acryloyl_pattern = Chem.MolFromSmarts('[CX3]=[CX3][CX3](=[OX1])[O,N,S]')
        # Define the pattern specifically to find the root CH2 for SMILES formatting
        self.root_pattern = Chem.MolFromSmarts('[CH2]=[CH][CX3](=[OX1])[O,N,S]')

    def generate_one(self, pol_val, max_len=1000):
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
            return smiles if valid else None
        
        except Exception as e:
            return None
            
    def is_stable(self, mol):
        """Checks if the GPT generated molecule violates basic stability heuristics."""
        unwanted_patterns = [
            '[O,S]-[O,S]',                       # Peroxides / Disulfides
            '[N]-[O]',                           # N-O single bonds
            '[N]=[N+]=[N-]',                     # Azides
            '[CX3](=[OX1])[OX2][CX3](=[OX1])',   # Anhydrides / mixed anhydrides
            'O=C-O-C(=O)-O'                      # Dicarbonates
        ]
        for smarts in unwanted_patterns:
            pat = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pat):
                return False
        return True
    
    def validate_and_root_molecule(self, smiles):
        """Strict validation that returns the rooted SMILES if valid, or None if invalid."""
        if not smiles:
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Check 1: Must be a single fragment (no dots in SMILES)
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            return None
            
        # Check 2: Chemical stability
        if not self.is_stable(mol):
            return None

        # Check 3: EXACTLY ONE polymerizable group (blocks Acrylate + Acrylamide hybrids)
        matches = mol.GetSubstructMatches(self.acryloyl_pattern)
        if len(matches) != 1:
            return None

        # Check 4: Format SMILES to start with C=CC(=O)
        root_matches = mol.GetSubstructMatches(self.root_pattern)
        if len(root_matches) >= 1:
            root_atom_idx = root_matches[0][0]
            try:
                smi = Chem.MolToSmiles(mol, isomericSmiles=False, rootedAtAtom=root_atom_idx)
                # Cleanup RDKit's occasional branching variations
                if smi.startswith('C=CC(=O)O') or smi.startswith('C=CC(O)='): 
                    smi = smi.replace('C=CC(O)=', 'C=CC(=O)') 
                elif smi.startswith('C=CC(=O)N') or smi.startswith('C=CC(N)='):
                    smi = smi.replace('C=CC(N)=', 'C=CC(=O)')
                elif smi.startswith('C=CC(=O)S') or smi.startswith('C=CC(S)='):
                    smi = smi.replace('C=CC(S)=', 'C=CC(=O)')
                return smi
            except:
                return None
        return None
    
    def mol_to_data(self, smiles):
        try:
            atoms = []
            bonds = []
            m1 = Chem.MolFromSmiles(smiles, sanitize=True)
            if m1 is None:
                return None
            m1 = Chem.AddHs(m1)
            
            atoms = self.preprocess.get_nodes_information(m1, [], chain_size=0)
            if not atoms:
                return None
            
            df_nodes = pd.DataFrame(atoms)
            nodes_features = pd.DataFrame(self.atom_encoder.transform(df_nodes.drop(['idx'], axis=1)).toarray())
            x = torch.tensor(nodes_features.astype('float32').values)
            
            bonds = self.preprocess.get_bonds_information(m1, [])
            if not bonds:
                return None
            df_bonds = pd.DataFrame(bonds)
            edge_index = torch.tensor([
                df_bonds.begin_idx.to_list() + df_bonds.end_idx.to_list(),
                df_bonds.end_idx.to_list() + df_bonds.begin_idx.to_list()
            ])
            
            edge_attrs = df_bonds[['type', 'is_conjugated', 'is_aromatic']]
            edge_attrs = pd.concat([edge_attrs, edge_attrs.sort_index(ascending=False)])
            edge_attr = torch.tensor(self.bond_encoder.transform(edge_attrs).toarray())
            
            edge_weight = torch.tensor([1.0] * edge_index.shape[1], dtype=torch.float32)
            
            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

            return mol_data
        except Exception as e:
            print(f"Error converting {smiles}: {e}")
            return None
    
    def load_gnn_model(self):
        self.preprocess = PreProcess(
            input_csv='polygraphpy/data/polarizability_data.csv',
            train_input_data_path='prediction_test',
            polymer_type='monomer',
            target='static_polarizability',
            gnn_output_path='./'
        )

        atoms_list, bonds_list = self.preprocess.extract_atoms_and_bonds_features_from_monomer_smiles()
        self.atom_encoder = self.preprocess.make_encoder(pd.DataFrame(atoms_list).drop_duplicates().reset_index(drop=True))
        self.bond_encoder = self.preprocess.make_encoder(pd.DataFrame(bonds_list).drop_duplicates().reset_index(drop=True))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = torch.load('polygraphpy/data/gnn_output/model_gcn.pt', weights_only=False)
        model = model.to(device)
        
        return model
    
    def fine_tunning_with_gnn(self, df: pd.DataFrame):
        df_results = []

        for row in df.itertuples():
            mol_data = self.mol_to_data(row.smiles_A)
            if mol_data is None:
                continue

            mol_data = mol_data.to(self.device)

            with torch.no_grad():
                batch = torch.zeros(mol_data.x.size(0), dtype=torch.long).to(self.device)
                pred = self.gnn_model(mol_data.x, mol_data.edge_index, mol_data.edge_weight, batch)

            df_results.append({
                'smiles_A': row.smiles_A, 
                'static_polarizability': row.static_polarizability, 
                'static_polarizability_pred': pred.cpu().numpy()[0][0]
            })
            
        return pd.DataFrame(df_results)
    
    def apply_error_threshold(self, df: pd.DataFrame):
        if df.empty:
            return df, df

        with open(f'polygraphpy/data/generative_data/scaler.pkl', 'rb') as file:
            loaded_encoder : OneHotEncoder = pickle.load(file)
        
        df['static_polarizability_original'] = loaded_encoder.inverse_transform(df['static_polarizability'].values.reshape(-1,1))
        df['static_polarizability_pred_original'] = loaded_encoder.inverse_transform(df['static_polarizability_pred'].values.reshape(-1,1))

        df['error'] = abs(df['static_polarizability_original'] - df['static_polarizability_pred_original'])/df['static_polarizability_original']

        df_filtered = df[df['error'] <= self.threshold]

        return df, df_filtered
    
    def post_processing(self, df: pd.DataFrame):
        print('Making post processing with GNN prediction model.')

        df = df.drop_duplicates(subset='smiles').reset_index(drop=True)
        df = df.rename(columns={'smiles': 'smiles_A'})
        
        # --- FIX: Apply the strict validation and rooting map ---
        # Any invalid molecule becomes `None`, then we drop the Nones.
        df['smiles_A'] = df['smiles_A'].apply(self.validate_and_root_molecule)
        df_filtered = df.dropna(subset=['smiles_A']).reset_index(drop=True)
        
        if df_filtered.empty:
            print("Warning: No valid mono-acrylates found after structural filtering.")
            return df, df_filtered

        # Load GNN for property prediction
        self.gnn_model = self.load_gnn_model()

        # Run GNN only on the structurally valid molecules
        df_results = self.fine_tunning_with_gnn(df_filtered)
        
        # Apply property error threshold
        df_final, df_final_filtered = self.apply_error_threshold(df_results)

        return df_final, df_final_filtered

    def run(self, targets):
        data = []

        for i in tqdm(targets):
            for j in range(self.monomers_number_per_target):
                smiles = self.generate_one(i)
                if smiles:
                    data.append({'smiles': smiles, 'static_polarizability': i})

        df = pd.DataFrame(data)
        print(f'Original data size: {len(df)}')

        if df.empty:
            print("No valid SMILES generated.")
            return df

        df, df_filtered = self.post_processing(df)
        print(f'Filtered data size: {len(df_filtered)}')

        df.to_csv(os.path.join(self.output_path, 'generated_molecules.csv'), index=False)

        if len(df_filtered) == 0:
            print('Filtered dataframe has 0 length. Consider changing your threshold.')
        else:
            df_filtered.to_csv(os.path.join(self.output_path, 'generated_molecules_filtered.csv'), index=False)
        
        return df