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
    """Prepares data for training a GPT-based generative model.

    This class reads a CSV of molecular data, filters it to include only monomers,
    converts SMILES to SELFIES, and standardizes the target property. The
    processed data and the scaler are saved for later use.

    :param input_csv: Path to the input CSV file containing molecular data.
    :type input_csv: str
    :param output_path: Directory to save the processed data and scaler.
                        Defaults to 'polygraphpy/data/generative_data/'.
    :type output_path: str, optional
    """
    def __init__(self, input_csv, output_path='polygraphpy/data/generative_data/'):
        self.input_csv = input_csv
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        """Executes the data pre-processing pipeline.

        It reads the input data, filters monomers, standardizes the target,
        encodes SMILES to SELFIES, and saves the results to CSV and pickle files.

        :return: The path to the output directory.
        :rtype: str
        """
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
    """A PyTorch Dataset for SELFIES strings.

    This class tokenizes SELFIES strings using a GPT tokenizer and prepares
    them for use with a PyTorch DataLoader.

    :param texts: A list of text strings (e.g., "polarizability: X selfies: Y").
    :type texts: list
    :param tokenizer: The GPT tokenizer instance.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param max_len: The maximum sequence length for tokenization. Defaults to 128.
    :type max_len: int, optional
    """
    def __init__(self, texts, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset."""
        return {k: v[idx] for k, v in self.encodings.items()}

class GenerativeTrainer:
    """Fine-tunes a GPT-2 model for molecular generation.

    This class loads a pre-processed dataset, initializes a GPT-2 model,
    and trains it to generate SELFIES strings based on a given polarizability
    target. The fine-tuned model is saved to disk.

    :param data_path: Path to the directory containing the pre-processed data.
    :type data_path: str
    :param model_output_path: Directory to save the trained model.
    :type model_output_path: str
    :param batch_size: Number of samples per training batch. Defaults to 4.
    :type batch_size: int, optional
    :param learning_rate: Learning rate for the optimizer. Defaults to 5e-5.
    :type learning_rate: float, optional
    :param epochs: Number of training epochs. Defaults to 100.
    :type epochs: int, optional
    """
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
        """Executes the model training process.

        If a model already exists, it skips training. Otherwise, it loads the
        data, initializes the tokenizer and model, runs the training loop,
        and saves the best model based on loss.

        :return: The path to the trained model directory.
        :rtype: str
        """
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
    """Generates and validates new molecules using a trained GPT model and a GNN.

    This class takes a list of target polarizability values, uses the GPT model
    to generate corresponding SELFIES strings, and then filters the generated
    molecules. Filtering involves checking for valid SMILES, structural properties,
    and using a GNN model to predict the polarizability and compare it to the
    target, keeping only those within a specified error threshold.

    :param model_path: Directory containing the trained GPT model.
    :type model_path: str
    :param output_path: Directory to save the generated molecules.
    :type output_path: str
    :param monomers_number_per_target: Number of molecules to generate for each target value.
    :type monomers_number_per_target: int
    :param threshold: The maximum allowed relative error between the GNN prediction
                      and the target polarizability for a generated molecule to be kept.
    :type threshold: float
    """
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

    def generate_one(self, pol_val, max_len=1000):
        """Generates a single molecule (as SMILES) based on a target polarizability.

        :param pol_val: The target polarizability value (scaled).
        :type pol_val: float
        :param max_len: Maximum length of the generated SELFIES string.
        :type max_len: int, optional
        :return: A valid SMILES string or None if generation fails.
        :rtype: str or None
        """
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
    
    def is_mixed_neutral_smiles(self, smiles: str) -> bool:
        """Checks if a SMILES string represents a molecule with multiple neutral fragments.

        :param smiles: The SMILES string to check.
        :type smiles: str
        :return: True if the molecule has multiple neutral fragments, False otherwise.
        :rtype: bool
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return False
        
        fragments = Chem.GetMolFrags(mol, asMols=True)
        
        if len(fragments) == 1:
            return False
        
        all_neutral = all(sum(atom.GetFormalCharge() for atom in frag.GetAtoms()) == 0 for frag in fragments)
        
        return all_neutral

    def is_acrylate(self, smi):
        """Checks if a SMILES string contains an acrylate substructure.

        :param smi: The SMILES string to check.
        :type smi: str
        :return: True if the molecule contains an acrylate group, False otherwise.
        :rtype: bool
        """
        acrylate_smarts = "C=C[C](=O)O"
        acrylate_mol = Chem.MolFromSmarts(acrylate_smarts)
        mol = Chem.MolFromSmiles(smi)

        return mol.HasSubstructMatch(acrylate_mol) if mol else False
    
    def mol_to_data(self, smiles):
        """Converts a SMILES string into a PyTorch Geometric `Data` object.

        This method is used to prepare generated molecules for GNN prediction.

        :param smiles: The SMILES string of the molecule.
        :type smiles: str
        :return: A `Data` object or None if the SMILES is invalid.
        :rtype: Data or None
        """
        atoms = []
        bonds = []
        m1 = Chem.MolFromSmiles(smiles, sanitize=True)
        if m1 is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        m1 = Chem.AddHs(m1)
        
        atoms = self.preprocess.get_nodes_information(m1, [], chain_size=0)

        if not atoms:
            print(f"No atoms extracted for SMILES: {smiles}")
            return None
        
        df_nodes = pd.DataFrame(atoms)
        nodes_features = pd.DataFrame(self.atom_encoder.transform(df_nodes.drop(['idx'], axis=1)).toarray())
        x = torch.tensor(nodes_features.astype('float32').values)
        
        bonds = self.preprocess.get_bonds_information(m1, [])
        if not bonds:
            print(f"No bonds extracted for SMILES: {smiles}")
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
        mol_data.validate()

        return mol_data
    
    def load_gnn_model(self):
        """Loads a pre-trained GNN model and its encoders for validation.

        :return: The loaded GNN model.
        :rtype: torch.nn.Module
        """
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
        """Validates generated molecules by predicting their polarizability with a GNN.

        :param df: DataFrame of generated molecules.
        :type df: pd.DataFrame
        :return: A DataFrame with GNN predictions added.
        :rtype: pd.DataFrame
        """
        df_results = pd.DataFrame()

        for i in df.itertuples():
            mol_data = self.mol_to_data(i.smiles_A).to(self.device)

            with torch.no_grad():
                batch = torch.zeros(mol_data.x.size(0), dtype=torch.long).to(self.device)
                pred = self.gnn_model(mol_data.x, mol_data.edge_index, mol_data.edge_weight, batch)

            df_results = pd.concat([df_results, pd.DataFrame({'smiles_A': i.smiles_A, 
                                                            'static_polarizability': i.static_polarizability, 
                                                            'static_polarizability_pred': pred.cpu().numpy()[0][0]}, index=[0])]).reset_index(drop=True)
            
        return df_results
    
    def apply_error_threshold(self, df: pd.DataFrame):
        """Filters generated molecules based on the GNN prediction error.

        This method compares the GNN-predicted polarizability with the target
        value and keeps only those within the specified error threshold.

        :param df: DataFrame with generated molecules and GNN predictions.
        :type df: pd.DataFrame
        :return: A tuple containing the original and the filtered DataFrames.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        with open(f'polygraphpy/data/generative_data/scaler.pkl', 'rb') as file:
            loaded_encoder : OneHotEncoder = pickle.load(file)
        
        df['static_polarizability_original'] = loaded_encoder.inverse_transform(df['static_polarizability'].values.reshape(-1,1))
        df['static_polarizability_pred_original'] = loaded_encoder.inverse_transform(df['static_polarizability_pred'].values.reshape(-1,1))

        df['error'] = abs(df['static_polarizability_original'] - df['static_polarizability_pred_original'])/df['static_polarizability_original']

        df_filtered = df[df['error'] <= self.threshold]

        return df, df_filtered
    
    def post_processing(self, df: pd.DataFrame):
        """Performs a series of post-processing and validation steps.

        This includes removing duplicates, checking for structural properties,
        and using the GNN to validate the generated molecules.

        :param df: DataFrame of generated molecules.
        :type df: pd.DataFrame
        :return: A tuple containing the original and filtered DataFrames.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        print('Making post processing with GNN prediction model.')

        df = df.drop_duplicates(subset='smiles').reset_index(drop=True)
        df = df.rename(columns={'smiles': 'smiles_A'})
        df['chain_size'] = 0
        df['id_A'] = df.index

        df_filtered = df[~df["smiles_A"].apply(self.is_mixed_neutral_smiles)].reset_index(drop=True)
        df_filtered["is_acrylate"] = df_filtered["smiles_A"].apply(self.is_acrylate)

        self.gnn_model = self.load_gnn_model()

        df = self.fine_tunning_with_gnn(df_filtered)
        df, df_filtered = self.apply_error_threshold(df)

        return df, df_filtered

    def run(self, targets):
        """Orchestrates the entire generation, validation, and saving process.

        :param targets: A list of target polarizability values (scaled) to generate molecules for.
        :type targets: list
        :return: The DataFrame of generated molecules.
        :rtype: pd.DataFrame
        """
        data = []

        for i in tqdm(targets):
            for j in range(self.monomers_number_per_target):
                smiles = self.generate_one(i)
                if smiles:
                    data.append({'smiles': smiles, 'static_polarizability': i})

        df = pd.DataFrame(data)
        print(f'Original data size: {len(df)}')

        df, df_filtered = self.post_processing(df)
        print(f'Filtered data size: {len(df_filtered)}')

        df.to_csv(os.path.join(self.output_path, 'generated_molecules.csv'), index=False)

        if len(df_filtered) == 0:
            print('Filtered dataframe has 0 length. Consider change your threshold.')
        else:
            df_filtered.to_csv(os.path.join(self.output_path, 'generated_molecules_filtered.csv'), index=False)
        
        return df