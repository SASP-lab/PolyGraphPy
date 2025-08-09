import pandas as pd
import torch
import random
import numpy as np
import os
from polygraphpy.gnn.pre_processing import PreProcess
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Batch, Data
from joblib import Parallel, delayed
from tqdm import tqdm

class GaModelLoader:
    def __init__(self, input_csv, gnn_output_path, train_input_data_path, polymer_type, prediction_target):

        self.preprocess = PreProcess(input_csv=input_csv, train_input_data_path=train_input_data_path,
                                     polymer_type=polymer_type, target=prediction_target, gnn_output_path=gnn_output_path)
        
        self.df = self.preprocess.run()

        atoms_list, bonds_list = self.preprocess.extract_atoms_and_bonds_features_from_monomer_smiles()

        self.atom_encoder = self.preprocess.make_encoder(pd.DataFrame(atoms_list).drop_duplicates().reset_index(drop=True))
        self.bond_encoder = self.preprocess.make_encoder(pd.DataFrame(bonds_list).drop_duplicates().reset_index(drop=True))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torch.load(os.path.join(gnn_output_path, 'model_gcn.pt'), weights_only=False, map_location=self.device)
        print('GNN model:')
        print(self.model)

    def get_components(self):
        return self.model, self.preprocess, self.atom_encoder, self.bond_encoder

class FragmentGA:
    def __init__(self, csv_path, model, preprocess, atom_encoder, bond_encoder, population_size=30, 
                 prediction_target='static_polarizability', target_polarizability=0.43):
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['chain_size'] == 0]
        self.target_value = target_polarizability
        self.target_column = prediction_target

        self._pre_process()

        self.model = model.eval()
        self.preprocess = preprocess
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        self.population_size = population_size
        self.fragments = self._extract_fragments()

        print(f'Fragments size: {len(self.fragments)}')
        print("Fragments sample: ")
        print(self.fragments[:15])

        self.device = next(model.parameters()).device

    def _pre_process(self):
        scaler = MinMaxScaler()
        self.df['target_scaled'] = scaler.fit_transform(self.df[self.target_column].values.reshape(-1,1))

        atoms_number = []

        for i in self.df['smiles_A'].values:
            mol = Chem.MolFromSmiles(i)
            mol_with_hs = Chem.AddHs(mol)
            num_all_atoms = mol_with_hs.GetNumAtoms()
            atoms_number.append(num_all_atoms)

        self.df['number_of_atoms'] = atoms_number

        print(f'Datsaset original size: {len(self.df)}')
        self.df = self.df[self.target_value <= self.df['target_scaled']*1.20].reset_index(drop=True)
        self.df = self.df[self.target_value >= self.df['target_scaled']*0.80].reset_index(drop=True)
        self.df = self.df[self.df['number_of_atoms'] <= 40].reset_index(drop=True)
        print(f'Datsaset size after filtering process: {len(self.df)}')

    def _extract_fragments(self):
        all_frags = set()

        acrylate_core = Chem.MolFromSmarts('C=C-C(=O)O-[*]')
        for smi in tqdm(self.df['smiles_A']):
            mol = Chem.MolFromSmiles(smi, sanitize=True)

            if mol is None:
                continue
            try:
                Chem.RemoveStereochemistry(mol)
                if not mol.HasSubstructMatch(acrylate_core):
                    continue
                frags = BRICS.BRICSDecompose(mol, minFragmentSize=3, keepNonLeafNodes=True)
                for f in frags:
                    frag_mol = Chem.MolFromSmiles(f, sanitize=True)
                    if frag_mol and '*' in f and Descriptors.MolWt(frag_mol) < 200:
                        all_frags.add(f)
            except:
                continue

        fragments = list(all_frags)[:800]
        return fragments

    def _mol_to_data(self, smiles):
        try:
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
            edge_attr = torch.tensor(self.bond_encoder.transform(edge_attrs).toarray(), dtype=torch.float32)

            edge_weight = torch.tensor([1.0] * edge_index.shape[1], dtype=torch.float32)

            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
            mol_data.validate()

            return mol_data
        
        except Exception as e:
            print(f"Error in _mol_to_data for SMILES {smiles}: {str(e)}")
            return None

    def _evaluate_fitness_batch(self, smiles_list, target_polarizability):
        data_list = []
        valid_smiles = []

        for smi in smiles_list:
            data = self._mol_to_data(smi)
            if data is not None:
                data_list.append(data)
                valid_smiles.append(smi)

        if not data_list:
            return [(smi, -1.0) for smi in smiles_list]
        batch = Batch.from_data_list(data_list).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch.x, batch.edge_index, batch.edge_weight, batch.batch).cpu().numpy()
        scores = [-abs(pred - target_polarizability) if pred > 0 else -1.0 for pred in predictions]

        return list(zip(valid_smiles, scores)) + [(smi, -1.0) for smi in smiles_list if smi not in valid_smiles]

    def run_parallel(self, generations=10, target_polarizability=0.555):
        print("Generating initial population...")
        population = Parallel(n_jobs=-1, backend='loky')(delayed(build_molecule)(self.fragments) for _ in tqdm(range(self.population_size)))
        population = [p for p in population if p is not None]

        if not population:
            print("Initial population empty. Check fragment generation.")
            return []
        
        for gen in tqdm(range(generations)):
            fitness_scores = self._evaluate_fitness_batch(population, target_polarizability)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            top_individuals = fitness_scores[:self.population_size // 2]

            if len(top_individuals) == 0:
                print("No valid molecules in this generation. Reinitializing population...")
                population = Parallel(n_jobs=-1, backend='loky')(delayed(build_molecule)(self.fragments) for _ in range(self.population_size))
                population = [p for p in population if p is not None]
                continue

            top_frags = set()

            for smi, _ in top_individuals:
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                if mol is None:
                    continue
                try:
                    top_frags.update(BRICS.BRICSDecompose(mol, minFragmentSize=3))
                except:
                    continue

            original_fragments = self.fragments
            self.fragments = list(top_frags)[:500] if top_frags else original_fragments

            new_population = Parallel(n_jobs=-1, backend='loky')(delayed(build_molecule)(self.fragments) for _ in range(self.population_size))

            self.fragments = original_fragments
            population = [p for p in new_population if p is not None]

            if not population:
                print("Crossover failed to produce valid molecules. Reinitializing population...")
                population = Parallel(n_jobs=-1, backend='loky')(delayed(build_molecule)(self.fragments) for _ in range(self.population_size))
                population = [p for p in population if p is not None]

        return fitness_scores

def build_molecule(fragments):
    acrylate_core = Chem.MolFromSmiles('C=CC(=O)O*')

    for attempt in range(100):
        r_frags = random.sample(fragments, k=random.randint(1, 3))[:100]

        try:
            mol_frags = [Chem.MolFromSmiles(f, sanitize=True) for f in r_frags]
            mol_frags.append(acrylate_core)

            if None in mol_frags:
                continue

            new_mol = BRICS.BRICSBuild(mol_frags)

            for mol in new_mol:
                Chem.RemoveStereochemistry(mol)
                smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                mol = Chem.MolFromSmiles(smi, sanitize=True)

                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts('C=C-C(=O)O')):
                    Chem.SanitizeMol(mol)

                    return smi
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with fragments {r_frags}: {str(e)}")
            continue

    return None