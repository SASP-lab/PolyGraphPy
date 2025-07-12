import os
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch_geometric.data import Data

class PreProcess():
    def __init__(self, input_csv: str = None, train_input_data_path: str = None,
                 polymer_type: str = None, target: str = None):
        self.input_csv = input_csv
        self.train_input_data_path = train_input_data_path
        self.polymer_type = polymer_type
        self.target = target
        self.scaler = MinMaxScaler()

        print('Reading GNN input file.')
        self.df = pd.read_csv(self.input_csv)

    def remove_outliers(self):
        Q1 = self.df[self.target].quantile(0.25)
        Q3 = self.df[self.target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[self.target] >= lower_bound) & (self.df[self.target] <= upper_bound)].reset_index(drop=True)

    def data_standardization(self):
        self.df[self.target + '_original'] = self.df[self.target]
        self.scaler = self.scaler.fit(self.df[[self.target]])
        self.df[self.target] = self.scaler.transform(self.df[[self.target]])

        # save the scaled data
        self.df.to_csv(self.input_csv)
    
    def extract_atoms_and_bonds_features_from_monomer_smiles(self) -> tuple[list, list]:
        print("Extracting unique features from atoms and bonds.")

        smiles_vec = self.df['smiles'].to_list()
        chain_vec = self.df['chain_size'].to_list()
        atoms_list = []
        bonds_list = []

        i = 0

        for smiles in tqdm(smiles_vec):
            m = Chem.MolFromSmiles(smiles)
            m = Chem.AddHs(m)
            
            for atom in m.GetAtoms():
                atoms_list.append({
                    'symbol': atom.GetSymbol(),
                    'atomic_num': atom.GetAtomicNum(),
                    'degree': atom.GetDegree(),
                    'mass': atom.GetMass(),
                    'hybridization': atom.GetHybridization(),
                    'radical_total_degree': atom.GetTotalDegree(),
                    'radical_total_valence': atom.GetTotalValence(),
                    'aromatic': int(atom.GetIsAromatic()),
                    'n_Hs': atom.GetTotalNumHs(),
                    'formal_charge': atom.GetFormalCharge(),
                    'chain_size': chain_vec[i]
                })
            
            i = i + 1

            for bond in m.GetBonds():
                bonds_list.append({
                    'type': bond.GetBondType().name,
                    'is_conjugated': bond.GetIsConjugated(),
                    'is_aromatic': bond.GetIsAromatic(),
                })

        return atoms_list, bonds_list
    
    def make_encoder(self, df_features: pd.DataFrame) -> OneHotEncoder:
        print("Making feature encoder.")

        encoder = OneHotEncoder()

        encoder.fit(df_features)

        return encoder
    
    def get_nodes_information(self, molecule: Chem.rdchem.Mol, atoms: list, chain_size: int) -> list:
        for atom in molecule.GetAtoms():
            symbol = atom.GetSymbol()

            atoms.append({
                    'idx': atom.GetIdx(),
                    'symbol': symbol,
                    'atomic_num': atom.GetAtomicNum(),
                    'degree': atom.GetDegree(),
                    'mass': atom.GetMass(),
                    'hybridization': atom.GetHybridization(),
                    'radical_total_degree': atom.GetTotalDegree(),
                    'radical_total_valence': atom.GetTotalValence(),
                    'aromatic': int(atom.GetIsAromatic()),
                    'n_Hs': atom.GetTotalNumHs(),
                    'formal_charge': atom.GetFormalCharge(),
                    'chain_size': chain_size,
                })

        return atoms

    def get_bonds_information(self, molecule: Chem.rdchem.Mol, bonds: list, dim:int = 0) -> list:
        for bond in molecule.GetBonds():
            bonds.append({
                'begin_idx': bond.GetBeginAtomIdx() + dim,
                'end_idx': bond.GetEndAtomIdx() + dim,
                'type': bond.GetBondType().name,
                'is_conjugated': bond.GetIsConjugated(),
                'is_aromatic': bond.GetIsAromatic(),
                'weight': 1.0,
            })

        return bonds
    
    def prepare_monomer_input_data(self, atom_encoder: OneHotEncoder, bond_encoder: OneHotEncoder):
        print(f'Training data preparation starting. {len(self.df)} to go.')
        for row in tqdm(self.df.itertuples()):
            atoms = []
            bonds = []
            
            m1 = Chem.MolFromSmiles(row.smiles)
            m1 = Chem.AddHs(m1)
            
            atoms = self.get_nodes_information(m1, atoms, row.chain_size)
            
            df_nodes = pd.DataFrame(atoms)
            nodes_features = pd.DataFrame(atom_encoder.transform(df_nodes.drop(['idx'], axis=1)).toarray())
            x = torch.tensor(nodes_features.astype('float32').values)
            
            bonds = self.get_bonds_information(m1, bonds)
            df_bonds = pd.DataFrame(bonds)
            
            connectivity = [df_bonds.begin_idx.to_list() + df_bonds.end_idx.to_list(), df_bonds.end_idx.to_list() + df_bonds.begin_idx.to_list()]
            edge_index = torch.tensor(connectivity)
            
            edge_attributes = df_bonds[['type', 'is_conjugated', 'is_aromatic']]
            edge_attributes = pd.concat([edge_attributes,edge_attributes.sort_index(ascending=False)]).reset_index(drop=True)
            edge_attr = torch.Tensor(pd.DataFrame(bond_encoder.transform(edge_attributes).toarray()).values)
            
            edge_weight = df_bonds[['weight']]
            edge_weight = pd.concat([edge_weight,edge_weight.sort_index(ascending=False)]).reset_index(drop=True)
            edge_weight = torch.tensor(edge_weight['weight'].astype('float32').values)
            
            y = torch.Tensor([row.__getattribute__(self.target)])
            mol_id = torch.Tensor([row.id])
            chain_size = torch.Tensor([row.chain_size])
            
            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_weight=edge_weight, mol_id=mol_id, chain_size=chain_size)
            mol_data.validate()
            
            torch.save(mol_data, f'{self.train_input_data_path}/{row.id}_{row.chain_size}.pt')
    
        print(f'Training data preparation finished.')

    def run(self):
        print('Removing outliers...')
        self.remove_outliers()
        print('Making data standardization...')
        self.data_standardization()

        if (len(os.listdir('polygraphpy/data/training_input_data/')) == len(self.df)):
            print(f'No training data preparation needed. Jumping to training.')

        else:
            atoms_list, bonds_list = self.extract_atoms_and_bonds_features_from_monomer_smiles()
            unique_atoms_features = pd.DataFrame(atoms_list).drop_duplicates().reset_index(drop=True)
            unique_bonds_features = pd.DataFrame(bonds_list).drop_duplicates().reset_index(drop=True)
            self.df['weights'] = 1

            atom_encoder = self.make_encoder(unique_atoms_features)
            bond_encoder = self.make_encoder(unique_bonds_features)
        
            self.prepare_monomer_input_data(atom_encoder, bond_encoder)
        
        return self.df