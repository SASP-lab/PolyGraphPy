import os
import pandas as pd
import torch
import stk
from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch_geometric.data import Data
from joblib import Parallel, delayed

from polygraphpy.utils.make_dummy_atom import replace_first_acrylate_cce

class PreProcess():
    def __init__(self, input_csv: str = None, train_input_data_path: str = None,
                 polymer_type: str = None, target: str = None, gnn_output_path: str = None):
        self.input_csv = input_csv
        self.train_input_data_path = train_input_data_path
        self.polymer_type = polymer_type
        self.target = target
        self.gnn_output_path = gnn_output_path
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

    def get_small_group_of_smiles_A(self):
        df_grouped = self.df.groupby('smiles_A').count().sort_values(by='id_A').reset_index()
        df_grouped = df_grouped[df_grouped['id_A'] >= 30]

        self.df = self.df[self.df['smiles_A'].isin(df_grouped['smiles_A'])]

        Q1 = self.df['static_polarizability'].quantile(0.15)
        Q3 = self.df['static_polarizability'].quantile(0.85)
        df_filtered = self.df[(self.df['static_polarizability'] >= Q1) & (self.df['static_polarizability'] <= Q3)].reset_index(drop=True)

        self.df = df_filtered

    def data_standardization(self):
        self.df[self.target + '_original'] = self.df[self.target]
        self.scaler = self.scaler.fit(self.df[[self.target]])
        self.df[self.target] = self.scaler.transform(self.df[[self.target]])

        # save the scaled data
        aux = ''
        if (self.polymer_type == 'copolymer'):
            aux = '_copoly'
        self.df.to_csv(f'{self.gnn_output_path}scaled_output{aux}.csv', index=False)
    
    def extract_atoms_and_bonds_features_from_monomer_smiles(self) -> tuple[list, list]:
        print("Extracting unique features from atoms and bonds.")

        df_aux = self.df.groupby(['smiles_A', 'chain_size']).count().reset_index()[['smiles_A', 'chain_size']]
        smiles_vec = df_aux['smiles_A'].to_list()
        chain_vec = df_aux['chain_size'].to_list()

        if self.polymer_type == 'copolymer':
            df_aux = self.df.groupby(['smiles_B', 'chain_size']).count().reset_index()[['smiles_B', 'chain_size']]
            smiles_vec = smiles_vec + df_aux['smiles_B'].to_list()
            chain_vec = chain_vec + df_aux['chain_size'].to_list()

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
                    'radical_total_degree': atom.GetTotalDegree(),
                    'radical_total_valence': atom.GetTotalValence(),
                    'aromatic': int(atom.GetIsAromatic()),
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
                    'radical_total_degree': atom.GetTotalDegree(),
                    'radical_total_valence': atom.GetTotalValence(),
                    'aromatic': int(atom.GetIsAromatic()),
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

    def get_building_block(self, smiles, contains_br: bool):
        if contains_br:
            bb = stk.BuildingBlock(smiles, [stk.IodoFactory()])
        else:
            bb = stk.BuildingBlock(smiles, [stk.BromoFactory()])
        
        return bb

    def build_molecule(self, smiles_A: str, smiles_B: str):
        contains_br_A = smiles_A.__contains__('Br')
        smiles_A = replace_first_acrylate_cce(smiles_A, contains_br_A)

        contains_br_B = smiles_B.__contains__('Br')
        smiles_B = replace_first_acrylate_cce(smiles_B, contains_br_B)

        bb1 = self.get_building_block(smiles_A, contains_br_A)
        bb2 = self.get_building_block(smiles_B, contains_br_B)
        
        polymer = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(bb1, bb2),
                repeating_unit='AB',
                num_repeating_units=1,
                optimizer=stk.MCHammer(
                    num_steps=3,
                    target_bond_length=1.54,
                    nonbond_sigma = 0.4,
                    random_seed=None
                ),
                orientations=[1, 0],
            ),
        )

        # Get bonder atom IDs from building blocks
        bb1_bonders = {id for bb in [bb1] for fg in bb.get_functional_groups() for id in fg.get_bonder_ids()}
        bb2_bonders = {id for bb in [bb2] for fg in bb.get_functional_groups() for id in fg.get_bonder_ids()}

        # Map bonder atom IDs to polymer
        bb1_polymer_bonders = {info.get_atom().get_id() for info in polymer.get_atom_infos() if info.get_building_block() is bb1 and info.get_building_block_atom().get_id() in bb1_bonders}
        bb2_polymer_bonders = {info.get_atom().get_id() for info in polymer.get_atom_infos() if info.get_building_block() is bb2 and info.get_building_block_atom().get_id() in bb2_bonders}

        # Find STK-created bond
        for bond in polymer.get_bonds():
            a1, a2 = bond.get_atom1().get_id(), bond.get_atom2().get_id()
            if (a1 in bb1_polymer_bonders and a2 in bb2_polymer_bonders) or (a1 in bb2_polymer_bonders and a2 in bb1_polymer_bonders):
                atom1 = a1
                atom2 = a2

        rdkit_polymer = polymer.to_rdkit_mol()
        rdkit_polymer = Chem.AddHs(rdkit_polymer)
        Chem.SanitizeMol(rdkit_polymer)
        
        rw_mol = Chem.RWMol(rdkit_polymer)
        if not contains_br_A and not contains_br_B:
            atoms_to_replace = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == 'Br']
        else:
            atoms_to_replace = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == 'I']
        for idx in sorted(atoms_to_replace, reverse=True):
            rw_mol.ReplaceAtom(idx, Chem.Atom('H'))
        rdkit_polymer = rw_mol.GetMol()
        Chem.SanitizeMol(rdkit_polymer)

        return rdkit_polymer, atom1, atom2
    
    def prepare_copolymer_input_data(self, atom_encoder: OneHotEncoder, bond_encoder: OneHotEncoder):
        print(f'Starting copolymer data preparation. {len(self.df)} to go.')

        def process_row(row):
            atoms = []
            bonds = []

            smiles_A = row.smiles_A
            smiles_B = row.smiles_B

            copoly, conn_atom1, conn_atom2 = self.build_molecule(smiles_A, smiles_B)

            atoms = self.get_nodes_information(copoly, atoms, row.chain_size)
            
            df_nodes = pd.DataFrame(atoms)
            nodes_features = pd.DataFrame(atom_encoder.transform(df_nodes.drop(['idx'], axis=1)).toarray())
            x = torch.tensor(nodes_features.astype('float32').values)

            bonds = self.get_bonds_information(copoly, bonds)
            df_bonds = pd.DataFrame(bonds)
            df_bonds.loc[df_bonds[['begin_idx', 'end_idx']].eq([conn_atom1, conn_atom2]).all(axis=1), 'weight'] = 0.5

            connectivity = [df_bonds.begin_idx.to_list() + df_bonds.end_idx.to_list(), df_bonds.end_idx.to_list() + df_bonds.begin_idx.to_list()]
            edge_index = torch.tensor(connectivity)
            
            edge_attributes = df_bonds[['type', 'is_conjugated', 'is_aromatic']]
            edge_attributes = pd.concat([edge_attributes, edge_attributes.sort_index(ascending=False)]).reset_index(drop=True)
            edge_attr = torch.Tensor(pd.DataFrame(bond_encoder.transform(edge_attributes).toarray()).values)
            
            edge_weight = df_bonds[['weight']]
            edge_weight = pd.concat([edge_weight, edge_weight.sort_index(ascending=False)]).reset_index(drop=True)
            edge_weight = torch.tensor(edge_weight['weight'].astype('float32').values)

            y = torch.Tensor([row.__getattribute__(self.target)])
            id_A = torch.Tensor([row.id_A])
            id_B = torch.Tensor([row.id_B])
            chain_size = torch.Tensor([row.chain_size])
            
            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_weight=edge_weight, id_A=id_A, id_B=id_B, chain_size=chain_size)
            mol_data.validate()

            torch.save(mol_data, f'{self.train_input_data_path}{row.id_A}_{row.id_B}_{row.chain_size}.pt')

        Parallel(n_jobs=-1)(delayed(process_row)(row) for row in tqdm(self.df.itertuples()))

        print(f'Training data preparation finished.')
    
    def prepare_monomer_input_data(self, atom_encoder: OneHotEncoder, bond_encoder: OneHotEncoder):
        print(f'Training data preparation starting. {len(self.df)} to go.')
        for row in tqdm(self.df.itertuples()):
            atoms = []
            bonds = []
            
            m1 = Chem.MolFromSmiles(row.smiles_A)
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
            mol_id = torch.Tensor([row.id_A])
            chain_size = torch.Tensor([row.chain_size])
            
            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_weight=edge_weight, mol_id=mol_id, chain_size=chain_size)
            mol_data.validate()
            
            torch.save(mol_data, f'{self.train_input_data_path}{row.id_A}_{row.chain_size}.pt')
    
        print(f'Training data preparation finished.')

    def run(self):
        print('Removing outliers...')
        self.remove_outliers()
        print('Making data standardization...')
        self.data_standardization()

        if (len(os.listdir(self.train_input_data_path)) == len(self.df)):
            print(f'No training data preparation needed. Jumping to training.')

        else:
            atoms_list, bonds_list = self.extract_atoms_and_bonds_features_from_monomer_smiles()
            unique_atoms_features = pd.DataFrame(atoms_list).drop_duplicates().reset_index(drop=True)
            unique_bonds_features = pd.DataFrame(bonds_list).drop_duplicates().reset_index(drop=True)
            self.df['weights'] = 1

            atom_encoder = self.make_encoder(unique_atoms_features)
            bond_encoder = self.make_encoder(unique_bonds_features)
        
            if self.polymer_type != 'copolymer':
                self.prepare_monomer_input_data(atom_encoder, bond_encoder)
            else:
                self.get_small_group_of_smiles_A()
                self.prepare_copolymer_input_data(atom_encoder, bond_encoder)
        
        return self.df