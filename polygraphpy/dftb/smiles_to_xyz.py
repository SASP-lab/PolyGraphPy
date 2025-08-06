import pandas as pd
import os
import logging
import stk
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from joblib import Parallel, delayed

from polygraphpy.utils.make_dummy_atom import replace_first_acrylate_cce

# Set up logging
logging.getLogger('rdkit').setLevel(logging.ERROR)
logging.basicConfig(filename='xyz_generation_errors.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class XyzGeneratorBase:
    """Base class for generating .xyz files from molecular structures."""
    
    def __init__(self, output_dir: str = 'polygraphpy/data/xyz_files'):
        """Initialize with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def write_xyz_file(self, mol: Chem.Mol, filename: str) -> None:
        """Write RDKit molecule to .xyz file."""
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        with open(filename+'.xyz', 'w') as f:
            f.write(f"{num_atoms}\n")
            f.write(f"Molecule ID: {os.path.basename(filename).split('.')[0]}\n")
            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                symbol = atom.GetSymbol()
                f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    
    def write_pdb_file(self, mol: Chem.Mol, filename: str) -> None:
        """Write RDKit molecule to .pdb file."""
        AllChem.MolToPDBFile(mol, filename+'.pdb')

class MonomerXyzGenerator(XyzGeneratorBase):
    """Generate .xyz files for monomers from SMILES strings."""
    
    def __init__(self, input_csv: str, output_dir: str = 'polygraphpy/data/xyz_files'):
        """Initialize with input CSV and output directory."""
        super().__init__(output_dir)
        self.df = pd.read_csv(input_csv)
    
    def process_row(self, row: pd.Series) -> str:
        """Process a single molecule row to generate .xyz file."""
        mol_id = row['id']

        xyz_filename = os.path.join(self.output_dir, f"monomer_{mol_id}.xyz")
        if os.path.exists(xyz_filename):
            return
        
        sml = row['smiles']
        try:
            m = Chem.MolFromSmiles(sml, sanitize=True)
            if m is None:
                logging.error(f"Failed to create molecule for ID {mol_id} with SMILES {sml}")
                return f"Skipping ID {mol_id}: Invalid SMILES"
            
            m_h = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxIterations = 1000
            params.numThreads = 1
            params.randomSeed = 42
            
            if AllChem.EmbedMolecule(m_h, params) == -1:
                logging.error(f"Embedding failed for ID {mol_id} with SMILES {sml}")
                return f"Skipping ID {mol_id}: Embedding failed"
            
            xyz_filename = os.path.join(self.output_dir, f"monomer_{mol_id}")
            self.write_xyz_file(m_h, xyz_filename)
            return f"Saved monomer: {xyz_filename}"
        
        except Exception as e:
            logging.error(f"Error processing ID {mol_id} with SMILES {sml}: {str(e)}")
            return f"Skipping ID {mol_id}: Exception occurred - {str(e)}"
    
    def generate(self) -> list:
        """Generate .xyz files for all monomers in parallel."""
        print("Generating .xyz files for monomers...")
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(self.process_row)(row) for _, row in tqdm(self.df.iterrows(), total=len(self.df))
        )
        return results

class PolymerXyzGenerator(XyzGeneratorBase):
    """Generate .xyz files for homopolymers from acrylate monomers."""
    
    def __init__(self, input_csv: str, output_dir: str = 'polygraphpy/data/xyz_files', polymer_chain_size: int = 2, polymer_type: str = 'homopolymer'):
        """Initialize with input CSV and output directory."""
        super().__init__(output_dir)
        self.df = pd.read_csv(input_csv)
        self.polymer_chain_size = polymer_chain_size
        self.polymer_type = polymer_type

        atoms_number = []
        for i in self.df['smiles'].values:
            mol = Chem.MolFromSmiles(i)
            mol_with_hs = Chem.AddHs(mol)
            num_all_atoms = mol_with_hs.GetNumAtoms()

            atoms_number.append(num_all_atoms)
        
        self.df['number_of_atoms'] = atoms_number
    
    def is_acrylate(self, smiles: str) -> bool:
        """Check if a SMILES string represents an acrylate."""
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol.HasSubstructMatch(Chem.MolFromSmarts('C=C-C(=O)O'))
    
    def build_and_save_polymer(self, smiles_A: str = None, smiles_B: str = None, mol_id_A: str = None, mol_id_B: str = None) -> str:
        """Build homopolymer or copolymer and save .xyz file."""
        
        if self.polymer_type == 'copolymer':
            xyz_filename = os.path.join(self.output_dir, f"copoly_{mol_id_A}_{mol_id_B}_chain_{self.polymer_chain_size}.xyz")
        else:
            xyz_filename = os.path.join(self.output_dir, f"homopoly_{mol_id_A}_chain_{self.polymer_chain_size}.xyz")
        if os.path.exists(xyz_filename):
            return
        
        try:
            if not self.is_acrylate(smiles_A):
                logging.warning(f"ID {mol_id_A} with SMILES {smiles_A} is not an acrylate")
                return f"Skipping ID {mol_id_A}: Not an acrylate"
            
            contains_br_A = smiles_A.__contains__('Br')
            smiles_A = replace_first_acrylate_cce(smiles_A, contains_br_A)

            if self.polymer_type == 'copolymer':
                contains_br_B = smiles_B.__contains__('Br')
                smiles_B = replace_first_acrylate_cce(smiles_B, contains_br_B)
            else:
                contains_br_B = contains_br_A
                smiles_B = smiles_A

            if not contains_br_A and not contains_br_B:
                bb1 = stk.BuildingBlock(smiles_A, [stk.BromoFactory()])
                bb2 = stk.BuildingBlock(smiles_B, [stk.BromoFactory()])
            else:
                bb1 = stk.BuildingBlock(smiles_A, [stk.IodoFactory()])
                bb2 = stk.BuildingBlock(smiles_B, [stk.IodoFactory()])
            
            polymer = stk.ConstructedMolecule(
                topology_graph=stk.polymer.Linear(
                    building_blocks=(bb1, bb2),
                    repeating_unit='AB',
                    num_repeating_units=self.polymer_chain_size,
                    optimizer=stk.MCHammer(
                        num_steps=4000,  # Increase steps for better optimization
                        target_bond_length=1.54,  # Target C-C single bond length (Å)
                        nonbond_sigma = 0.4,
                        random_seed=None
                    ),
                    orientations=[1, 0],
                ),
            )
            
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
            
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxIterations = 3000
            params.numThreads = 1
            params.randomSeed = np.random.randint(0, 10000)

            if AllChem.EmbedMolecule(rdkit_polymer, params) == -1:
                if self.polymer_type == 'copolymer':
                    logging.warning(f"Embedding failed for copolymer {mol_id_A}_{mol_id_B}")
                else:
                     logging.warning(f"Embedding failed for copolymer {mol_id_A}")
            
            AllChem.MMFFOptimizeMolecule(rdkit_polymer, maxIters=1000)
            polymer = polymer.with_position_matrix(
                position_matrix=rdkit_polymer.GetConformer().GetPositions()
            )
                    
            if self.polymer_type == 'copolymer':
                xyz_filename = os.path.join(self.output_dir, f"copoly_{mol_id_A}_{mol_id_B}_chain_{self.polymer_chain_size}")
            else:
                xyz_filename = os.path.join(self.output_dir, f"homopoly_{mol_id_A}_chain_{self.polymer_chain_size}")
            self.write_xyz_file(rdkit_polymer, xyz_filename)
            self.write_pdb_file(rdkit_polymer, xyz_filename)
            return f"Saved homopolymer: {xyz_filename}"
        
        except Exception as e:
            return f"Skipping ID {mol_id_A}: Exception occurred - {str(e)}"
    
    def rcb_partition(self, points, num_levels=4, points_per_subdomain=None):
        if num_levels is None and points_per_subdomain is None:
            num_levels = 3
        elif num_levels is not None and points_per_subdomain is not None:
            raise ValueError("Provide either num_levels or points_per_subdomain, not both")
        elif points_per_subdomain is not None:
            N = points.shape[0]
            num_subdomains = int(N / points_per_subdomain)
            num_levels = int(math.log2(num_subdomains))

        D = points.shape[1]
        N = points.shape[0]
        indices = np.arange(N)
        assignments = np.zeros(N, dtype=int)

        def recurse(ind, start_id, level, dim):
            if level == 0 or len(ind) <= 1:
                assignments[ind] = start_id
                return
            sorted_ind = ind[np.argsort(points[ind, dim])]
            mid = len(sorted_ind) // 2
            left = sorted_ind[:mid]
            right = sorted_ind[mid:]
            next_dim = (dim + 1) % D
            recurse(left, start_id, level - 1, next_dim)
            recurse(right, start_id + (1 << (level - 1)), level - 1, next_dim)

        recurse(indices, 0, num_levels, 0)
        return assignments
    
    def generate_copolymers(self, df: pd.DataFrame) -> pd.DataFrame:
        df_acrylates = pd.DataFrame()
        
        for i in range(len(df)):
            for j in range(i+1,len(df)):
                smiles_A = df.loc[i, 'smiles']
                smiles_B = df.loc[j, 'smiles']
                id_A = df.loc[i, 'id']
                id_B = df.loc[j, 'id']

                df_aux = pd.DataFrame({'smiles_A': smiles_A, 'smiles_B': smiles_B, 'id_A': id_A, 'id_B': id_B}, index=[0])

                df_acrylates = pd.concat([df_acrylates, df_aux]).reset_index(drop=True)
        
        return df_acrylates

    def has_metal(self, smiles):
        metallic_elements = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Al',
                             'Fe', 'Zn', 'Cu', 'Ag', 'Au', 'Ni', 'Co', 'Mn', 'Cr', 'Ti', 'V']

        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return any(atom.GetSymbol() in metallic_elements for atom in mol.GetAtoms())
            
        return False 
            
    def generate(self) -> list:
        """Generate .xyz files for homopolymers and copolymers from acrylate monomers."""
        print("Filtering dataset for acrylate monomers...")
        df_acrylates = self.df[self.df['smiles'].apply(self.is_acrylate)].copy()
        print(f"Found {len(df_acrylates)} acrylate monomers")
        
        if self.polymer_type == 'copolymer':
            print("Building copolymers in parallel...")

            atoms_limit = 25
            print(f"Filtering molecules by number of atoms. Condition: number of atoms <= {atoms_limit}")
            print(f"Original size: {len(df_acrylates)}")
            df_acrylates = df_acrylates[df_acrylates['number_of_atoms'] <= atoms_limit].reset_index(drop=True)
            print(f"Filtered size: {len(df_acrylates)}")

            print("Filtering out metallic molecules...")
            df_acrylates = df_acrylates[~df_acrylates['smiles'].apply(self.has_metal)]

            print("Creating partitions using RCB method...")
            coords = df_acrylates[['mw', 'complexity', 'polararea', 'xlogp', 'rotbonds', 'heavycnt']].values
            num_levels = 4
            assignments = self.rcb_partition(coords, num_levels=num_levels)
            df_acrylates['cluster'] = assignments

            print("Sampling from partitions...")
            number_of_samples = 9 # number of samples per cluster
            sampled_df = df_acrylates.groupby('cluster').apply(lambda x: x.sample(n=number_of_samples), include_groups=False).reset_index(drop=True)
            n = len(sampled_df)
            print(f"{int((n*(n-1))/2)} possible copolymers to be created...")

            df_acrylates = self.generate_copolymers(sampled_df)
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(self.build_and_save_polymer)(row['smiles_A'], row['smiles_B'], row['id_A'], row['id_B'])
                for _, row in tqdm(df_acrylates.iterrows(), total=len(df_acrylates))
            )

        else:
            print("Building homopolymers in parallel...")
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(self.build_and_save_polymer)(row['smiles'], None, row['id'], None)
                for _, row in tqdm(df_acrylates.iterrows(), total=len(df_acrylates))
            )

        return results