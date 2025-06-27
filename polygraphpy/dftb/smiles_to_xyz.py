import pandas as pd
import os
import logging
import stk
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from joblib import Parallel, delayed

# Set up logging
logging.getLogger('rdkit').setLevel(logging.ERROR)
logging.basicConfig(filename='xyz_generation_errors.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class XyzGeneratorBase:
    """Base class for generating .xyz files from molecular structures."""
    
    def __init__(self, output_dir: str = 'polygraphpy/data/xyz_files'):
        """Initialize with output directory."""
        print(output_dir)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def write_xyz_file(self, mol: Chem.Mol, filename: str) -> None:
        """Write RDKit molecule to .xyz file."""
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        with open(filename, 'w') as f:
            f.write(f"{num_atoms}\n")
            f.write(f"Molecule ID: {os.path.basename(filename).split('.')[0]}\n")
            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                symbol = atom.GetSymbol()
                f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

class MonomerXyzGenerator(XyzGeneratorBase):
    """Generate .xyz files for monomers from SMILES strings."""
    
    def __init__(self, input_csv: str, output_dir: str = 'polygraphpy/data/xyz_files'):
        """Initialize with input CSV and output directory."""
        super().__init__(output_dir)
        self.df = pd.read_csv(input_csv)
    
    def process_row(self, row: pd.Series) -> str:
        """Process a single molecule row to generate .xyz file."""
        mol_id = row['id']
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
            
            xyz_filename = os.path.join(self.output_dir, f"monomer_{mol_id}.xyz")
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
    
    def __init__(self, input_csv: str, output_dir: str = 'polygraphpy/data/xyz_files', polymer_chain_size: int = 2):
        """Initialize with input CSV and output directory."""
        super().__init__(output_dir)
        self.df = pd.read_csv(input_csv)
        self.polymer_chain_size = polymer_chain_size
    
    def is_acrylate(self, smiles: str) -> bool:
        """Check if a SMILES string represents an acrylate."""
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol.HasSubstructMatch(Chem.MolFromSmarts('C=C-C(=O)O'))
    
    def replace_first_acrylate_cce(self, smiles: str, contains_br: bool) -> str:
        """Replace C=C in acrylate group with single bond and add Br atoms."""
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        # Flexible SMARTS pattern for acrylate group
        acrylate_pattern = Chem.MolFromSmarts('[C:1]=[C:2][C:3](=O)[O:4]')
        if not mol.HasSubstructMatch(acrylate_pattern):
            raise ValueError("No acrylate group found")

        # Get the first acrylate match
        matches = mol.GetSubstructMatches(acrylate_pattern)
        match = matches[0]
        c1_idx, c2_idx = match[0], match[1]  # Indices of [C:1]=[C:2]

        # Existing verification: Check the bond between c1_idx and c2_idx
        bond = mol.GetBondBetweenAtoms(c1_idx, c2_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError("Expected double bond not found in acrylate group")

        # Additional bond-based verification (inspired by provided snippet)
        found_cc_double = False
        for bond in mol.GetBonds():
            if bond.GetIdx() == bond.GetIdx() and bond.GetBondType() == Chem.BondType.DOUBLE:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                if (atom1.GetIdx() == c1_idx and atom2.GetIdx() == c2_idx) or \
                (atom1.GetIdx() == c2_idx and atom2.GetIdx() == c1_idx):
                    if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
                        found_cc_double = True
                        break
        if not found_cc_double:
            raise ValueError("No carbon-carbon double bond found in acrylate group at matched indices")

        # Modify the molecule
        rw_mol = Chem.RWMol(mol)
        rw_bond = rw_mol.GetBondBetweenAtoms(c1_idx, c2_idx)
        rw_bond.SetBondType(Chem.BondType.SINGLE)
        if not contains_br:
            br1 = rw_mol.AddAtom(Chem.Atom('Br'))
            br2 = rw_mol.AddAtom(Chem.Atom('Br'))
        else:
            br1 = rw_mol.AddAtom(Chem.Atom('I'))
            br2 = rw_mol.AddAtom(Chem.Atom('I'))
        rw_mol.AddBond(c1_idx, br1, Chem.BondType.SINGLE)
        rw_mol.AddBond(c2_idx, br2, Chem.BondType.SINGLE)
        
        # Sanitize the modified molecule
        try:
            Chem.SanitizeMol(rw_mol)
        except Chem.MolSanitizeException as e:
            raise ValueError("Failed to sanitize modified molecule: " + str(e))
        
        return Chem.MolToSmiles(rw_mol, isomericSmiles=True)
    
    def build_and_save_homopolymer(self, sml: str, mol_id: str) -> str:
        """Build homopolymer and save .xyz file."""
        try:
            if not self.is_acrylate(sml):
                logging.warning(f"ID {mol_id} with SMILES {sml} is not an acrylate")
                return f"Skipping ID {mol_id}: Not an acrylate"
            
            contains_br = sml.__contains__('Br')
            
            #[TODO]: user defined
            sml = self.replace_first_acrylate_cce(sml, contains_br)

            if not contains_br:
                bb1 = stk.BuildingBlock(sml, [stk.BromoFactory()])
                bb2 = stk.BuildingBlock(sml, [stk.BromoFactory()])
            else:
                bb1 = stk.BuildingBlock(sml, [stk.IodoFactory()])
                bb2 = stk.BuildingBlock(sml, [stk.IodoFactory()])
            
            polymer = stk.ConstructedMolecule(
                topology_graph=stk.polymer.Linear(
                    building_blocks=(bb1, bb2),
                    repeating_unit='AB',
                    num_repeating_units=self.polymer_chain_size,
                    optimizer=stk.Collapser(scale_steps=False),
                ),
            )
            
            rdkit_polymer = polymer.to_rdkit_mol()
            rdkit_polymer = Chem.AddHs(rdkit_polymer)
            Chem.SanitizeMol(rdkit_polymer)
            
            rw_mol = Chem.RWMol(rdkit_polymer)
            if not contains_br:
                atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == 'Br']
            else:
                atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == 'I']
            for idx in sorted(atoms_to_remove, reverse=True):
                rw_mol.RemoveAtom(idx)
            rdkit_polymer = rw_mol.GetMol()
            Chem.SanitizeMol(rdkit_polymer)
            
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxIterations = 1000
            params.numThreads = 1
            params.randomSeed = 42
            if AllChem.EmbedMolecule(rdkit_polymer, params) == -1:
                logging.warning(f"Embedding failed for homopolymer {mol_id}")
            
            AllChem.MMFFOptimizeMolecule(rdkit_polymer)
            polymer = polymer.with_position_matrix(
                position_matrix=rdkit_polymer.GetConformer().GetPositions()
            )
            
            xyz_filename = os.path.join(self.output_dir, f"homopoly_{mol_id}_chain_{self.polymer_chain_size}.xyz")
            self.write_xyz_file(rdkit_polymer, xyz_filename)
            return f"Saved homopolymer: {xyz_filename}"
        
        except Exception as e:
            logging.error(f"Error building homopolymer {mol_id} with SMILES {sml}: {str(e)}")
            return f"Skipping ID {mol_id}: Exception occurred - {str(e)}"
    
    def generate(self) -> list:
        """Generate .xyz files for homopolymers from acrylate monomers."""
        print("Filtering dataset for acrylate monomers...")
        df_acrylates = self.df[self.df['smiles'].apply(self.is_acrylate)].copy()
        print(f"Found {len(df_acrylates)} acrylate monomers")
        
        print("Building homopolymers in parallel...")
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(self.build_and_save_homopolymer)(row['smiles'], row['id'])
            for _, row in tqdm(df_acrylates.iterrows(), total=len(df_acrylates))
        )
        return results