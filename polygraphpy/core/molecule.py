from rdkit import Chem

class Molecule:
    """Base class for molecular structures."""
    
    def __init__(self, smiles: str):
        """Initialize with SMILES string."""
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
    
    def validate_smiles(self) -> bool:
        """Validate the SMILES string."""
        return self.mol is not None