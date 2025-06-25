from polygraphpy.core.molecule import Molecule

class Polymer(Molecule):
    """Base class for polymers."""
    
    def __init__(self, smiles: str, repeat_units: int = 4):
        """Initialize with SMILES string and number of repeat units."""
        super().__init__(smiles)
        self.repeat_units = repeat_units