from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

def get_vdw_volume(smiles_list: list, verbose: bool = False) -> float:

    if len(smiles_list) == 0:
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "O=C1CCCCCCCCCCC1",
            "C1CCCC1",
            "CC(=O)c1ccc(C)cc1",
            "O=C2c1ccccc1C(=O)c3ccccc23",
            "O=P(c1ccccc1)(c2ccccc2)c3ccccc3",
            "CN1CCCC1c2cccnc2",
            "CCOC(=O)C1(CCN(C)CC1)c2ccccc2",
            "Cc1cc(=O)n(c2ccccc2)n1C",
            "CN(C)P(N(C)C)N(C)C",
            "CCC(=O)C(CC(C)N(C)C)(c1ccccc1)c2ccccc2",
            "NC(=O)NN=Cc1ccc(o1)N(=O)=O",
            "CC1=CNC(=O)NC1=O",
            "O=C1NC(=O)NC(=O)C1(CC)CC"
        ]

    # Compute VDW volume using DoubleCubicLatticeVolume

    if verbose:
        print(f"{'SMILES':<50} | {'VDW Volume (Å³)':>15}")
        print("-" * 70)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if verbose:
            if mol is None:
                print(f"{smi:<50} | Invalid SMILES")
                continue

        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if verbose:
            if success != 0:
                print(f"{smi:<50} | Embedding failed")
                continue

        AllChem.UFFOptimizeMolecule(mol)
        dclv = rdMolDescriptors.DoubleCubicLatticeVolume(mol)
        volume = dclv.GetVDWVolume()

        if verbose:
            print(f"{smi:<50} | {volume:15.2f}")
        
        return volume