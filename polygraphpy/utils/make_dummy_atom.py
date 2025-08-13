from rdkit import Chem

def replace_first_acrylate_cce(smiles: str, contains_br: bool) -> str:
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