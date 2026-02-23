
from chemlint.tools.featurization.supported.ecfps import smiles_to_ecfp_dataset
from chemlint.tools.featurization.supported.maccs import smiles_to_maccs_dataset
from chemlint.tools.featurization.supported.rdkit import smiles_to_rdkit_fp_dataset
from chemlint.tools.featurization.supported.cats import smiles_to_cats_dataset
from chemlint.tools.featurization.supported.atompair import smiles_to_atompair_dataset
from chemlint.tools.featurization.supported.torsion import smiles_to_torsion_dataset
from chemlint.tools.featurization.supported.avalon import smiles_to_avalon_dataset


def get_all_complex_descriptor_tools():
    """Return a list of all complex descriptor tools for MCP registration."""
    return [
        smiles_to_ecfp_dataset,
        smiles_to_maccs_dataset,
        smiles_to_rdkit_fp_dataset,
        smiles_to_cats_dataset,
        smiles_to_atompair_dataset,
        smiles_to_torsion_dataset,
        smiles_to_avalon_dataset
    ]

