"""

These are several metric used to approximate the complexity of a molecule from different angles.

# 	1.	BertzCT: Captures structural branching and connectivity.
#   2.  Böttcher Complexity: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00723
# 	2.	Shannon Entropy of the molecular graph: Measures the structural disorder and diversity of the molecule.
# 	3.	Shannon Entropy of the SMILES string: Measures the structural disorder and diversity of SMILES.
# 	5.	Number of structural motifs: Assesses the chemical diversity and functional complexity.

In this script I have reimplemented code from
- Zach Pearson: https://github.com/boskovicgroup/bottchercomplexity/tree/main
- Nadine Schneider & Peter Ertl: https://github.com/rdkit/rdkit/blob/master/Contrib/ChiralPairs/ChiralDescriptors.py

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from rdkit import Chem
from rdkit.Chem.GraphDescriptors import BertzCT
from molml_mcp.tools.featurization.SMILES_encoding import _tokenize_smiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def _calculate_smiles_branches(smiles: str) -> int:
    """ Count branches in SMILES string (occurrences of '(').

    :param smiles: SMILES string
    :return: Number of branches
    """
    num_branches = smiles.count('(')
    return num_branches


def _calculate_molecular_shannon_entropy(smiles: str) -> float | None:
    """ Compute Shannon entropy of molecular graph based on element distribution.
    Formula: I = N*log2(N) - Σ(Ni*log2(Ni)), where N = total atoms, Ni = atoms of type i.

    :param smiles: SMILES string
    :return: Shannon entropy of molecular graph
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    # Get the symbol of each atom (element)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Calculate frequency of each element
    element_counts = Counter(elements)

    # Total number of elements
    N = len(elements)

    # Calculate Nlog2(N)
    entropy_part1 = N * np.log2(N)

    # Calculate the sum of Ni * log2(Ni) for each distinct element
    entropy_part2 = sum(Ni * np.log2(Ni) for Ni in element_counts.values())

    # Calculate the entropy
    entropy = entropy_part1 - entropy_part2

    return entropy


def _calculate_smiles_shannon_entropy(smiles: str) -> float:
    """ Calculate Shannon entropy of SMILES tokens.
    Formula: H = -Σ(pi*log2(pi)), where pi = probability of token i.

    :param smiles: SMILES string
    :return: Shannon entropy
    """

    tokens = _tokenize_smiles(smiles)

    # Count the frequency of each token in the SMILES string
    char_counts = Counter(tokens)

    # Total number of tokens in the SMILES string
    N = len(tokens)

    # Calculate the probabilities
    probabilities = [count / N for count in char_counts.values()]

    # Calculate Shannon Entropy
    shannon_entropy = sum(p * -np.log2(p) for p in probabilities)

    return shannon_entropy


def _calculate_num_tokens(smiles: str) -> int:
    """ Count tokens in SMILES string (handles multi-character tokens like Cl, Br, @@).

    :param smiles: SMILES string
    :return: Number of tokens
    """
    tokens = _tokenize_smiles(smiles)
    return len(tokens)


def _calculate_bertz_complexity(smiles: str, **kwargs) -> float | None:
    """ Compute Bertz complexity (measures structural branching and connectivity).
    
    :param smiles: SMILES string
    :return: BertzCT complexity score
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    return BertzCT(mol, **kwargs)


def _calculate_bottcher_complexity(smiles: str, debug: bool = False) -> float | None:
    """ Compute Böttcher complexity (atom-wise complexity measure).
    Implementation based on github.com/boskovicgroup/bottchercomplexity

    :param smiles: SMILES string
    :param debug: Print intermediate values
    :return: Böttcher complexity score
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    def _determineAtomSubstituents(atomID, mol, distanceMatrix, verbose=False):
        """

        Copyright (c) 2017, Novartis Institutes for BioMedical Research Inc.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

            * Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
            * Redistributions in binary form must reproduce the above
            copyright notice, this list of conditions and the following
            disclaimer in the documentation and/or other materials provided
            with the distribution.
            * Neither the name of Novartis Institutes for BioMedical Research Inc.
            nor the names of its contributors may be used to endorse or promote
            products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        Created by Nadine Schneider & Peter Ertl, July 2017

        https://github.com/rdkit/rdkit/blob/master/Contrib/ChiralPairs/ChiralDescriptors.py
        """

        atomPaths = distanceMatrix[atomID]
        # determine the direct neighbors of the atom
        neighbors = [n for n, i in enumerate(atomPaths) if i == 1]
        # store the ids of the neighbors (substituents)
        subs = defaultdict(list)
        # track in how many substituents an atom is involved (can happen in rings)
        sharedNeighbors = defaultdict(int)
        # determine the max path length for each substituent
        maxShell = defaultdict(int)
        for n in neighbors:
            subs[n].append(n)
            sharedNeighbors[n] += 1
            maxShell[n] = 0
        # second shell of neighbors
        mindist = 2
        # max distance from atom
        maxdist = int(np.max(atomPaths))
        for d in range(mindist, maxdist + 1):
            if verbose:
                print("Shell: ", d)
            newShell = [n for n, i in enumerate(atomPaths) if i == d]
            for aidx in newShell:
                if verbose:
                    print("Atom ", aidx, " in shell ", d)
                atom = mol.GetAtomWithIdx(aidx)
                # find neighbors of the current atom that are part of the substituent already
                for n in atom.GetNeighbors():
                    nidx = n.GetIdx()
                    for k, v in subs.items():
                        # is the neighbor in the substituent and is not in the same shell as the current atom
                        # and we haven't added the current atom already then put it in the correct substituent list
                        if nidx in v and nidx not in newShell and aidx not in v:
                            subs[k].append(aidx)
                            sharedNeighbors[aidx] += 1
                            maxShell[k] = d
                            if verbose:
                                print("Atom ", aidx, " assigned to ", nidx)
        if verbose:
            print(subs)
            print(sharedNeighbors)

        return subs, sharedNeighbors, maxShell

    def _GetChemicalNonequivs(atom, mol):
        """ D

        Current failures: Does not distinguish between cyclopentyl and pentyl (etc.)
                        and so unfairly underestimates complexity.

        :param atom:
        :param themol:
        :return:
        """

        num_unique_substituents = 0
        substituents = [[], [], [], []]
        for item, key in enumerate(_determineAtomSubstituents(atom.GetIdx(), mol, Chem.GetDistanceMatrix(mol))[0]):
            for subatom in _determineAtomSubstituents(atom.GetIdx(), mol, Chem.GetDistanceMatrix(mol))[0][key]:
                substituents[item].append(mol.GetAtomWithIdx(subatom).GetSymbol())
                num_unique_substituents = len(
                    set(tuple(tuple(substituent) for substituent in substituents if substituent)))

        return num_unique_substituents

    def _GetBottcherLocalDiversity(atom):
        """ E

        The number of different non-hydrogen elements or isotopes (including deuterium
        and tritium) in the atom's microenvironment.

        CH4 - the carbon has e_i of 1
        Carbonyl carbon of an amide e.g. CC(=O)N e_i = 3
            while N and O have e_i = 2
        """

        neighbors = []
        for neighbor in atom.GetNeighbors():
            neighbors.append(str(neighbor.GetSymbol()))
        if atom.GetSymbol() in set(neighbors):
            return len(set(neighbors))
        else:
            return len(set(neighbors)) + 1

    def _GetNumIsomericPossibilities(atom):
        """ S

        RDKit marks atoms where there is potential for isomerization with a tag
        called _CIPCode. If it exists for an atom, note that S = 2, otherwise 1.
        """

        try:
            if (atom.GetProp('_CIPCode')):
                return 2
        except:
            return 1

    def _GetNumValenceElectrons(atom):
        """ V

        The number of valence electrons the atom would have if it were unbonded and
        neutral
        """

        valence = {1: ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],  # Alkali Metals
                2: ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],  # Alkali Earth Metals
                # transition metals???
                3: ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh'],  #
                4: ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl'],
                5: ['N', 'P', 'As', 'Sb', 'Bi', 'Mc'],  # Pnictogens
                6: ['O', 'S', 'Se', 'Te', 'Po', 'Lv'],  # Chalcogens
                7: ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],  # Halogens
                8: ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']}  # Noble Gases
        for k in valence:
            if atom.GetSymbol() in valence[k]:
                return k
        return 0

    def _GetBottcherBondIndex(atom):
        """ B

        Represents the total number of bonds to other atoms with V_i*b_i > 1, so
        basically bonds to atoms other than Hydrogen

        Here we can leverage the fact that RDKit does not even report Hydrogens by
        default to simply loop over the bonds. We will have to account for molecules
        that have hydrogens turned on before we can submit this code as a patch
        though.
        """

        b_sub_i_ranking = 0
        bonds = []
        for bond in atom.GetBonds():
            bonds.append(str(bond.GetBondType()))
        for bond in bonds:
            if bond == 'SINGLE':
                b_sub_i_ranking += 1
            if bond == 'DOUBLE':
                b_sub_i_ranking += 2
            if bond == 'TRIPLE':
                b_sub_i_ranking += 3
        if 'AROMATIC' in bonds:
            # This list can be expanded as errors arise.
            if atom.GetSymbol() == 'C':
                b_sub_i_ranking += 3
            elif atom.GetSymbol() == 'N':
                b_sub_i_ranking += 2
            elif atom.GetSymbol() == 'O':  # I expanded this to O
                b_sub_i_ranking += 2

        if b_sub_i_ranking == 0:
            b_sub_i_ranking += 1

        return b_sub_i_ranking

    # Main function body of _calculate_bottcher_complexity
    try:
        complexity = 0
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        atoms = mol.GetAtoms()
        atom_stereo_classes = []
        atoms_corrected_for_symmetry = []
        for atom in atoms:
            if atom.GetProp('_CIPRank') in atom_stereo_classes:
                continue
            else:
                atoms_corrected_for_symmetry.append(atom)
                atom_stereo_classes.append(atom.GetProp('_CIPRank'))
        for atom in atoms_corrected_for_symmetry:
            d = _GetChemicalNonequivs(atom, mol)
            e = _GetBottcherLocalDiversity(atom)
            s = _GetNumIsomericPossibilities(atom)
            V = _GetNumValenceElectrons(atom)
            b = _GetBottcherBondIndex(atom)
            if debug:
                print(f'Atom: {atom.GetSymbol()}')
                print('\tSymmetry Class: ' + str(atom.GetProp('_CIPRank')))
                print('\tCurrent Parameter Values:')
                print('\t\td_sub_i: ' + str(d))
                print('\t\te_sub_i: ' + str(e))
                print('\t\ts_sub_i: ' + str(s))
                print('\t\tV_sub_i: ' + str(V))
                print('\t\tb_sub_i: ' + str(b))
            complexity += d * e * s * math.log(V * b, 2)
        if debug:
            print('Current Complexity Score: ' + str(complexity))
            return None
    except:
        return None

    return complexity


def add_complexity_columns(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    metrics: List[str],
    output_filename: str,
    explanation: str = "Dataset with molecular complexity metrics"
) -> Dict:
    """
    Add molecular complexity metric columns to dataset.
    
    Available metrics: 'branches', 'num_tokens', 'molecular_entropy', 'smiles_entropy', 'bertz', 'bottcher'
    
    Args:
        input_filename: CSV dataset filename
        project_manifest_path: Path to manifest.json
        smiles_column: Column with SMILES
        metrics: List of metric names to compute
        output_filename: Output dataset name
        explanation: Description for saved dataset
        
    Returns:
        Dictionary with output_filename, n_rows, columns_added, n_failed, preview, summary
    """
    # Metric name to function mapping
    METRIC_FUNCTIONS = {
        'branches': _calculate_smiles_branches,
        'num_tokens': _calculate_num_tokens,
        'molecular_entropy': _calculate_molecular_shannon_entropy,
        'smiles_entropy': _calculate_smiles_shannon_entropy,
        'bertz': _calculate_bertz_complexity,
        'bottcher': _calculate_bottcher_complexity,
    }
    
    # Validate metrics
    invalid_metrics = [m for m in metrics if m not in METRIC_FUNCTIONS]
    if invalid_metrics:
        raise ValueError(
            f"Invalid metrics: {invalid_metrics}. "
            f"Available: {list(METRIC_FUNCTIONS.keys())}"
        )
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Make copy for modifications
    df_copy = df.copy()
    
    # Track failures per metric
    n_failed = {}
    columns_added = []
    
    # Compute each metric
    for metric in metrics:
        func = METRIC_FUNCTIONS[metric]
        column_name = metric
        
        # Initialize column
        values = []
        failures = 0
        
        # Compute for each SMILES
        for smiles in df_copy[smiles_column]:
            if pd.isna(smiles):
                values.append(None)
                failures += 1
            else:
                try:
                    value = func(str(smiles))
                    # Handle functions that return (None, error_msg) tuples
                    if isinstance(value, tuple):
                        values.append(None)
                        failures += 1
                    else:
                        values.append(value)
                except Exception:
                    values.append(None)
                    failures += 1
        
        # Add column to dataframe
        df_copy[column_name] = values
        columns_added.append(column_name)
        n_failed[metric] = failures
    
    # Save dataset (always create new resource for traceability)
    saved_filename = _store_resource(
        df_copy,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    # Get preview of new columns
    preview_cols = [smiles_column] + columns_added
    preview = df_copy[preview_cols].head(5).to_dict('records')
    
    return {
        "output_filename": saved_filename,
        "n_rows": len(df_copy),
        "columns_added": columns_added,
        "n_failed": n_failed,
        "preview": preview,
        "summary": (
            f"Added {len(columns_added)} complexity metric(s) to {len(df_copy)} rows. "
            f"Columns: {', '.join(columns_added)}"
        )
    }


def get_all_complexity_tools():
    """
    Returns all MCP-exposed molecular complexity functions.
    """
    return [
        add_complexity_columns,
    ]