"""
Data quality report generation for molecular datasets.

Provides comprehensive analysis including completeness, SMILES validity,
PAINS screening, duplicates, physicochemical properties, and activity distribution.

This module leverages existing tools in the codebase:
- filter_by_pains() for PAINS screening
- filter_by_lipinski_ro5() for Rule of Five analysis
- filter_by_veber_rules() for Veber rules
- filter_by_qed() for QED scoring
- find_duplicates_dataset() for duplicate analysis
- calculate_simple_descriptors() for descriptor calculation
- detect_outliers_iqr() for outlier detection
- get_dataset_summary() for column statistics
- scaffold_analysis() for scaffold diversity
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from collections import Counter
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Fragments
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from chemlint.infrastructure.resources import _load_resource, _store_resource
from chemlint.tools.core.dataset_ops import get_dataset_summary
from chemlint.tools.featurization.simple_descriptors import calculate_simple_descriptors
from chemlint.tools.core.filtering import filter_by_pains, filter_by_lipinski_ro5, filter_by_veber_rules, filter_by_qed
from chemlint.tools.cleaning.deduplication import find_duplicates_dataset
from chemlint.tools.core.outliers import detect_outliers_iqr
from chemlint.tools.reports.scaffold_analysis import scaffold_analysis
from chemlint.constants import COMMON_SOLVENTS, COMMON_SALT_SMILES


# ============================================================================
# HELPER FUNCTIONS - Unique functions not available elsewhere
# ============================================================================

def _analyze_smiles_validity(df: pd.DataFrame, smiles_col: str) -> Dict:
    """
    Check SMILES validity and collect examples of invalid SMILES.
    
    Returns dict with validity statistics and examples.
    """
    valid_flags = []
    invalid_examples = []
    
    for idx, smi in enumerate(df[smiles_col]):
        if pd.isna(smi) or smi == '' or not isinstance(smi, str):
            valid_flags.append(False)
            if len(invalid_examples) < 5:
                invalid_examples.append({
                    'row': idx,
                    'smiles': str(smi) if pd.notna(smi) else '(empty)',
                    'reason': 'Empty or NaN'
                })
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                valid_flags.append(False)
                if len(invalid_examples) < 5:
                    invalid_examples.append({
                        'row': idx,
                        'smiles': smi[:50],  # Truncate long SMILES
                        'reason': 'Invalid SMILES syntax'
                    })
            else:
                valid_flags.append(True)
    
    n_valid = sum(valid_flags)
    n_invalid = len(valid_flags) - n_valid
    n_total = len(df)
    
    return {
        'n_valid': int(n_valid),
        'n_invalid': int(n_invalid),
        'n_total': n_total,
        'pct_valid': float(n_valid / n_total * 100) if n_total > 0 else 0,
        'pct_invalid': float(n_invalid / n_total * 100) if n_total > 0 else 0,
        'invalid_examples': invalid_examples,
        'valid_flags': valid_flags
    }


def _analyze_activity_distribution(activity_values: np.ndarray, 
                                   activity_type: str = 'continuous',
                                   units: str = 'nM') -> Dict:
    """
    Analyze activity distribution including normality tests.
    
    activity_type: 'continuous' or 'classification'
    units: Activity units for continuous data (e.g., 'nM', 'μM')
    
    Returns dict with distribution statistics.
    """
    # Remove NaN values
    values_clean = activity_values[~np.isnan(activity_values)]
    n_total = len(activity_values)
    n_valid = len(values_clean)
    n_missing = n_total - n_valid
    
    if activity_type == 'classification':
        # Binary classification analysis
        n_positive = int(np.sum(values_clean == 1))
        n_negative = int(np.sum(values_clean == 0))
        balance = float(n_positive / n_valid) if n_valid > 0 else 0
        
        return {
            'type': 'classification',
            'n_valid': int(n_valid),
            'n_missing': int(n_missing),
            'n_positive': n_positive,
            'n_negative': n_negative,
            'balance': balance
        }
    
    else:  # continuous
        if len(values_clean) < 3:
            return {'type': 'continuous', 'error': 'Insufficient data'}
        
        # Linear scale statistics
        linear_stats = {
            'mean': float(values_clean.mean()),
            'median': float(np.median(values_clean)),
            'geometric_mean': float(stats.gmean(values_clean[values_clean > 0])) if np.any(values_clean > 0) else np.nan,
            'std': float(values_clean.std()),
            'min': float(values_clean.min()),
            'max': float(values_clean.max()),
            'range_log_units': float(np.log10(values_clean.max() / values_clean.min())) if values_clean.min() > 0 else np.nan,
            'p10': float(np.percentile(values_clean, 10)),
            'p25': float(np.percentile(values_clean, 25)),
            'p75': float(np.percentile(values_clean, 75)),
            'p90': float(np.percentile(values_clean, 90)),
            'skewness': float(stats.skew(values_clean)),
            'kurtosis': float(stats.kurtosis(values_clean))
        }
        
        # Log-transformed statistics (assuming nM units, convert to pValue)
        log_values = -np.log10(values_clean / 1e9)
        log_stats = {
            'mean': float(log_values.mean()),
            'median': float(np.median(log_values)),
            'std': float(log_values.std()),
            'min': float(log_values.min()),
            'max': float(log_values.max()),
            'skewness': float(stats.skew(log_values)),
            'kurtosis': float(stats.kurtosis(log_values))
        }
        
        # Normality test on log-transformed values
        if len(log_values) >= 3:
            try:
                w_stat, p_value = stats.shapiro(log_values[:5000])  # Shapiro-Wilk limited to 5000 samples
                normality_test = {
                    'w_statistic': float(w_stat),
                    'p_value': float(p_value),
                    'is_normal': bool(p_value > 0.05)
                }
            except:
                normality_test = None
        else:
            normality_test = None
        
        # Activity bins
        bins = {
            '< 100': int(np.sum(values_clean < 100)),
            '100-1000': int(np.sum((values_clean >= 100) & (values_clean < 1000))),
            '1000-10000': int(np.sum((values_clean >= 1000) & (values_clean < 10000))),
            '> 10000': int(np.sum(values_clean >= 10000))
        }
        
        # Outlier detection on log scale
        q25_log = np.percentile(log_values, 25)
        q75_log = np.percentile(log_values, 75)
        iqr_log = q75_log - q25_log
        lower_bound_log = q25_log - 1.5 * iqr_log
        upper_bound_log = q75_log + 1.5 * iqr_log
        
        ultra_potent = int(np.sum(log_values > upper_bound_log))
        weak_outliers = int(np.sum(log_values < lower_bound_log))
        
        return {
            'type': 'continuous',
            'units': units,
            'n_valid': int(n_valid),
            'n_missing': int(n_missing),
            'linear_stats': linear_stats,
            'log_stats': log_stats,
            'normality_test': normality_test,
            'bins': bins,
            'outliers': {
                'ultra_potent': ultra_potent,
                'weak_outliers': weak_outliers
            }
        }


def _analyze_stereochemistry(smiles_list: List[str]) -> Dict:
    """
    Analyze stereochemistry in molecules.
    
    Returns dict with stereochemistry statistics including chiral centers,
    E/Z double bonds, and overall stereochemical completeness.
    """
    from rdkit.Chem import FindMolChiralCenters, Descriptors
    
    total_molecules = 0
    molecules_with_stereo = 0
    
    # Chiral centers
    total_chiral_centers = 0
    specified_chiral_centers = 0
    unspecified_chiral_centers = 0
    molecules_with_chiral = 0
    
    # Double bonds
    total_double_bonds = 0
    specified_double_bonds = 0
    unspecified_double_bonds = 0
    molecules_with_double_bonds = 0
    
    # Track stereochemical completeness
    fully_specified_molecules = 0
    partially_specified_molecules = 0
    no_stereo_molecules = 0
    
    for smi in smiles_list:
        if pd.isna(smi) or smi == '' or not isinstance(smi, str):
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        total_molecules += 1
        
        # Count chiral centers
        chiral_centers = FindMolChiralCenters(mol, includeUnassigned=True)
        n_chiral = len(chiral_centers)
        n_specified_chiral = 0
        n_unspecified_chiral = 0
        
        if n_chiral > 0:
            molecules_with_chiral += 1
            total_chiral_centers += n_chiral
            
            # Count specified vs unspecified
            n_specified_chiral = sum(1 for _, label in chiral_centers if label in ['R', 'S'])
            n_unspecified_chiral = n_chiral - n_specified_chiral
            
            specified_chiral_centers += n_specified_chiral
            unspecified_chiral_centers += n_unspecified_chiral
        
        # Count E/Z double bonds
        n_double_bonds = 0
        n_specified_db = 0
        n_unspecified_db = 0
        
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                # Check if it's a C=C bond (most relevant for E/Z)
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                
                # Skip if not carbon-carbon double bond
                if atom1.GetSymbol() != 'C' or atom2.GetSymbol() != 'C':
                    continue
                
                # Check if bond has potential for E/Z stereochemistry
                # (each carbon must have at least one non-hydrogen substituent)
                if atom1.GetDegree() >= 2 and atom2.GetDegree() >= 2:
                    n_double_bonds += 1
                    
                    # Check if stereochemistry is specified
                    stereo = bond.GetStereo()
                    if stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
                        n_specified_db += 1
                    else:
                        n_unspecified_db += 1
        
        if n_double_bonds > 0:
            molecules_with_double_bonds += 1
            total_double_bonds += n_double_bonds
            specified_double_bonds += n_specified_db
            unspecified_double_bonds += n_unspecified_db
        
        # Determine stereochemical completeness for this molecule
        total_stereo_features = n_chiral + n_double_bonds
        
        if total_stereo_features == 0:
            no_stereo_molecules += 1
        else:
            molecules_with_stereo += 1
            specified_features = n_specified_chiral + n_specified_db
            
            if specified_features == total_stereo_features:
                fully_specified_molecules += 1
            elif specified_features > 0:
                partially_specified_molecules += 1
    
    # Calculate percentages
    pct_with_stereo = (molecules_with_stereo / total_molecules * 100) if total_molecules > 0 else 0
    pct_chiral = (specified_chiral_centers / total_chiral_centers * 100) if total_chiral_centers > 0 else 0
    pct_db = (specified_double_bonds / total_double_bonds * 100) if total_double_bonds > 0 else 0
    pct_fully_specified = (fully_specified_molecules / molecules_with_stereo * 100) if molecules_with_stereo > 0 else 0
    
    return {
        'n_molecules': int(total_molecules),
        'n_with_stereochemistry': int(molecules_with_stereo),
        'pct_with_stereochemistry': float(pct_with_stereo),
        'chiral_centers': {
            'total': int(total_chiral_centers),
            'specified': int(specified_chiral_centers),
            'unspecified': int(unspecified_chiral_centers),
            'pct_specified': float(pct_chiral),
            'n_molecules_with_chiral': int(molecules_with_chiral)
        },
        'double_bonds': {
            'total': int(total_double_bonds),
            'specified': int(specified_double_bonds),
            'unspecified': int(unspecified_double_bonds),
            'pct_specified': float(pct_db),
            'n_molecules_with_bonds': int(molecules_with_double_bonds)
        },
        'completeness': {
            'fully_specified': int(fully_specified_molecules),
            'partially_specified': int(partially_specified_molecules),
            'no_stereochemistry': int(no_stereo_molecules),
            'pct_fully_specified': float(pct_fully_specified)
        }
    }


def _analyze_activity_correlations(df: pd.DataFrame, 
                                    smiles_col: str, 
                                    activity_col: str,
                                    activity_type: str) -> Dict:
    """
    Analyze correlations between molecular properties/features and bioactivity.
    
    For regression: Computes Pearson and Spearman correlations
    For classification: Computes feature distribution differences between classes
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with SMILES and activity
    smiles_col : str
        SMILES column name
    activity_col : str
        Activity column name
    activity_type : str
        'regression' or 'classification'
    
    Returns
    -------
    dict
        Correlation/overrepresentation statistics
    """
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu
    
    # Properties to calculate
    properties = {}
    functional_groups = {}
    
    # Calculate properties and functional groups for each molecule
    for idx, smi in enumerate(df[smiles_col]):
        if pd.isna(smi) or not isinstance(smi, str):
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Calculate physicochemical properties
        if idx == 0:  # Initialize on first molecule
            properties['MolWt'] = []
            properties['LogP'] = []
            properties['TPSA'] = []
            properties['NumHDonors'] = []
            properties['NumHAcceptors'] = []
            properties['NumRotatableBonds'] = []
            properties['NumAromaticRings'] = []
            properties['FractionCSP3'] = []
            properties['NumRings'] = []
        
        properties['MolWt'].append(Descriptors.MolWt(mol))
        properties['LogP'].append(Descriptors.MolLogP(mol))
        properties['TPSA'].append(Descriptors.TPSA(mol))
        properties['NumHDonors'].append(Descriptors.NumHDonors(mol))
        properties['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
        properties['NumRotatableBonds'].append(Descriptors.NumRotatableBonds(mol))
        properties['NumAromaticRings'].append(Descriptors.NumAromaticRings(mol))
        properties['FractionCSP3'].append(Descriptors.FractionCSP3(mol))
        properties['NumRings'].append(Descriptors.RingCount(mol))
        
        # Calculate functional groups
        if idx == 0:
            functional_groups['Aromatic_rings'] = []
            functional_groups['Aliphatic_OH'] = []
            functional_groups['Aromatic_OH'] = []
            functional_groups['Carboxylic_acid'] = []
            functional_groups['Ester'] = []
            functional_groups['Amide'] = []
            functional_groups['Primary_amine'] = []
            functional_groups['Secondary_amine'] = []
            functional_groups['Tertiary_amine'] = []
            functional_groups['Halogen'] = []
            functional_groups['Nitro'] = []
        
        functional_groups['Aromatic_rings'].append(Descriptors.NumAromaticRings(mol))
        functional_groups['Aliphatic_OH'].append(Fragments.fr_Al_OH(mol))
        functional_groups['Aromatic_OH'].append(Fragments.fr_Ar_OH(mol))
        functional_groups['Carboxylic_acid'].append(Fragments.fr_COO(mol) + Fragments.fr_COO2(mol))
        functional_groups['Ester'].append(Fragments.fr_ester(mol))
        functional_groups['Amide'].append(Fragments.fr_amide(mol))
        functional_groups['Primary_amine'].append(Fragments.fr_NH2(mol))
        functional_groups['Secondary_amine'].append(Fragments.fr_NH1(mol))
        functional_groups['Tertiary_amine'].append(Fragments.fr_NH0(mol))
        functional_groups['Halogen'].append(Fragments.fr_halogen(mol))
        functional_groups['Nitro'].append(Fragments.fr_nitro(mol))
    
    # Get activity values (aligned with calculated properties)
    activity_values = df[activity_col].values[:len(properties['MolWt'])]
    
    # Remove NaN activities
    valid_indices = ~np.isnan(activity_values)
    activity_values = activity_values[valid_indices]
    
    for key in properties:
        properties[key] = np.array(properties[key])[valid_indices]
    for key in functional_groups:
        functional_groups[key] = np.array(functional_groups[key])[valid_indices]
    
    if activity_type == 'regression':
        # Compute correlations for regression
        property_correlations = {}
        fg_correlations = {}
        
        for prop_name, prop_values in properties.items():
            if len(prop_values) > 2:
                try:
                    pearson_r, pearson_p = pearsonr(prop_values, activity_values)
                    spearman_r, spearman_p = spearmanr(prop_values, activity_values)
                    property_correlations[prop_name] = {
                        'pearson_r': float(pearson_r),
                        'pearson_p': float(pearson_p),
                        'spearman_r': float(spearman_r),
                        'spearman_p': float(spearman_p)
                    }
                except:
                    pass
        
        for fg_name, fg_values in functional_groups.items():
            if len(fg_values) > 2 and np.sum(fg_values > 0) > 5:  # Only if feature is present in at least 5 molecules
                try:
                    pearson_r, pearson_p = pearsonr(fg_values, activity_values)
                    spearman_r, spearman_p = spearmanr(fg_values, activity_values)
                    fg_correlations[fg_name] = {
                        'pearson_r': float(pearson_r),
                        'pearson_p': float(pearson_p),
                        'spearman_r': float(spearman_r),
                        'spearman_p': float(spearman_p)
                    }
                except:
                    pass
        
        # Find top correlations
        all_correlations = []
        for name, corr in property_correlations.items():
            all_correlations.append({
                'feature': name,
                'type': 'property',
                'correlation': corr['spearman_r'],
                'p_value': corr['spearman_p']
            })
        for name, corr in fg_correlations.items():
            all_correlations.append({
                'feature': name,
                'type': 'functional_group',
                'correlation': corr['spearman_r'],
                'p_value': corr['spearman_p']
            })
        
        # Sort by absolute correlation
        all_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'analysis_type': 'regression',
            'n_samples': int(len(activity_values)),
            'property_correlations': property_correlations,
            'functional_group_correlations': fg_correlations,
            'top_correlations': all_correlations[:10],
            'significant_correlations': [c for c in all_correlations if c['p_value'] < 0.05][:10]
        }
    
    else:  # classification
        # Compare feature distributions between active (1) and inactive (0)
        active_mask = activity_values == 1
        inactive_mask = activity_values == 0
        
        n_active = int(np.sum(active_mask))
        n_inactive = int(np.sum(inactive_mask))
        
        property_differences = {}
        fg_differences = {}
        
        for prop_name, prop_values in properties.items():
            active_vals = prop_values[active_mask]
            inactive_vals = prop_values[inactive_mask]
            
            if len(active_vals) > 0 and len(inactive_vals) > 0:
                try:
                    # Mann-Whitney U test for difference
                    u_stat, p_value = mannwhitneyu(active_vals, inactive_vals, alternative='two-sided')
                    
                    property_differences[prop_name] = {
                        'active_mean': float(np.mean(active_vals)),
                        'inactive_mean': float(np.mean(inactive_vals)),
                        'active_median': float(np.median(active_vals)),
                        'inactive_median': float(np.median(inactive_vals)),
                        'difference': float(np.mean(active_vals) - np.mean(inactive_vals)),
                        'fold_change': float(np.mean(active_vals) / np.mean(inactive_vals)) if np.mean(inactive_vals) != 0 else np.inf,
                        'p_value': float(p_value)
                    }
                except:
                    pass
        
        for fg_name, fg_values in functional_groups.items():
            active_vals = fg_values[active_mask]
            inactive_vals = fg_values[inactive_mask]
            
            # Calculate presence percentage
            pct_active = (np.sum(active_vals > 0) / len(active_vals) * 100) if len(active_vals) > 0 else 0
            pct_inactive = (np.sum(inactive_vals > 0) / len(inactive_vals) * 100) if len(inactive_vals) > 0 else 0
            
            if pct_active > 5 or pct_inactive > 5:  # Only if present in >5% of at least one class
                try:
                    u_stat, p_value = mannwhitneyu(active_vals, inactive_vals, alternative='two-sided')
                    
                    fg_differences[fg_name] = {
                        'pct_active': float(pct_active),
                        'pct_inactive': float(pct_inactive),
                        'enrichment': float(pct_active / pct_inactive) if pct_inactive > 0 else np.inf,
                        'active_mean': float(np.mean(active_vals)),
                        'inactive_mean': float(np.mean(inactive_vals)),
                        'p_value': float(p_value)
                    }
                except:
                    pass
        
        # Find top differences
        all_differences = []
        for name, diff in property_differences.items():
            all_differences.append({
                'feature': name,
                'type': 'property',
                'difference': abs(diff['difference']),
                'fold_change': diff['fold_change'],
                'p_value': diff['p_value'],
                'direction': 'higher_in_active' if diff['difference'] > 0 else 'higher_in_inactive'
            })
        for name, diff in fg_differences.items():
            all_differences.append({
                'feature': name,
                'type': 'functional_group',
                'enrichment': diff['enrichment'],
                'pct_active': diff['pct_active'],
                'pct_inactive': diff['pct_inactive'],
                'p_value': diff['p_value'],
                'direction': 'enriched_in_active' if diff['enrichment'] > 1 else 'enriched_in_inactive'
            })
        
        # Sort by p-value
        all_differences.sort(key=lambda x: x['p_value'])
        
        return {
            'analysis_type': 'classification',
            'n_active': n_active,
            'n_inactive': n_inactive,
            'n_samples': int(len(activity_values)),
            'property_differences': property_differences,
            'functional_group_differences': fg_differences,
            'top_differences': all_differences[:10],
            'significant_differences': [d for d in all_differences if d['p_value'] < 0.05][:10]
        }


def _analyze_special_features(smiles_list: List[str]) -> Dict:
    """
    Analyze special molecular features including organometallics, 
    non-standard isotopes, and ring size distribution.
    
    Returns dict with statistics on these advanced structural features.
    """
    # Common metals for organometallic detection
    metal_symbols = [
        'Li', 'Na', 'K', 'Rb', 'Cs',  # Alkali metals
        'Be', 'Mg', 'Ca', 'Sr', 'Ba',  # Alkaline earth
        'Al', 'Ga', 'In', 'Sn', 'Pb',  # Post-transition metals
        'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # Transition metals (3d)
        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # Transition metals (4d)
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  # Transition metals (5d)
        'B', 'Si', 'Ge', 'As', 'Sb', 'Bi', 'Se', 'Te',  # Metalloids
    ]
    
    total_molecules = 0
    organometallic_count = 0
    isotope_count = 0
    
    metal_counts = {}
    isotope_info = []
    
    # Ring size distribution
    ring_sizes = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 'larger': 0}
    molecules_with_rings = 0
    total_rings = 0
    
    examples = {
        'organometallic': [],
        'isotopes': [],
        'unusual_rings': []
    }
    
    for smi in smiles_list:
        if pd.isna(smi) or smi == '' or not isinstance(smi, str):
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        total_molecules += 1
        
        # Check for organometallic compounds
        has_metal = False
        found_metals = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in metal_symbols:
                has_metal = True
                if symbol not in found_metals:
                    found_metals.append(symbol)
                metal_counts[symbol] = metal_counts.get(symbol, 0) + 1
        
        if has_metal:
            organometallic_count += 1
            if len(examples['organometallic']) < 3:
                examples['organometallic'].append({
                    'smiles': smi[:60],
                    'metals': found_metals
                })
        
        # Check for non-standard isotopes
        has_isotope = False
        found_isotopes = []
        for atom in mol.GetAtoms():
            isotope = atom.GetIsotope()
            if isotope != 0:  # 0 means natural abundance
                has_isotope = True
                symbol = atom.GetSymbol()
                isotope_label = f"{isotope}{symbol}"
                if isotope_label not in found_isotopes:
                    found_isotopes.append(isotope_label)
        
        if has_isotope:
            isotope_count += 1
            if len(examples['isotopes']) < 3:
                examples['isotopes'].append({
                    'smiles': smi[:60],
                    'isotopes': found_isotopes
                })
            isotope_info.extend(found_isotopes)
        
        # Analyze ring sizes
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            molecules_with_rings += 1
            total_rings += ring_info.NumRings()
            
            for ring in ring_info.AtomRings():
                ring_size = len(ring)
                if ring_size in ring_sizes:
                    ring_sizes[ring_size] += 1
                elif ring_size > 8:
                    ring_sizes['larger'] += 1
                
                # Collect examples of unusual ring sizes (3, 4, or >8)
                if (ring_size in [3, 4] or ring_size > 8) and len(examples['unusual_rings']) < 3:
                    examples['unusual_rings'].append({
                        'smiles': smi[:60],
                        'ring_size': ring_size
                    })
    
    # Calculate percentages
    pct_organometallic = (organometallic_count / total_molecules * 100) if total_molecules > 0 else 0
    pct_isotope = (isotope_count / total_molecules * 100) if total_molecules > 0 else 0
    pct_with_rings = (molecules_with_rings / total_molecules * 100) if total_molecules > 0 else 0
    
    # Top metals found
    top_metals = {k: v for k, v in sorted(metal_counts.items(), key=lambda x: x[1], reverse=True)[:10]}
    
    # Isotope summary
    isotope_summary = {}
    for iso in set(isotope_info):
        isotope_summary[iso] = isotope_info.count(iso)
    
    # Ring statistics
    avg_rings_per_molecule = total_rings / molecules_with_rings if molecules_with_rings > 0 else 0
    
    return {
        'n_molecules': int(total_molecules),
        'organometallic': {
            'count': int(organometallic_count),
            'pct': float(pct_organometallic),
            'metals_found': top_metals,
            'n_metal_types': len(metal_counts)
        },
        'isotopes': {
            'count': int(isotope_count),
            'pct': float(pct_isotope),
            'isotopes_found': isotope_summary,
            'n_isotope_types': len(isotope_summary)
        },
        'ring_distribution': {
            'molecules_with_rings': int(molecules_with_rings),
            'pct_with_rings': float(pct_with_rings),
            'total_rings': int(total_rings),
            'avg_rings_per_molecule': float(avg_rings_per_molecule),
            'sizes': {str(k): v for k, v in ring_sizes.items()}
        },
        'examples': examples
    }


def _analyze_salts_fragments_solvents(smiles_list: List[str]) -> Dict:
    """
    Analyze presence of salts, fragments, and solvents in molecules.
    
    Uses predefined lists from constants.py:
    - COMMON_SALT_SMILES: Common inorganic salts
    - COMMON_SOLVENTS: Common organic solvents
    
    Returns dict with statistics on multi-component entries, common salts,
    and solvents.
    """
    def _canonicalize_fragment(frag: str) -> str:
        """Canonicalize a SMILES fragment for comparison. Returns original if canonicalization fails."""
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                return frag
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except:
            return frag
    
    # Use predefined salt and solvent lists from constants
    common_salts = COMMON_SALT_SMILES
    
    # Build solvent lookup with names
    common_solvents = {
        'O': 'Water',
        'CCO': 'Ethanol',
        'C': 'Methane',
        'CO': 'Methanol',
        'CC(C)=O': 'Acetone',
        'CC(=O)O': 'Acetic acid',
        'C1CCOC1': 'Tetrahydrofuran (THF)',
        'ClCCl': 'Dichloromethane (DCM)',
        'ClC(Cl)Cl': 'Chloroform',
        'CC#N': 'Acetonitrile',
        'CS(C)=O': 'DMSO',
        'CN(C)C=O': 'DMF',
        'c1ccccc1': 'Benzene',
        'Cc1ccccc1': 'Toluene',
        'c1ccncc1': 'Pyridine',
        'C1CCCCC1': 'Cyclohexane',
        'CCOC(C)=O': 'Ethyl acetate',
    }
    
    # Check which predefined solvents are in the constants list
    for solv_smiles in COMMON_SOLVENTS:
        if solv_smiles not in common_solvents:
            # Add with a generic name if not already mapped
            common_solvents[solv_smiles] = f'Solvent_{solv_smiles[:20]}'
    
    # Canonicalize salt SMILES for reliable matching
    common_salts_canonical = {}
    for salt_smiles, salt_name in common_salts.items():
        canonical = _canonicalize_fragment(salt_smiles)
        common_salts_canonical[canonical] = salt_name
    
    # Canonicalize solvent SMILES for reliable matching
    common_solvents_canonical = {}
    for solv_smiles, solv_name in common_solvents.items():
        canonical = _canonicalize_fragment(solv_smiles)
        common_solvents_canonical[canonical] = solv_name
    
    total_molecules = 0
    fragmented_molecules = 0
    multi_component_molecules = 0
    
    salt_counts = {name: 0 for name in set(common_salts.values())}
    solvent_counts = {name: 0 for name in common_solvents.values()}
    
    fragment_distribution = []
    examples = {
        'multi_component': [],
        'with_salts': [],
        'with_solvents': []
    }
    
    for smi in smiles_list:
        if pd.isna(smi) or smi == '' or not isinstance(smi, str):
            continue
        
        total_molecules += 1
        
        # Check if fragmented (contains '.')
        if '.' in smi:
            fragmented_molecules += 1
            
            # Split into fragments
            fragments = smi.split('.')
            fragment_distribution.append(len(fragments))
            
            if len(fragments) > 1:
                multi_component_molecules += 1
                
                # Check for salts (canonicalize fragments for accurate matching)
                found_salts = []
                for frag in fragments:
                    frag_canonical = _canonicalize_fragment(frag)
                    for salt_canonical, salt_name in common_salts_canonical.items():
                        if frag_canonical == salt_canonical:
                            salt_counts[salt_name] += 1
                            if salt_name not in found_salts:
                                found_salts.append(salt_name)
                            break
                
                # Check for solvents (canonicalize fragments for accurate matching)
                found_solvents = []
                for frag in fragments:
                    frag_canonical = _canonicalize_fragment(frag)
                    for solv_canonical, solv_name in common_solvents_canonical.items():
                        if frag_canonical == solv_canonical:
                            solvent_counts[solv_name] += 1
                            if solv_name not in found_solvents:
                                found_solvents.append(solv_name)
                            break
                
                # Collect examples
                if found_salts and len(examples['with_salts']) < 3:
                    examples['with_salts'].append({
                        'smiles': smi[:80],
                        'salts': found_salts,
                        'n_fragments': len(fragments)
                    })
                
                if found_solvents and len(examples['with_solvents']) < 3:
                    examples['with_solvents'].append({
                        'smiles': smi[:80],
                        'solvents': found_solvents,
                        'n_fragments': len(fragments)
                    })
                
                if len(examples['multi_component']) < 3:
                    examples['multi_component'].append({
                        'smiles': smi[:80],
                        'n_fragments': len(fragments)
                    })
        else:
            fragment_distribution.append(1)
    
    # Calculate percentages
    pct_fragmented = (fragmented_molecules / total_molecules * 100) if total_molecules > 0 else 0
    pct_multi_component = (multi_component_molecules / total_molecules * 100) if total_molecules > 0 else 0
    
    # Get top salts detected
    top_salts = {k: v for k, v in sorted(salt_counts.items(), key=lambda x: x[1], reverse=True) if v > 0}
    
    # Get top solvents detected
    top_solvents = {k: v for k, v in sorted(solvent_counts.items(), key=lambda x: x[1], reverse=True) if v > 0}
    
    # Determine if desalting/defragmentation is needed
    needs_desalting = pct_fragmented > 10 or len(top_salts) > 0
    
    # Fragment statistics
    if fragment_distribution:
        max_fragments = max(fragment_distribution)
        avg_fragments = sum(fragment_distribution) / len(fragment_distribution)
    else:
        max_fragments = 0
        avg_fragments = 0
    
    return {
        'n_molecules': int(total_molecules),
        'n_fragmented': int(fragmented_molecules),
        'n_multi_component': int(multi_component_molecules),
        'pct_fragmented': float(pct_fragmented),
        'pct_multi_component': float(pct_multi_component),
        'fragment_stats': {
            'max_fragments': int(max_fragments),
            'avg_fragments': float(avg_fragments)
        },
        'salts_detected': top_salts,
        'solvents_detected': top_solvents,
        'n_salt_types': len(top_salts),
        'n_solvent_types': len(top_solvents),
        'needs_desalting': needs_desalting,
        'examples': examples
    }


def _analyze_charge_state(smiles_list: List[str]) -> Dict:
    """
    Analyze charge states in molecules.
    
    Returns dict with charge statistics including formal charges,
    zwitterions, and charge distribution.
    """
    total_molecules = 0
    neutral_molecules = 0
    charged_molecules = 0
    
    positive_only = 0
    negative_only = 0
    zwitterions = 0
    
    total_positive_charges = 0
    total_negative_charges = 0
    
    charge_distribution = []
    examples = {
        'positive': [],
        'negative': [],
        'zwitterion': []
    }
    
    for smi in smiles_list:
        if pd.isna(smi) or smi == '' or not isinstance(smi, str):
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        total_molecules += 1
        
        # Calculate formal charge
        total_charge = Chem.GetFormalCharge(mol)
        
        # Count individual positive and negative charges
        pos_charges = 0
        neg_charges = 0
        
        for atom in mol.GetAtoms():
            fc = atom.GetFormalCharge()
            if fc > 0:
                pos_charges += fc
            elif fc < 0:
                neg_charges += abs(fc)
        
        total_positive_charges += pos_charges
        total_negative_charges += neg_charges
        
        # Classify molecule
        if total_charge == 0 and pos_charges == 0 and neg_charges == 0:
            neutral_molecules += 1
        elif pos_charges > 0 and neg_charges > 0:
            # Zwitterion: has both positive and negative charges but net may be neutral
            zwitterions += 1
            charged_molecules += 1
            if len(examples['zwitterion']) < 3:
                examples['zwitterion'].append({
                    'smiles': smi[:60],
                    'pos_charges': int(pos_charges),
                    'neg_charges': int(neg_charges),
                    'net_charge': int(total_charge)
                })
        elif pos_charges > 0:
            positive_only += 1
            charged_molecules += 1
            if len(examples['positive']) < 3:
                examples['positive'].append({
                    'smiles': smi[:60],
                    'charge': int(total_charge)
                })
        elif neg_charges > 0:
            negative_only += 1
            charged_molecules += 1
            if len(examples['negative']) < 3:
                examples['negative'].append({
                    'smiles': smi[:60],
                    'charge': int(total_charge)
                })
        else:
            neutral_molecules += 1
        
        charge_distribution.append(int(total_charge))
    
    # Calculate percentages
    pct_neutral = (neutral_molecules / total_molecules * 100) if total_molecules > 0 else 0
    pct_charged = (charged_molecules / total_molecules * 100) if total_molecules > 0 else 0
    pct_zwitterions = (zwitterions / total_molecules * 100) if total_molecules > 0 else 0
    
    # Charge distribution summary
    charge_counts = {}
    for charge in set(charge_distribution):
        charge_counts[str(charge)] = charge_distribution.count(charge)
    
    # Determine if neutralization is needed
    needs_neutralization = pct_charged > 10  # If >10% molecules are charged
    
    return {
        'n_molecules': int(total_molecules),
        'n_neutral': int(neutral_molecules),
        'n_charged': int(charged_molecules),
        'pct_neutral': float(pct_neutral),
        'pct_charged': float(pct_charged),
        'charge_types': {
            'positive_only': int(positive_only),
            'negative_only': int(negative_only),
            'zwitterions': int(zwitterions),
            'pct_zwitterions': float(pct_zwitterions)
        },
        'total_charges': {
            'positive': int(total_positive_charges),
            'negative': int(total_negative_charges)
        },
        'charge_distribution': charge_counts,
        'needs_neutralization': needs_neutralization,
        'examples': examples
    }


def _analyze_functional_groups(smiles_list: List[str]) -> Dict:
    """
    Detect common functional groups in molecules.
    
    Returns dict with functional group counts.
    """
    functional_groups = {
        'Aromatic rings': [],
        'Carbonyl': [],
        'Carboxylic acid': [],
        'Amide': [],
        'Ether': [],
        'Halogen': [],
        'Hydroxyl': [],
        'Primary amine': [],
        'Ester': [],
        'Tertiary amine': [],
        'Sulfonamide': [],
        'Ketone': [],
        'Secondary amine': [],
        'Thiol': [],
        'Nitro': []
    }
    
    halogen_breakdown = {'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
    
    for smi in smiles_list:
        if pd.isna(smi) or smi == '':
            for key in functional_groups:
                functional_groups[key].append(0)
            continue
        
        # Convert to string if needed
        if not isinstance(smi, str):
            try:
                smi = str(smi)
            except:
                for key in functional_groups:
                    functional_groups[key].append(0)
                continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for key in functional_groups:
                functional_groups[key].append(0)
            continue
        
        # Count functional groups
        functional_groups['Aromatic rings'].append(Descriptors.NumAromaticRings(mol))
        functional_groups['Carbonyl'].append(Fragments.fr_C_O(mol))
        functional_groups['Carboxylic acid'].append(Fragments.fr_COO(mol) + Fragments.fr_COO2(mol))
        functional_groups['Amide'].append(Fragments.fr_amide(mol))
        functional_groups['Ether'].append(Fragments.fr_ether(mol))
        
        n_halogen = Fragments.fr_halogen(mol)
        functional_groups['Halogen'].append(n_halogen)
        
        # Halogen breakdown
        if n_halogen > 0:
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in halogen_breakdown:
                    halogen_breakdown[symbol] += 1
        
        functional_groups['Hydroxyl'].append(Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol))
        functional_groups['Primary amine'].append(Fragments.fr_NH2(mol))
        functional_groups['Ester'].append(Fragments.fr_ester(mol))
        functional_groups['Tertiary amine'].append(Fragments.fr_NH0(mol))
        functional_groups['Sulfonamide'].append(Fragments.fr_sulfonamd(mol))
        functional_groups['Ketone'].append(Fragments.fr_ketone(mol))
        functional_groups['Secondary amine'].append(Fragments.fr_NH1(mol))
        functional_groups['Thiol'].append(Fragments.fr_SH(mol))
        functional_groups['Nitro'].append(Fragments.fr_nitro(mol))
    
    # Summarize
    summary = {}
    n_molecules = len(smiles_list)
    
    for name, counts in functional_groups.items():
        n_present = sum(1 for c in counts if c > 0)
        avg_per_mol = np.mean([c for c in counts if c > 0]) if n_present > 0 else 0
        
        summary[name] = {
            'count': int(n_present),
            'pct_dataset': float(n_present / n_molecules * 100) if n_molecules > 0 else 0,
            'avg_per_molecule': float(avg_per_mol)
        }
    
    summary['halogen_breakdown'] = {k: int(v) for k, v in halogen_breakdown.items()}
    
    return summary


# ============================================================================
# MAIN REPORT GENERATION FUNCTIONS
# ============================================================================

def _perform_quality_report_calculations(
    input_filename: str,
    project_manifest_path: str,
    smiles_col: str = 'SMILES',
    activity_col: Optional[str] = None,
    activity_type: Optional[str] = None,
    activity_units: str = 'nM',
    output_name: str = 'quality_report',
    explanation: str = 'Generated comprehensive data quality report calculations'
) -> str:
    """
    Perform all quality report calculations and store results as JSON.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    project_manifest_path : str
        Path to manifest.json.
    smiles_col : str, default='SMILES'
        Column containing SMILES strings.
    activity_col : str, optional
        Column containing bioactivity values.
    activity_type : str, optional
        'classification' or 'regression'. Required if activity_col provided.
    activity_units : str, default='nM'
        Units for continuous activity data.
    output_name : str, default='quality_report'
        Prefix for output JSON file.
    explanation : str
        Description for manifest.
    
    Returns
    -------
    str
        JSON filename containing all calculation results.
    """
    from pathlib import Path
    import json
    
    # Validate inputs
    if activity_col is not None and activity_type is None:
        raise ValueError(
            "activity_type must be specified when activity_col is provided. "
            "Valid options: 'classification' or 'regression'"
        )
    if activity_type is not None and activity_type not in ['classification', 'regression']:
        raise ValueError(
            f"activity_type must be 'classification' or 'regression', got '{activity_type}'"
        )
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset")
    
    has_activity = activity_col is not None and activity_col in df.columns
    
    n_molecules = len(df)
    
    # Track temporary files for cleanup
    temp_files = []
    
    try:
        # ========================================================================
        # Run all analyses by calling existing tools and helpers
        # ========================================================================
        
        # 1. Dataset overview (use get_dataset_summary)
        summary_result = get_dataset_summary(project_manifest_path, input_filename)
        column_stats = summary_result['column_summaries']
        
        # 2. Completeness analysis (from get_dataset_summary)
        completeness = {
            col: {
                'missing': stats.get('n_missing', 0),
                'pct_missing': (stats.get('n_missing', 0) / n_molecules * 100) if n_molecules > 0 else 0
            }
            for col, stats in column_stats.items()
        }
        
        # 3. SMILES validity (unique helper)
        validity_result = _analyze_smiles_validity(df, smiles_col)
        
        # 4. PAINS screening (existing tool)
        # Note: filter_by_pains modifies dataset, so we just analyze the original
        # MolFromSmiles handles canonicalization internally
        pains_smiles = df[smiles_col].dropna().tolist()
        n_pains = 0
        pains_examples = []
        for idx, smi in enumerate(pains_smiles[:1000]):  # Sample for performance
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol:
                # Check PAINS patterns (simplified - full analysis would call filter_by_pains)
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                catalog = FilterCatalog(params)
                matches = catalog.GetMatches(mol)
                if len(matches) > 0:
                    n_pains += 1
                    if len(pains_examples) < 3:
                        pains_examples.append({
                            'smiles': smi,
                            'patterns': [m.GetDescription() for m in matches]
                        })
        
        pains_result = {
            'n_pains': int(n_pains),
            'pct_pains': float(n_pains / len(pains_smiles) * 100) if len(pains_smiles) > 0 else 0,
            'examples': pains_examples
        }
        
        # 5. Duplicates (existing tool)
        dup_temp_name = f"{output_name}_duplicates_temp"
        dup_result = find_duplicates_dataset(
            input_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_col=smiles_col,
            output_filename=dup_temp_name,
            explanation="Temporary duplicate analysis for quality report",
            label_col=activity_col if has_activity else None,
            is_binary_label=(activity_type == 'classification')
        )
        temp_files.append(dup_result['output_filename'])
        
        # 6. Physicochemical properties (existing tool)
        # Create temporary output for descriptors
        desc_temp_name = f"{output_name}_descriptors_temp"
        desc_result = calculate_simple_descriptors(
            input_filename=input_filename,
            smiles_column=smiles_col,
            descriptor_names=['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 
                             'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount', 'FractionCSP3'],
            project_manifest_path=project_manifest_path,
            output_filename=desc_temp_name,
            explanation="Temporary descriptors for quality report"
        )
        temp_files.append(desc_result['output_filename'])
        
        # Load descriptor dataset
        df_with_desc = _load_resource(project_manifest_path, desc_result['output_filename'])
        
        # Calculate property statistics
        prop_cols = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 
                     'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount', 'FractionCSP3']
        property_stats = {}
        for col in prop_cols:
            if col in df_with_desc.columns:
                data = df_with_desc[col].dropna()
                if len(data) > 0:
                    property_stats[col] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'q25': float(data.quantile(0.25)),
                        'q75': float(data.quantile(0.75))
                    }
        
        # 7. Lipinski analysis (existing tool - use filtering logic)
        lipinski_violations = []
        for idx, row in df_with_desc.iterrows():
            n_viol = 0
            if pd.notna(row.get('MolWt')) and row['MolWt'] > 500:
                n_viol += 1
            if pd.notna(row.get('MolLogP')) and row['MolLogP'] > 5:
                n_viol += 1
            if pd.notna(row.get('NumHDonors')) and row['NumHDonors'] > 5:
                n_viol += 1
            if pd.notna(row.get('NumHAcceptors')) and row['NumHAcceptors'] > 10:
                n_viol += 1
            lipinski_violations.append(n_viol)
        
        lipinski_result = {
            '0_violations': int(lipinski_violations.count(0)),
            '1_violation': int(lipinski_violations.count(1)),
            '2+_violations': int(sum(1 for v in lipinski_violations if v >= 2)),
            'pct_compliant': float(lipinski_violations.count(0) / len(lipinski_violations) * 100) if len(lipinski_violations) > 0 else 0
        }
        
        # 8. Veber analysis
        veber_pass = 0
        for idx, row in df_with_desc.iterrows():
            tpsa_ok = pd.notna(row.get('TPSA')) and row['TPSA'] <= 140
            rotbonds_ok = pd.notna(row.get('NumRotatableBonds')) and row['NumRotatableBonds'] <= 10
            if tpsa_ok and rotbonds_ok:
                veber_pass += 1
        
        veber_result = {
            'pass': int(veber_pass),
            'pct_compliant': float(veber_pass / n_molecules * 100) if n_molecules > 0 else 0
        }
        
        # 9. QED analysis (existing tool logic)
        qed_scores = []
        for smi in df[smiles_col]:
            if pd.notna(smi) and isinstance(smi, str):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    qed_scores.append(QED.qed(mol))
                else:
                    qed_scores.append(np.nan)
            else:
                qed_scores.append(np.nan)
        
        qed_array = np.array([q for q in qed_scores if not np.isnan(q)])
        if len(qed_array) > 0:
            qed_result = {
                'mean': float(qed_array.mean()),
                'median': float(np.median(qed_array)),
                'high': int(np.sum(qed_array >= 0.7)),
                'moderate': int(np.sum((qed_array >= 0.5) & (qed_array < 0.7))),
                'low': int(np.sum(qed_array < 0.5))
            }
        else:
            qed_result = None
        
        # 10. Outlier detection (existing tool)
        outlier_temp_name = f"{output_name}_outliers_temp"
        outlier_result = detect_outliers_iqr(
            input_filename=desc_result['output_filename'],
            project_manifest_path=project_manifest_path,
            columns=prop_cols,
            output_filename=outlier_temp_name,
            explanation="Temporary outlier detection",
            multiplier=1.5
        )
        temp_files.append(outlier_result['output_filename'])
        
        # 11. Activity analysis (optional, unique helper)
        activity_result = None
        activity_correlation_result = None
        if has_activity:
            activity_values = df[activity_col].values
            activity_result = _analyze_activity_distribution(
                activity_values,
                activity_type=activity_type,
                units=activity_units
            )
            
            # Perform correlation analysis
            activity_correlation_result = _analyze_activity_correlations(
                df,
                smiles_col,
                activity_col,
                activity_type
            )
        
        # 12. Scaffold diversity (existing tool)
        scaffold_temp_name = f"{output_name}_scaffold_temp"
        scaffold_result = scaffold_analysis(
            dataset_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_column=smiles_col,
            output_filename=scaffold_temp_name,
            scaffold_type='bemis_murcko',
            activity_column=activity_col if has_activity else None,
            activity_type=activity_type if has_activity else None,
            explanation="Temporary scaffold analysis for quality report"
        )
        temp_files.append(scaffold_result.get('report_json_filename', ''))
        temp_files.append(scaffold_result.get('report_text_filename', ''))
        temp_files.append(scaffold_result.get('dataset_with_scaffolds_filename', ''))
        
        # Extract key scaffold metrics
        scaffold_metrics = {
            'n_unique_scaffolds': scaffold_result['n_unique_scaffolds'],
            'diversity_ratio': scaffold_result['diversity_ratio'],
            'gini_coefficient': scaffold_result['gini_coefficient'],
            'shannon_entropy': scaffold_result['shannon_entropy']
        }
        
        # 13. Functional group analysis (unique helper)
        smiles_list = df[smiles_col].tolist()
        functional_groups_result = _analyze_functional_groups(smiles_list)
        
        # 14. Stereochemistry analysis (unique helper)
        stereochemistry_result = _analyze_stereochemistry(smiles_list)
        
        # 15. Charge state analysis (unique helper)
        charge_state_result = _analyze_charge_state(smiles_list)
        
        # 16. Salts/fragments/solvents analysis (unique helper)
        # This already canonicalizes SMILES internally for accurate detection
        salts_fragments_result = _analyze_salts_fragments_solvents(smiles_list)
        
        # 17. Special features analysis (unique helper)
        special_features_result = _analyze_special_features(smiles_list)
        
        # 18. Generate recommendations
        recommendations = []
        critical_issues = []
        
        # Calculate metrics for recommendations
        avg_missing = np.mean([stats['pct_missing'] for stats in completeness.values()])
        dup_pct = (dup_result.get('n_duplicates', 0) / n_molecules * 100) if n_molecules > 0 else 0
        
        if validity_result['pct_invalid'] > 5:
            recommendations.append("1. Remove invalid SMILES using clean_smiles tool")
            critical_issues.append(f"High invalid SMILES rate: {validity_result['pct_invalid']:.1f}%")
        
        if pains_result['pct_pains'] > 10:
            recommendations.append("2. Filter PAINS patterns using filter_by_pains tool")
            critical_issues.append(f"High PAINS rate: {pains_result['pct_pains']:.1f}%")
        
        if dup_result.get('n_duplicates', 0) / n_molecules > 0.1:
            recommendations.append("3. Deduplicate dataset using deduplicate_dataset tool")
            critical_issues.append(f"High duplicate rate: {dup_pct:.1f}%")
        
        if lipinski_result['pct_compliant'] < 50:
            recommendations.append("4. Consider filtering by Lipinski rules using filter_by_lipinski_ro5")
            critical_issues.append(f"Low Lipinski compliance: {lipinski_result['pct_compliant']:.1f}%")
        
        if avg_missing > 10:
            recommendations.append("5. Address missing values in key columns")
            critical_issues.append(f"High missing data rate: {avg_missing:.1f}%")
        
        if charge_state_result['needs_neutralization']:
            recommendations.append("6. Neutralize charged molecules using neutralize_charges tool")
            critical_issues.append(f"High charged molecule rate: {charge_state_result['pct_charged']:.1f}%")
        
        if salts_fragments_result['needs_desalting']:
            recommendations.append("7. Remove salts and fragments using desalt and defragment tools")
            critical_issues.append(f"High fragmented molecule rate: {salts_fragments_result['pct_fragmented']:.1f}%")
        
        if special_features_result['organometallic']['pct'] > 5:
            recommendations.append("8. Review organometallic compounds - may require specialized handling")
            critical_issues.append(f"Organometallic compounds detected: {special_features_result['organometallic']['pct']:.1f}%")
        
        if special_features_result['isotopes']['pct'] > 1:
            recommendations.append("9. Review non-standard isotopes - may affect descriptor calculations")
            critical_issues.append(f"Non-standard isotopes detected: {special_features_result['isotopes']['pct']:.1f}%")
        
        # ========================================================================
        # Store calculation results as JSON
        # ========================================================================
        
        calc_data = {
            'metadata': {
                'dataset': input_filename,
                'generated': datetime.now().isoformat(),
                'n_molecules': n_molecules,
                'smiles_col': smiles_col,
                'activity_col': activity_col,
                'activity_type': activity_type if has_activity else None,
                'columns': df.columns.tolist()
            },
            'completeness': completeness,
            'smiles_validity': validity_result,
            'pains': pains_result,
            'duplicates': dup_result,
            'physicochemical_properties': property_stats,
            'lipinski': lipinski_result,
            'veber': veber_result,
            'qed': qed_result,
            'outliers': {
                'n_extreme': outlier_result.get('n_extreme_molecules', 0),
                'examples': outlier_result.get('extreme_molecules', [])
            },
            'activity': activity_result,
            'activity_correlations': activity_correlation_result,
            'scaffold_diversity': scaffold_metrics,
            'functional_groups': functional_groups_result,
            'stereochemistry': stereochemistry_result,
            'charge_state': charge_state_result,
            'salts_fragments': salts_fragments_result,
            'special_features': special_features_result,
            'recommendations': recommendations,
            'critical_issues': critical_issues
        }
        
        # Store calculation results
        json_filename = _store_resource(
            calc_data,
            project_manifest_path,
            f"{output_name}_calculations",
            explanation,
            'json'
        )
        
    finally:
        # Clean up all temporary files
        manifest_path = Path(project_manifest_path)
        manifest_dir = manifest_path.parent
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Remove temp files from manifest and filesystem
        # Only delete files that are in the temp_files list
        for temp_file in temp_files:
            if temp_file:  # Skip empty strings
                # Delete file from filesystem
                file_path = manifest_dir / temp_file
                if file_path.exists():
                    file_path.unlink()
        
        # Update manifest - keep only resources NOT in temp_files
        resources_to_keep = []
        for resource in manifest['resources']:
            if resource['filename'] not in temp_files:
                resources_to_keep.append(resource)
        
        # Update manifest
        manifest['resources'] = resources_to_keep
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    return json_filename


def data_quality_analysis(
    input_filename: str,
    project_manifest_path: str,
    smiles_col: str = 'SMILES',
    activity_col: Optional[str] = None,
    activity_type: Optional[str] = None,
    activity_units: str = 'nM',
    output_name: str = 'quality_report',
    explanation: str = 'Generated comprehensive data quality report'
) -> Dict:
    """
    🚀 PRIMARY TOOL FOR COMPREHENSIVE DATA QUALITY ASSESSMENT 🚀
    
    Comprehensive molecular dataset quality analysis with 19 diagnostic sections:
    1. **Dataset Overview**: Basic statistics and column information
    2. **Data Completeness**: Missing value analysis per column
    3. **SMILES Validity**: Invalid/unparseable molecular structures
    4. **PAINS Patterns**: Pan-Assay Interference Compounds detection
    5. **Duplicates & Conflicts**: Identical molecules with conflicting labels
    6. **Physicochemical Properties**: MW, LogP, H-bonds, TPSA, rotatable bonds, etc.
    7. **Lipinski Rule of Five**: Oral bioavailability prediction
    8. **Veber Rules**: Additional drug-likeness criteria
    9. **QED Drug-Likeness**: Quantitative drug-likeness scoring
    10. **Outlier Detection**: Molecules with unusual properties
    11. **Activity Distribution**: Classification balance or regression range analysis
    12. **Activity Correlations**: Property-activity relationships
    13. **Scaffold Diversity**: Chemical space coverage metrics
    14. **Functional Group Analysis**: Common structural motifs
    15. **Stereochemistry Analysis**: Chiral centers and E/Z bonds
    16. **Charge State Analysis**: Formal charges and zwitterions
    17. **Salts/Fragments/Solvents**: Multi-component entries detection
    18. **Special Features**: Organometallics, isotopes, unusual rings
    19. **Recommended Workflow**: Prioritized cleaning steps
    
    Produces both a detailed JSON report with all calculations and a human-readable
    text report with interpretations and actionable recommendations.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    project_manifest_path : str
        Path to manifest.json.
    smiles_col : str, default='SMILES'
        Column containing SMILES strings.
    activity_col : str, optional
        Column containing bioactivity values.
    activity_type : str, optional
        'classification' or 'regression'. Required if activity_col provided.
    activity_units : str, default='nM'
        Units for continuous activity data.
    output_name : str, default='quality_report'
        Prefix for output report files.
    explanation : str
        Description for manifest.
    
    Returns
    -------
    dict
        Contains report_text, calculations_json, report_txt, n_issues, critical_issues, key_metrics.
    """
    # Step 1: Perform all calculations
    json_filename = _perform_quality_report_calculations(
        input_filename=input_filename,
        project_manifest_path=project_manifest_path,
        smiles_col=smiles_col,
        activity_col=activity_col,
        activity_type=activity_type,
        activity_units=activity_units,
        output_name=output_name,
        explanation=f"{explanation} - calculations"
    )
    
    # Step 2: Load calculation results
    calc_data = _load_resource(project_manifest_path, json_filename)
    
    # Extract data from calculation results
    metadata = calc_data['metadata']
    n_molecules = metadata['n_molecules']
    has_activity = metadata['activity_col'] is not None
    
    completeness = calc_data['completeness']
    validity_result = calc_data['smiles_validity']
    pains_result = calc_data['pains']
    dup_result = calc_data['duplicates']
    property_stats = calc_data['physicochemical_properties']
    lipinski_result = calc_data['lipinski']
    veber_result = calc_data['veber']
    qed_result = calc_data['qed']
    outlier_result = calc_data['outliers']
    activity_result = calc_data['activity']
    activity_correlation_result = calc_data['activity_correlations']
    scaffold_metrics = calc_data['scaffold_diversity']
    functional_groups_result = calc_data['functional_groups']
    stereochemistry_result = calc_data['stereochemistry']
    charge_state_result = calc_data['charge_state']
    salts_fragments_result = calc_data['salts_fragments']
    special_features_result = calc_data['special_features']
    recommendations = calc_data['recommendations']
    critical_issues = calc_data['critical_issues']
    
    # ========================================================================
    # Generate formatted text report (14 sections)
    # ========================================================================
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("MOLECULAR DATASET QUALITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Dataset: {metadata['dataset']}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Section 1: Overview
    report_lines.append("-" * 80)
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total molecules: {n_molecules}")
    report_lines.append(f"Columns: {', '.join(metadata['columns'])}")
    report_lines.append(f"SMILES column: {metadata['smiles_col']}")
    if has_activity:
        report_lines.append(f"Activity column: {metadata['activity_col']} ({metadata['activity_type']})")
    else:
        report_lines.append("Activity column: None")
    report_lines.append("")
    
    # Section 2: Completeness
    report_lines.append("-" * 80)
    report_lines.append("2. DATA COMPLETENESS")
    report_lines.append("-" * 80)
    report_lines.append("Checks for missing values in each column. High missing rates may indicate data quality issues.")
    report_lines.append("")
    for col, stats in completeness.items():
        status = "✓" if stats['pct_missing'] < 5 else "⚠" if stats['pct_missing'] < 20 else "✗"
        report_lines.append(f"{status} {col}: {stats['missing']} missing ({stats['pct_missing']:.1f}%)")
    report_lines.append("")
    
    # Section 3: SMILES Validity
    report_lines.append("-" * 80)
    report_lines.append("3. SMILES VALIDITY")
    report_lines.append("-" * 80)
    report_lines.append("SMILES are text representations of molecular structures. Invalid SMILES cannot be processed.")
    report_lines.append("")
    report_lines.append(f"Valid SMILES: {validity_result['n_valid']} ({validity_result['pct_valid']:.1f}%)")
    report_lines.append(f"Invalid SMILES: {validity_result['n_invalid']} ({validity_result['pct_invalid']:.1f}%)")
    if validity_result['invalid_examples']:
        report_lines.append("Examples of invalid SMILES:")
        for ex in validity_result['invalid_examples'][:3]:
            report_lines.append(f"  - Row {ex['row']}: {ex['reason']}")
    report_lines.append("")
    
    # Section 4: PAINS
    report_lines.append("-" * 80)
    report_lines.append("4. PAINS PATTERNS")
    report_lines.append("-" * 80)
    report_lines.append("PAINS (Pan-Assay Interference Compounds) are molecules that might show false positives in assays.")
    report_lines.append("")
    report_lines.append(f"Molecules with PAINS: {pains_result['n_pains']} ({pains_result['pct_pains']:.1f}%)")
    if pains_result['examples']:
        report_lines.append("Example PAINS patterns:")
        for ex in pains_result['examples']:
            report_lines.append(f"  - {', '.join(ex['patterns'])}")
    report_lines.append("")
    
    # Section 5: Duplicates
    report_lines.append("-" * 80)
    report_lines.append("5. DUPLICATES & CONFLICTS")
    report_lines.append("-" * 80)
    report_lines.append("Identifies molecules that appear multiple times. Activity conflicts occur when the same")
    report_lines.append("molecule has different activity values, which may indicate measurement variability.")
    report_lines.append("")
    report_lines.append(f"Unique SMILES: {dup_result['n_unique']}")
    report_lines.append(f"Duplicate entries: {dup_result.get('n_duplicates', 0)}")
    if has_activity and 'conflict_summary' in dup_result:
        conflict_info = dup_result['conflict_summary']
        report_lines.append(f"Activity conflicts: {conflict_info.get('n_conflicts', 0)}")
        if conflict_info.get('n_conflicts', 0) > 0:
            report_lines.append(f"  Max fold difference: {conflict_info.get('max_fold_diff', 'N/A'):.2f}")
    report_lines.append("")
    
    # Section 6: Physicochemical Properties
    report_lines.append("-" * 80)
    report_lines.append("6. PHYSICOCHEMICAL PROPERTIES")
    report_lines.append("-" * 80)
    report_lines.append("Basic molecular properties that influence drug-like behavior and biological activity.")
    report_lines.append("")
    prop_display_names = {
        'MolWt': 'Molecular Weight',
        'MolLogP': 'LogP',
        'NumHDonors': 'H-Bond Donors',
        'NumHAcceptors': 'H-Bond Acceptors',
        'TPSA': 'TPSA',
        'NumRotatableBonds': 'Rotatable Bonds',
        'NumAromaticRings': 'Aromatic Rings',
        'HeavyAtomCount': 'Heavy Atoms',
        'FractionCSP3': 'Fraction Csp3'
    }
    for prop, display_name in prop_display_names.items():
        if prop in property_stats:
            stats = property_stats[prop]
            report_lines.append(f"{display_name}:")
            report_lines.append(f"  Mean ± SD: {stats['mean']:.2f} ± {stats['std']:.2f}")
            report_lines.append(f"  Median [Q1-Q3]: {stats['median']:.2f} [{stats['q25']:.2f}-{stats['q75']:.2f}]")
            report_lines.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
    report_lines.append("")
    
    # Section 7: Lipinski
    report_lines.append("-" * 80)
    report_lines.append("7. LIPINSKI RULE OF FIVE")
    report_lines.append("-" * 80)
    report_lines.append("A set of rules predicting oral bioavailability: MW≤500, LogP≤5, H-donors≤5, H-acceptors≤10.")
    report_lines.append("Molecules with 0-1 violations are typically considered drug-like.")
    report_lines.append("")
    report_lines.append(f"0 violations: {lipinski_result['0_violations']} ({lipinski_result['pct_compliant']:.1f}%)")
    report_lines.append(f"1 violation: {lipinski_result['1_violation']}")
    report_lines.append(f"2+ violations: {lipinski_result['2+_violations']}")
    report_lines.append("")
    
    # Section 8: Veber
    report_lines.append("-" * 80)
    report_lines.append("8. VEBER RULES")
    report_lines.append("-" * 80)
    report_lines.append("Additional oral bioavailability rules: rotatable bonds≤10 and TPSA≤140.")
    report_lines.append("Complements Lipinski rules for predicting drug-like properties.")
    report_lines.append("")
    report_lines.append(f"Compliant: {veber_result['pass']} ({veber_result['pct_compliant']:.1f}%)")
    report_lines.append("")
    
    # Section 9: QED
    report_lines.append("-" * 80)
    report_lines.append("9. QED (DRUG-LIKENESS)")
    report_lines.append("-" * 80)
    report_lines.append("Quantitative Estimate of Drug-likeness (QED) scores molecules from 0 to 1.")
    report_lines.append("Higher scores (≥0.7) indicate better drug-like properties. Take with a grain of salt.")
    report_lines.append("")
    if qed_result:
        report_lines.append(f"Mean QED: {qed_result['mean']:.3f}")
        report_lines.append(f"Median QED: {qed_result['median']:.3f}")
        report_lines.append(f"High (≥0.7): {qed_result['high']}")
        report_lines.append(f"Moderate (0.5-0.7): {qed_result['moderate']}")
        report_lines.append(f"Low (<0.5): {qed_result['low']}")
    else:
        report_lines.append("QED analysis not available")
    report_lines.append("")
    
    # Section 10: Outliers
    report_lines.append("-" * 80)
    report_lines.append("10. OUTLIER DETECTION")
    report_lines.append("-" * 80)
    report_lines.append("Identifies molecules with unusual physicochemical properties that differ significantly from")
    report_lines.append("the dataset distribution using the Interquartile Range (IQR) method.")
    report_lines.append("")
    report_lines.append(f"Molecules with extreme properties: {outlier_result.get('n_extreme', 0)}")
    if outlier_result.get('examples'):
        report_lines.append("Top outliers:")
        for mol_info in outlier_result['examples'][:3]:
            report_lines.append(f"  - Row {mol_info['index']}: {mol_info['n_outlier_properties']} outlier properties")
    report_lines.append("")
    
    # Section 11: Activity (optional)
    if activity_result:
        report_lines.append("-" * 80)
        report_lines.append("11. ACTIVITY DISTRIBUTION")
        report_lines.append("-" * 80)
        report_lines.append("Analysis of bioactivity values. For classification: checks class balance.")
        report_lines.append("For regression: examines value distribution and range.")
        report_lines.append("")
        if activity_result['type'] == 'classification':
            report_lines.append(f"Positive: {activity_result['n_positive']}")
            report_lines.append(f"Negative: {activity_result['n_negative']}")
            report_lines.append(f"Balance: {activity_result['balance']:.2%}")
        else:
            linear = activity_result['linear_stats']
            log = activity_result['log_stats']
            report_lines.append(f"Units: {activity_result['units']}")
            report_lines.append(f"Range: {linear['min']:.2f} - {linear['max']:.2f} ({linear['range_log_units']:.1f} log units)")
            report_lines.append(f"Median: {linear['median']:.2f}")
            report_lines.append(f"Geometric mean: {linear['geometric_mean']:.2f}")
            if activity_result.get('normality_test'):
                norm = activity_result['normality_test']
                report_lines.append(f"Log-scale normality: {'Yes' if norm['is_normal'] else 'No'} (p={norm['p_value']:.3f})")
        report_lines.append("")
    
    # Section 12: Activity Correlations (optional)
    if activity_correlation_result:
        report_lines.append("-" * 80)
        report_lines.append("12. ACTIVITY CORRELATIONS")
        report_lines.append("-" * 80)
        report_lines.append("Analysis of relationships between molecular features and bioactivity.")
        
        if activity_correlation_result['analysis_type'] == 'regression':
            report_lines.append("Correlation analysis (Spearman rank correlation):")
            report_lines.append("")
            
            corr = activity_correlation_result
            report_lines.append(f"Samples analyzed: {corr['n_samples']}")
            report_lines.append("")
            
            if corr['significant_correlations']:
                report_lines.append(f"Significant correlations (p<0.05): {len(corr['significant_correlations'])}")
                report_lines.append("")
                report_lines.append("Top correlations with activity:")
                for i, feat in enumerate(corr['top_correlations'][:8], 1):
                    sig = "*" if feat['p_value'] < 0.05 else " "
                    report_lines.append(f"  {i}. {feat['feature']}: r={feat['correlation']:+.3f} (p={feat['p_value']:.4f}){sig}")
                report_lines.append("")
                report_lines.append("* = statistically significant (p<0.05)")
                report_lines.append("Positive correlation: higher values → higher activity")
                report_lines.append("Negative correlation: higher values → lower activity")
            else:
                report_lines.append("⚠ No statistically significant correlations found")
                report_lines.append("This may indicate:")
                report_lines.append("  - Complex non-linear relationships")
                report_lines.append("  - Insufficient sample size")
                report_lines.append("  - Activity not driven by simple physicochemical properties")
        
        else:  # classification
            report_lines.append("Feature enrichment analysis (Mann-Whitney U test):")
            report_lines.append("")
            
            diff = activity_correlation_result
            report_lines.append(f"Active compounds: {diff['n_active']}")
            report_lines.append(f"Inactive compounds: {diff['n_inactive']}")
            report_lines.append("")
            
            if diff['significant_differences']:
                report_lines.append(f"Significant differences (p<0.05): {len(diff['significant_differences'])}")
                report_lines.append("")
                report_lines.append("Top differentiating features:")
                for i, feat in enumerate(diff['top_differences'][:8], 1):
                    sig = "*" if feat['p_value'] < 0.05 else " "
                    if feat['type'] == 'property':
                        report_lines.append(f"  {i}. {feat['feature']}: {feat['direction'].replace('_', ' ')} (p={feat['p_value']:.4f}){sig}")
                    else:
                        report_lines.append(f"  {i}. {feat['feature']}: {feat['pct_active']:.1f}% active vs {feat['pct_inactive']:.1f}% inactive (p={feat['p_value']:.4f}){sig}")
                report_lines.append("")
                report_lines.append("* = statistically significant (p<0.05)")
            else:
                report_lines.append("⚠ No statistically significant differences found")
                report_lines.append("Active and inactive compounds may have similar property distributions")
        
        report_lines.append("")
    
    # Section 13: Scaffold Diversity
    report_lines.append("-" * 80)
    report_lines.append("13. SCAFFOLD DIVERSITY")
    report_lines.append("-" * 80)
    report_lines.append("Scaffolds are core molecular frameworks. Higher diversity indicates a more varied chemical")
    report_lines.append("space, important for avoiding overfitting in machine learning models.")
    report_lines.append("")
    report_lines.append(f"Unique scaffolds: {scaffold_metrics['n_unique_scaffolds']}")
    report_lines.append(f"Diversity ratio: {scaffold_metrics['diversity_ratio']:.3f}")
    report_lines.append(f"Gini coefficient: {scaffold_metrics['gini_coefficient']:.3f}")
    report_lines.append(f"Shannon entropy: {scaffold_metrics['shannon_entropy']:.2f}")
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append(f"  - Diversity ratio: {'High' if scaffold_metrics['diversity_ratio'] > 0.5 else 'Moderate' if scaffold_metrics['diversity_ratio'] > 0.3 else 'Low'} scaffold diversity")
    report_lines.append(f"  - Gini coefficient: {'Low' if scaffold_metrics['gini_coefficient'] < 0.3 else 'Moderate' if scaffold_metrics['gini_coefficient'] < 0.6 else 'High'} inequality")
    report_lines.append(f"  - Shannon entropy: {'High' if scaffold_metrics['shannon_entropy'] > 4 else 'Moderate' if scaffold_metrics['shannon_entropy'] > 2 else 'Low'} diversity")
    report_lines.append("")
    
    # Section 14: Functional Groups
    report_lines.append("-" * 80)
    report_lines.append("14. FUNCTIONAL GROUP ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("Shows the most common chemical functional groups in your dataset.")
    report_lines.append("")
    top_groups = sorted(
        [(name, stats['count']) for name, stats in functional_groups_result.items() if name != 'halogen_breakdown'],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for name, count in top_groups:
        pct = functional_groups_result[name]['pct_dataset']
        report_lines.append(f"{name}: {count} molecules ({pct:.1f}%)")
    report_lines.append("")
    
    # Section 15: Stereochemistry
    report_lines.append("-" * 80)
    report_lines.append("15. STEREOCHEMISTRY ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("Analyzes stereochemical information including chiral centers and E/Z double bonds.")
    report_lines.append("Specified stereochemistry is important for accurate biological activity prediction.")
    report_lines.append("")
    
    stereo = stereochemistry_result
    report_lines.append(f"Molecules with stereochemistry: {stereo['n_with_stereochemistry']} ({stereo['pct_with_stereochemistry']:.1f}%)")
    report_lines.append("")
    
    report_lines.append("Chiral Centers:")
    chiral = stereo['chiral_centers']
    report_lines.append(f"  Total chiral centers: {chiral['total']}")
    report_lines.append(f"  Specified (R/S): {chiral['specified']} ({chiral['pct_specified']:.1f}%)")
    report_lines.append(f"  Unspecified: {chiral['unspecified']}")
    report_lines.append(f"  Molecules with chiral centers: {chiral['n_molecules_with_chiral']}")
    report_lines.append("")
    
    report_lines.append("E/Z Double Bonds:")
    db = stereo['double_bonds']
    report_lines.append(f"  Total stereogenic double bonds: {db['total']}")
    report_lines.append(f"  Specified (E/Z): {db['specified']} ({db['pct_specified']:.1f}%)")
    report_lines.append(f"  Unspecified: {db['unspecified']}")
    report_lines.append(f"  Molecules with stereogenic bonds: {db['n_molecules_with_bonds']}")
    report_lines.append("")
    
    report_lines.append("Stereochemical Completeness:")
    comp = stereo['completeness']
    report_lines.append(f"  Fully specified: {comp['fully_specified']} ({comp['pct_fully_specified']:.1f}% of molecules with stereo)")
    report_lines.append(f"  Partially specified: {comp['partially_specified']}")
    report_lines.append(f"  No stereochemistry: {comp['no_stereochemistry']}")
    report_lines.append("")
    
    # Section 16: Charge State
    report_lines.append("-" * 80)
    report_lines.append("16. CHARGE STATE ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("Analyzes formal charges, zwitterions, and charge distribution.")
    report_lines.append("Most ML models expect neutral molecules, so charged species may need neutralization.")
    report_lines.append("")
    
    charge = charge_state_result
    report_lines.append(f"Neutral molecules: {charge['n_neutral']} ({charge['pct_neutral']:.1f}%)")
    report_lines.append(f"Charged molecules: {charge['n_charged']} ({charge['pct_charged']:.1f}%)")
    report_lines.append("")
    
    report_lines.append("Charge Types:")
    ct = charge['charge_types']
    report_lines.append(f"  Positive only: {ct['positive_only']}")
    report_lines.append(f"  Negative only: {ct['negative_only']}")
    report_lines.append(f"  Zwitterions: {ct['zwitterions']} ({ct['pct_zwitterions']:.1f}%)")
    report_lines.append("")
    
    report_lines.append("Charge Distribution:")
    tc = charge['total_charges']
    report_lines.append(f"  Total positive charges: {tc['positive']}")
    report_lines.append(f"  Total negative charges: {tc['negative']}")
    report_lines.append("")
    
    if charge['needs_neutralization']:
        report_lines.append("⚠ NEUTRALIZATION RECOMMENDED")
        report_lines.append(f"  {charge['pct_charged']:.1f}% of molecules are charged")
        report_lines.append("  Consider using neutralize_charges tool to prepare for ML modeling")
    else:
        report_lines.append("✓ Most molecules are neutral")
    
    if charge['examples']['positive']:
        report_lines.append("")
        report_lines.append("Example positive ions:")
        for ex in charge['examples']['positive'][:2]:
            report_lines.append(f"  - Charge {ex['charge']:+d}: {ex['smiles']}")
    
    if charge['examples']['negative']:
        report_lines.append("")
        report_lines.append("Example negative ions:")
        for ex in charge['examples']['negative'][:2]:
            report_lines.append(f"  - Charge {ex['charge']:+d}: {ex['smiles']}")
    
    if charge['examples']['zwitterion']:
        report_lines.append("")
        report_lines.append("Example zwitterions:")
        for ex in charge['examples']['zwitterion'][:2]:
            report_lines.append(f"  - Net charge {ex['net_charge']:+d} (+{ex['pos_charges']}/-{ex['neg_charges']}): {ex['smiles']}")
    
    report_lines.append("")
    
    # Section 17: Salts/Fragments/Solvents
    report_lines.append("-" * 80)
    report_lines.append("17. SALTS/FRAGMENTS/SOLVENTS")
    report_lines.append("-" * 80)
    report_lines.append("Detects multi-component entries (containing '.'), common salts, and solvents.")
    report_lines.append("These should typically be removed before ML modeling using desalt/defragment tools.")
    report_lines.append("")
    
    sf = salts_fragments_result
    report_lines.append(f"Fragmented entries: {sf['n_fragmented']} ({sf['pct_fragmented']:.1f}%)")
    report_lines.append(f"Multi-component entries: {sf['n_multi_component']} ({sf['pct_multi_component']:.1f}%)")
    report_lines.append("")
    
    if sf['fragment_stats']['max_fragments'] > 1:
        report_lines.append("Fragment Statistics:")
        report_lines.append(f"  Maximum fragments in single entry: {sf['fragment_stats']['max_fragments']}")
        report_lines.append(f"  Average fragments per entry: {sf['fragment_stats']['avg_fragments']:.2f}")
        report_lines.append("")
    
    if sf['salts_detected']:
        report_lines.append(f"Common Salts Detected ({sf['n_salt_types']} types):")
        for salt_name, count in list(sf['salts_detected'].items())[:5]:
            report_lines.append(f"  - {salt_name}: {count}")
        report_lines.append("")
    
    if sf['solvents_detected']:
        report_lines.append(f"Common Solvents Detected ({sf['n_solvent_types']} types):")
        for solv_name, count in list(sf['solvents_detected'].items())[:5]:
            report_lines.append(f"  - {solv_name}: {count}")
        report_lines.append("")
    
    if sf['needs_desalting']:
        report_lines.append("⚠ DESALTING/DEFRAGMENTATION RECOMMENDED")
        if sf['pct_fragmented'] > 10:
            report_lines.append(f"  {sf['pct_fragmented']:.1f}% of molecules contain fragments")
        if sf['n_salt_types'] > 0:
            report_lines.append(f"  {sf['n_salt_types']} different salt types detected")
        report_lines.append("  Consider using desalt and defragment tools")
    else:
        report_lines.append("✓ Most molecules are single-component")
    
    if sf['examples']['with_salts']:
        report_lines.append("")
        report_lines.append("Examples with salts:")
        for ex in sf['examples']['with_salts'][:2]:
            report_lines.append(f"  - {', '.join(ex['salts'])} ({ex['n_fragments']} fragments): {ex['smiles']}")
    
    if sf['examples']['with_solvents']:
        report_lines.append("")
        report_lines.append("Examples with solvents:")
        for ex in sf['examples']['with_solvents'][:2]:
            report_lines.append(f"  - {', '.join(ex['solvents'])} ({ex['n_fragments']} fragments): {ex['smiles']}")
    
    report_lines.append("")
    
    # Section 18: Special Features
    report_lines.append("-" * 80)
    report_lines.append("18. SPECIAL FEATURES")
    report_lines.append("-" * 80)
    report_lines.append("Analysis of organometallic compounds, non-standard isotopes, and ring size distribution.")
    report_lines.append("These features may require special consideration in modeling or data processing.")
    report_lines.append("")
    
    sp = special_features_result
    
    # Organometallic compounds
    report_lines.append("Organometallic Compounds:")
    org = sp['organometallic']
    report_lines.append(f"  Count: {org['count']} ({org['pct']:.1f}%)")
    if org['metals_found']:
        report_lines.append(f"  Metal types detected: {org['n_metal_types']}")
        report_lines.append("  Most common metals:")
        for metal, count in list(org['metals_found'].items())[:5]:
            report_lines.append(f"    - {metal}: {count}")
    if sp['examples']['organometallic']:
        report_lines.append("  Examples:")
        for ex in sp['examples']['organometallic'][:2]:
            report_lines.append(f"    - {', '.join(ex['metals'])}: {ex['smiles']}")
    report_lines.append("")
    
    # Non-standard isotopes
    report_lines.append("Non-Standard Isotopes:")
    iso = sp['isotopes']
    report_lines.append(f"  Count: {iso['count']} ({iso['pct']:.1f}%)")
    if iso['isotopes_found']:
        report_lines.append(f"  Isotope types detected: {iso['n_isotope_types']}")
        if len(iso['isotopes_found']) <= 10:
            report_lines.append("  Isotopes found:")
            for isotope, count in sorted(iso['isotopes_found'].items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"    - {isotope}: {count}")
    if sp['examples']['isotopes']:
        report_lines.append("  Examples:")
        for ex in sp['examples']['isotopes'][:2]:
            report_lines.append(f"    - {', '.join(ex['isotopes'])}: {ex['smiles']}")
    report_lines.append("")
    
    # Ring size distribution
    report_lines.append("Ring Size Distribution:")
    rings = sp['ring_distribution']
    report_lines.append(f"  Molecules with rings: {rings['molecules_with_rings']} ({rings['pct_with_rings']:.1f}%)")
    report_lines.append(f"  Total rings: {rings['total_rings']}")
    report_lines.append(f"  Average rings per molecule: {rings['avg_rings_per_molecule']:.2f}")
    report_lines.append("  Ring size distribution:")
    for size in ['3', '4', '5', '6', '7', '8', 'larger']:
        count = rings['sizes'].get(size, 0)
        if count > 0:
            size_label = f"{size}-membered" if size != 'larger' else ">8-membered"
            report_lines.append(f"    - {size_label}: {count}")
    if sp['examples']['unusual_rings']:
        report_lines.append("  Examples of unusual ring sizes:")
        for ex in sp['examples']['unusual_rings'][:2]:
            report_lines.append(f"    - {ex['ring_size']}-membered ring: {ex['smiles']}")
    report_lines.append("")
    
    # Section 19: Recommendations
    report_lines.append("-" * 80)
    report_lines.append("19. RECOMMENDED CLEANING WORKFLOW")
    report_lines.append("-" * 80)
    report_lines.append("Actionable steps to improve dataset quality based on the issues detected above.")
    report_lines.append("")
    
    if not recommendations:
        report_lines.append("✓ Dataset quality is good. No critical issues detected.")
        report_lines.append("Optional improvements:")
        report_lines.append("  - Standardize SMILES using canonicalize_smiles")
        report_lines.append("  - Remove salts/solvents using desalt and remove_solvents")
    else:
        report_lines.append("Critical issues detected. Recommended workflow:")
        for rec in recommendations:
            report_lines.append(rec)
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Join all lines
    report_text = '\n'.join(report_lines)
    
    # ========================================================================
    # Save text report
    # ========================================================================
    
    txt_filename = _store_resource(
        report_text,
        project_manifest_path,
        f"{output_name}_txt",
        f"{explanation} (TXT)",
        'txt'
    )
    
    # Calculate duplicate percentage for key metrics
    dup_pct = (dup_result.get('n_duplicates', 0) / n_molecules * 100) if n_molecules > 0 else 0
    
    return {
        'report_text': report_text,
        'calculations_json': json_filename,
        'report_txt': txt_filename,
        'n_issues': len(critical_issues),
        'critical_issues': critical_issues,
        'n_molecules': n_molecules,
        'key_metrics': {
            'completeness': completeness,
            'validity': validity_result,
            'pct_pains': pains_result['pct_pains'],
            'pct_lipinski_compliant': lipinski_result['pct_compliant'],
            'pct_duplicates': float(dup_pct),
            'scaffold_diversity': scaffold_metrics
        }
    }

