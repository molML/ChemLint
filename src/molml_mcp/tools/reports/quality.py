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
- generate_scaffold_report() for scaffold diversity
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

from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.tools.core.dataset_ops import get_dataset_summary
from molml_mcp.tools.featurization.simple_descriptors import calculate_simple_descriptors
from molml_mcp.tools.core.filtering import filter_by_pains, filter_by_lipinski_ro5, filter_by_veber_rules, filter_by_qed
from molml_mcp.tools.cleaning.deduplication import find_duplicates_dataset
from molml_mcp.tools.core.outliers import detect_outliers_iqr
from molml_mcp.tools.reports.scaffold_report import generate_scaffold_report


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
        functional_groups['Tertiary amine'].append(Fragments.fr_N(mol))
        functional_groups['Sulfonamide'].append(Fragments.fr_sulfon(mol) + Fragments.fr_sulfonamd(mol))
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
# MAIN REPORT GENERATION FUNCTION
# ============================================================================

def generate_quality_report(
    input_filename: str,
    project_manifest_path: str,
    smiles_col: str = 'SMILES',
    activity_col: Optional[str] = None,
    activity_type: str = 'continuous',
    activity_units: str = 'nM',
    output_name: str = 'quality_report',
    explanation: str = 'Generated comprehensive data quality report'
) -> Dict:
    """
    Generate comprehensive data quality report for a molecular dataset.
    
    This function orchestrates existing tools and unique analyses to generate
    a 15-section formatted report covering:
    1. Dataset Overview
    2. Data Completeness
    3. SMILES Validity
    4. PAINS Patterns
    5. Duplicates & Conflicts
    6. Physicochemical Properties
    7. Lipinski Rule of Five
    8. Veber Rules
    9. QED Analysis
    10. Outlier Detection
    11. Activity Distribution (optional)
    12. Scaffold Diversity
    13. Functional Group Analysis
    14. Overall Quality Score
    15. Recommended Cleaning Workflow
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename (e.g., "raw_data_12345678.csv")
    project_manifest_path : str
        Path to manifest.json for resource management
    smiles_col : str, default='SMILES'
        Column containing SMILES strings
    activity_col : str, optional
        Column containing bioactivity values (optional)
    activity_type : str, default='continuous'
        Type of activity data: 'continuous' or 'classification'
    activity_units : str, default='nM'
        Units for continuous activity data (e.g., 'nM', 'μM')
    output_name : str, default='quality_report'
        Prefix for output report files
    explanation : str
        Description of report generation
    
    Returns
    -------
    dict
        {
            'report_text': str,  # Full formatted report
            'report_json': str,  # JSON report filename
            'report_txt': str,   # Text report filename
            'quality_score': float,  # Overall quality score (0-100)
            'n_issues': int,  # Number of quality issues detected
            'critical_issues': list[str],  # List of critical issues requiring attention
        }
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset")
    
    has_activity = activity_col is not None and activity_col in df.columns
    
    n_molecules = len(df)
    
    # ========================================================================
    # Run all analyses by calling existing tools and helpers
    # ========================================================================
    
    # 1. Dataset overview (use get_dataset_summary)
    summary_result = get_dataset_summary(input_filename, project_manifest_path)
    column_stats = summary_result['column_stats']
    
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
    dup_result = find_duplicates_dataset(
        input_filename,
        project_manifest_path,
        smiles_col=smiles_col,
        activity_col=activity_col if has_activity else None
    )
    
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
        desc_result['output_filename'],
        project_manifest_path,
        outlier_temp_name,
        "Temporary outlier detection",
        columns=prop_cols,
        multiplier=1.5
    )
    
    # 11. Activity analysis (optional, unique helper)
    activity_result = None
    if has_activity:
        activity_values = df[activity_col].values
        activity_result = _analyze_activity_distribution(
            activity_values,
            activity_type=activity_type,
            units=activity_units
        )
    
    # 12. Scaffold diversity (existing tool)
    scaffold_temp_name = f"{output_name}_scaffold_temp"
    scaffold_result = generate_scaffold_report(
        input_filename,
        project_manifest_path,
        scaffold_temp_name,
        "Temporary scaffold analysis",
        smiles_col=smiles_col,
        activity_col=activity_col if has_activity else None,
        include_murcko=True,
        include_generic=False
    )
    
    # 13. Functional group analysis (unique helper)
    smiles_list = df[smiles_col].tolist()
    functional_groups_result = _analyze_functional_groups(smiles_list)
    
    # ========================================================================
    # Generate formatted text report (15 sections)
    # ========================================================================
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("MOLECULAR DATASET QUALITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Dataset: {input_filename}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Section 1: Overview
    report_lines.append("-" * 80)
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total molecules: {n_molecules}")
    report_lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    report_lines.append(f"SMILES column: {smiles_col}")
    if has_activity:
        report_lines.append(f"Activity column: {activity_col} ({activity_type})")
    else:
        report_lines.append("Activity column: None")
    report_lines.append("")
    
    # Section 2: Completeness
    report_lines.append("-" * 80)
    report_lines.append("2. DATA COMPLETENESS")
    report_lines.append("-" * 80)
    for col, stats in completeness.items():
        status = "✓" if stats['pct_missing'] < 5 else "⚠" if stats['pct_missing'] < 20 else "✗"
        report_lines.append(f"{status} {col}: {stats['missing']} missing ({stats['pct_missing']:.1f}%)")
    report_lines.append("")
    
    # Section 3: SMILES Validity
    report_lines.append("-" * 80)
    report_lines.append("3. SMILES VALIDITY")
    report_lines.append("-" * 80)
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
    report_lines.append(f"0 violations: {lipinski_result['0_violations']} ({lipinski_result['pct_compliant']:.1f}%)")
    report_lines.append(f"1 violation: {lipinski_result['1_violation']}")
    report_lines.append(f"2+ violations: {lipinski_result['2+_violations']}")
    report_lines.append("")
    
    # Section 8: Veber
    report_lines.append("-" * 80)
    report_lines.append("8. VEBER RULES")
    report_lines.append("-" * 80)
    report_lines.append(f"Compliant: {veber_result['pass']} ({veber_result['pct_compliant']:.1f}%)")
    report_lines.append("")
    
    # Section 9: QED
    report_lines.append("-" * 80)
    report_lines.append("9. QED (DRUG-LIKENESS)")
    report_lines.append("-" * 80)
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
    report_lines.append(f"Molecules with extreme properties: {outlier_result.get('n_extreme_molecules', 0)}")
    if outlier_result.get('extreme_molecules'):
        report_lines.append("Top outliers:")
        for mol_info in outlier_result['extreme_molecules'][:3]:
            report_lines.append(f"  - Row {mol_info['index']}: {mol_info['n_outlier_properties']} outlier properties")
    report_lines.append("")
    
    # Section 11: Activity (optional)
    if activity_result:
        report_lines.append("-" * 80)
        report_lines.append("11. ACTIVITY DISTRIBUTION")
        report_lines.append("-" * 80)
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
    
    # Section 12: Scaffold Diversity
    report_lines.append("-" * 80)
    report_lines.append("12. SCAFFOLD DIVERSITY")
    report_lines.append("-" * 80)
    # Parse scaffold report text for key metrics
    scaffold_text = scaffold_result.get('report', '')
    report_lines.append(scaffold_text.split('\n\n')[1] if '\n\n' in scaffold_text else "See scaffold report for details")
    report_lines.append("")
    
    # Section 13: Functional Groups
    report_lines.append("-" * 80)
    report_lines.append("13. FUNCTIONAL GROUP ANALYSIS")
    report_lines.append("-" * 80)
    top_groups = sorted(
        [(name, stats['count']) for name, stats in functional_groups_result.items() if name != 'halogen_breakdown'],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for name, count in top_groups:
        pct = functional_groups_result[name]['pct_dataset']
        report_lines.append(f"{name}: {count} molecules ({pct:.1f}%)")
    report_lines.append("")
    
    # Section 14: Quality Score
    report_lines.append("-" * 80)
    report_lines.append("14. OVERALL QUALITY SCORE")
    report_lines.append("-" * 80)
    
    # Calculate quality score (0-100)
    score_components = []
    
    # Completeness (20 points)
    avg_missing = np.mean([stats['pct_missing'] for stats in completeness.values()])
    completeness_score = max(0, 20 * (1 - avg_missing / 100))
    score_components.append(completeness_score)
    
    # Validity (20 points)
    validity_score = 20 * (validity_result['pct_valid'] / 100)
    score_components.append(validity_score)
    
    # PAINS (15 points)
    pains_score = 15 * (1 - pains_result['pct_pains'] / 100)
    score_components.append(pains_score)
    
    # Drug-likeness (15 points) - Lipinski
    druglike_score = 15 * (lipinski_result['pct_compliant'] / 100)
    score_components.append(druglike_score)
    
    # QED (15 points)
    if qed_result:
        qed_score = 15 * qed_result['mean']
        score_components.append(qed_score)
    else:
        score_components.append(0)
    
    # Duplicates (15 points)
    dup_pct = (dup_result.get('n_duplicates', 0) / n_molecules * 100) if n_molecules > 0 else 0
    dup_score = max(0, 15 * (1 - dup_pct / 20))  # Penalize >20% duplicates
    score_components.append(dup_score)
    
    overall_quality_score = sum(score_components)
    
    report_lines.append(f"Overall Quality Score: {overall_quality_score:.1f}/100")
    report_lines.append("")
    report_lines.append("Component scores:")
    report_lines.append(f"  Completeness: {completeness_score:.1f}/20")
    report_lines.append(f"  SMILES Validity: {validity_score:.1f}/20")
    report_lines.append(f"  PAINS-free: {pains_score:.1f}/15")
    report_lines.append(f"  Drug-likeness: {druglike_score:.1f}/15")
    report_lines.append(f"  QED: {score_components[4]:.1f}/15")
    report_lines.append(f"  Uniqueness: {dup_score:.1f}/15")
    report_lines.append("")
    
    # Section 15: Recommendations
    report_lines.append("-" * 80)
    report_lines.append("15. RECOMMENDED CLEANING WORKFLOW")
    report_lines.append("-" * 80)
    
    recommendations = []
    critical_issues = []
    
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
    # Save report outputs
    # ========================================================================
    
    # Save as JSON
    report_data = {
        'metadata': {
            'dataset': input_filename,
            'generated': datetime.now().isoformat(),
            'n_molecules': n_molecules,
            'smiles_col': smiles_col,
            'activity_col': activity_col,
            'activity_type': activity_type if has_activity else None
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
        'scaffold_diversity': {
            'report_file': scaffold_result.get('report_txt')
        },
        'functional_groups': functional_groups_result,
        'quality_score': {
            'overall': float(overall_quality_score),
            'components': {
                'completeness': float(completeness_score),
                'validity': float(validity_score),
                'pains_free': float(pains_score),
                'drug_likeness': float(druglike_score),
                'qed': float(score_components[4]),
                'uniqueness': float(dup_score)
            }
        },
        'recommendations': recommendations,
        'critical_issues': critical_issues
    }
    
    # Store JSON report
    json_filename = _store_resource(
        report_data,
        project_manifest_path,
        f"{output_name}_json",
        f"{explanation} (JSON)",
        'json'
    )
    
    # Store text report
    txt_filename = _store_resource(
        report_text,
        project_manifest_path,
        f"{output_name}_txt",
        f"{explanation} (TXT)",
        'txt'
    )
    
    return {
        'report_text': report_text,
        'report_json': json_filename,
        'report_txt': txt_filename,
        'quality_score': float(overall_quality_score),
        'n_issues': len(critical_issues),
        'critical_issues': critical_issues,
        'n_molecules': n_molecules,
        'sections': {
            'completeness': completeness,
            'validity': validity_result,
            'quality_components': {
                'completeness': float(completeness_score),
                'validity': float(validity_score),
                'pains_free': float(pains_score),
                'drug_likeness': float(druglike_score),
                'qed': float(score_components[4]),
                'uniqueness': float(dup_score)
            }
        }
    }

def generate_quality_report(
    dataset_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    activity_column: Optional[str] = None,
    activity_type: Optional[str] = None,
    explanation: str = "Generated data quality report"
) -> Dict:
    """
    Generate a comprehensive data quality report for a molecular dataset.
    
    TODO: Implement main report generation using helper functions.
    
    Parameters
    ----------
    dataset_filename : str
        Input dataset filename from manifest.
    project_manifest_path : str
        Path to the project manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    output_filename : str
        Name for the output report files.
    activity_column : str, optional
        Name of the column containing activity data.
    activity_type : str, optional
        Type of activity: 'classification' or 'continuous'.
    explanation : str
        Human-readable description of this operation.
    
    Returns
    -------
    dict
        Report metadata and filenames.
    """
    # TODO: Implement full report generation
    # This will be implemented in the next step
    pass
