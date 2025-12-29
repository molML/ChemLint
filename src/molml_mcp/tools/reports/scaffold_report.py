"""
Scaffold analysis report generation for molecular datasets.

Provides comprehensive analysis of molecular scaffolds including distribution,
diversity metrics, structural outliers, and bioactivity enrichment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from collections import Counter
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem

from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.tools.core_mol.scaffolds import _get_scaffold


def _calculate_gini_coefficient(counts: List[int]) -> float:
    """
    Calculate Gini coefficient for scaffold distribution inequality.
    
    Gini = 0: perfect equality (all scaffolds have same count)
    Gini = 1: perfect inequality (one scaffold has all molecules)
    
    Typical values:
    - 0.0-0.3: Low inequality (diverse scaffolds)
    - 0.3-0.6: Moderate inequality
    - 0.6-1.0: High inequality (few dominant scaffolds)
    """
    if len(counts) == 0:
        return 0.0
    
    counts = np.array(sorted(counts))
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n


def _calculate_shannon_entropy(counts: List[int]) -> float:
    """
    Calculate Shannon entropy for scaffold diversity.
    
    Higher entropy = more diverse scaffold distribution
    Lower entropy = dominated by few scaffolds
    
    Typical interpretation:
    - 0-2: Low diversity (few dominant scaffolds)
    - 2-4: Moderate diversity
    - 4+: High diversity (many different scaffolds)
    """
    counts = np.array(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    
    probabilities = counts / total
    # Filter out zeros to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def _calculate_scaffold_similarity(scaffold_smiles: str, all_scaffold_smiles: List[str], 
                                   radius: int = 2, n_bits: int = 2048) -> float:
    """
    Calculate average Tanimoto similarity of a scaffold to all other scaffolds.
    Returns average similarity (0-1 scale).
    """
    from rdkit import DataStructs
    
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return np.nan
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    similarities = []
    for other_smiles in all_scaffold_smiles:
        if other_smiles == scaffold_smiles:
            continue
        other_mol = Chem.MolFromSmiles(other_smiles)
        if other_mol is None:
            continue
        other_fp = AllChem.GetMorganFingerprintAsBitVect(other_mol, radius, nBits=n_bits)
        sim = DataStructs.TanimotoSimilarity(fp, other_fp)
        similarities.append(sim)
    
    if len(similarities) == 0:
        return np.nan
    
    return np.mean(similarities)


def _perform_enrichment_analysis(df: pd.DataFrame, scaffold_col: str, 
                                 activity_col: str, activity_type: str) -> Dict:
    """
    Perform statistical enrichment analysis for activity in scaffolds.
    
    For classification: Uses Fisher's exact test
    For regression: Uses t-test comparing scaffold's mean to overall mean
    """
    results = {
        'privileged_scaffolds': [],
        'inactive_scaffolds': [],
        'overall_stats': {}
    }
    
    # Remove NaN scaffolds
    df_valid = df[df[scaffold_col].notna()].copy()
    
    if activity_type == 'classification':
        # Overall activity rate
        overall_active = df_valid[activity_col].sum()
        overall_total = len(df_valid)
        overall_rate = overall_active / overall_total if overall_total > 0 else 0
        
        results['overall_stats'] = {
            'total': overall_total,
            'active': int(overall_active),
            'rate': overall_rate
        }
        
        # Analyze each scaffold (minimum 5 molecules for statistical power)
        scaffold_counts = df_valid[scaffold_col].value_counts()
        
        for scaffold, count in scaffold_counts.items():
            if count < 5:  # Skip rare scaffolds
                continue
            
            scaffold_df = df_valid[df_valid[scaffold_col] == scaffold]
            scaffold_active = scaffold_df[activity_col].sum()
            scaffold_rate = scaffold_active / count
            
            # Fisher's exact test
            # Contingency table: [[scaffold_active, scaffold_inactive], 
            #                     [other_active, other_inactive]]
            scaffold_inactive = count - scaffold_active
            other_active = overall_active - scaffold_active
            other_inactive = overall_total - count - other_active
            
            _, p_value = stats.fisher_exact(
                [[scaffold_active, scaffold_inactive], 
                 [other_active, other_inactive]]
            )
            
            # Classify as privileged (>70% active) or inactive (<30% active)
            if scaffold_rate > 0.7 and p_value < 0.05:
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                results['privileged_scaffolds'].append({
                    'scaffold': scaffold,
                    'rate': scaffold_rate,
                    'active': int(scaffold_active),
                    'total': count,
                    'p_value': p_value,
                    'significance': sig
                })
            elif scaffold_rate < 0.3 and p_value < 0.05:
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                results['inactive_scaffolds'].append({
                    'scaffold': scaffold,
                    'rate': scaffold_rate,
                    'active': int(scaffold_active),
                    'total': count,
                    'p_value': p_value,
                    'significance': sig
                })
        
        # Sort by significance
        results['privileged_scaffolds'].sort(key=lambda x: x['p_value'])
        results['inactive_scaffolds'].sort(key=lambda x: x['p_value'])
        
    else:  # regression
        # Overall mean activity
        overall_mean = df_valid[activity_col].mean()
        overall_std = df_valid[activity_col].std()
        
        results['overall_stats'] = {
            'mean': overall_mean,
            'std': overall_std,
            'median': df_valid[activity_col].median(),
            'total': len(df_valid)
        }
        
        # Analyze each scaffold
        scaffold_counts = df_valid[scaffold_col].value_counts()
        
        for scaffold, count in scaffold_counts.items():
            if count < 5:  # Skip rare scaffolds
                continue
            
            scaffold_df = df_valid[df_valid[scaffold_col] == scaffold]
            scaffold_mean = scaffold_df[activity_col].mean()
            scaffold_std = scaffold_df[activity_col].std()
            
            # T-test comparing scaffold mean to overall mean
            if count > 1 and scaffold_std > 0:
                t_stat, p_value = stats.ttest_1samp(
                    scaffold_df[activity_col].values, overall_mean
                )
                
                # Classify as high-activity (mean > overall + 0.5*std) or low-activity
                if scaffold_mean > overall_mean + 0.5 * overall_std and p_value < 0.05:
                    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                    results['privileged_scaffolds'].append({
                        'scaffold': scaffold,
                        'mean': scaffold_mean,
                        'std': scaffold_std,
                        'total': count,
                        'p_value': p_value,
                        'significance': sig
                    })
                elif scaffold_mean < overall_mean - 0.5 * overall_std and p_value < 0.05:
                    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                    results['inactive_scaffolds'].append({
                        'scaffold': scaffold,
                        'mean': scaffold_mean,
                        'std': scaffold_std,
                        'total': count,
                        'p_value': p_value,
                        'significance': sig
                    })
        
        # Sort by significance
        results['privileged_scaffolds'].sort(key=lambda x: x['p_value'])
        results['inactive_scaffolds'].sort(key=lambda x: x['p_value'])
    
    return results


def generate_scaffold_report(
    dataset_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    scaffold_type: str = 'bemis_murcko',
    activity_column: Optional[str] = None,
    activity_type: Optional[str] = None,
    outlier_threshold: float = 0.2,
    top_n: int = 10,
    explanation: str = "Generated scaffold analysis report"
) -> Dict:
    """
    Generate a comprehensive scaffold analysis report for a molecular dataset.
    
    This function creates a detailed report analyzing the distribution, diversity, 
    and characteristics of molecular scaffolds in a dataset. The report includes:
    - Overview statistics (total molecules, unique scaffolds, diversity metrics)
    - Distribution analysis (singleton, rare, common, abundant scaffolds)
    - Top most common scaffolds
    - Structural outliers (scaffolds dissimilar to others)
    - Optional: Activity enrichment analysis for privileged/inactive scaffolds
    
    The report is saved as both a formatted text file and a structured JSON file
    for programmatic access.
    
    Parameters
    ----------
    dataset_filename : str
        Input dataset filename from manifest (must contain SMILES column).
    project_manifest_path : str
        Path to the project manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    output_filename : str
        Name for the output report files (will be versioned with unique ID).
        Two files will be created: {output_filename}.txt and {output_filename}.json
    scaffold_type : str, default='bemis_murcko'
        Type of scaffold to extract. Options:
        - 'bemis_murcko': Bemis-Murcko scaffold (rings + linkers, retains some substituents)
        - 'generic': Generic skeleton (all atoms → carbon, all bonds → single)
        - 'cyclic_skeleton': Cyclic skeleton (rings + linkers only, no side chains)
    activity_column : str, optional
        Name of the column containing activity data for enrichment analysis.
        If None, activity enrichment section will be skipped.
    activity_type : str, optional
        Type of activity data: 'classification' (binary 0/1) or 'regression' (continuous).
        Required if activity_column is provided.
    outlier_threshold : float, default=0.2
        Tanimoto similarity threshold for identifying structural outliers.
        Scaffolds with average similarity < threshold are considered outliers.
    top_n : int, default=10
        Number of top scaffolds to show in the report.
    explanation : str
        Human-readable description of this operation.
        
    Returns
    -------
    dict
        Contains:
        - report_text_filename: Filename of the formatted text report (.txt)
        - report_json_filename: Filename of the structured JSON report (.json)
        - dataset_with_scaffolds_filename: Dataset with scaffold column added
        - n_molecules: Total number of molecules
        - n_unique_scaffolds: Number of unique scaffolds
        - n_no_scaffold: Number of molecules without scaffolds
        - diversity_ratio: Ratio of unique scaffolds to total molecules
        - summary: Brief summary of findings
        - report: Full text report content
        
    Example
    -------
    >>> result = generate_scaffold_report(
    ...     'my_dataset.csv',
    ...     '/path/to/manifest.json',
    ...     'SMILES',
    ...     'scaffold_report',
    ...     scaffold_type='bemis_murcko',
    ...     activity_column='active',
    ...     activity_type='classification'
    ... )
    >>> print(result['summary'])
    Analyzed 3,347 molecules with 487 unique scaffolds (14.5% diversity).
    """
    # Validate inputs
    if activity_column is not None and activity_type is None:
        raise ValueError("activity_type must be specified when activity_column is provided")
    if activity_type is not None and activity_type not in ['classification', 'regression']:
        raise ValueError("activity_type must be 'classification' or 'regression'")
    
    # Load dataset
    df = _load_resource(project_manifest_path, dataset_filename)
    n_molecules = len(df)
    
    # Validate columns
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found. Available: {df.columns.tolist()}")
    if activity_column is not None and activity_column not in df.columns:
        raise ValueError(f"Activity column '{activity_column}' not found. Available: {df.columns.tolist()}")
    
    print(f"Analyzing scaffolds for {n_molecules} molecules...")
    
    # Calculate scaffolds
    scaffolds = []
    comments = []
    for smi in df[smiles_column]:
        scaffold, comment = _get_scaffold(str(smi) if pd.notna(smi) else None, scaffold_type)
        scaffolds.append(scaffold)
        comments.append(comment)
    
    df['scaffold'] = scaffolds
    df['scaffold_comment'] = comments
    
    # Count scaffolds
    n_no_scaffold = df['scaffold'].isna().sum()
    df_with_scaffold = df[df['scaffold'].notna()]
    scaffold_counts = Counter(df_with_scaffold['scaffold'])
    n_unique_scaffolds = len(scaffold_counts)
    diversity_ratio = n_unique_scaffolds / n_molecules if n_molecules > 0 else 0
    
    # Distribution categories
    singleton = sum(1 for c in scaffold_counts.values() if c == 1)
    rare = sum(1 for c in scaffold_counts.values() if 2 <= c <= 5)
    common = sum(1 for c in scaffold_counts.values() if 6 <= c <= 20)
    abundant = sum(1 for c in scaffold_counts.values() if c > 20)
    
    singleton_mols = sum(c for c in scaffold_counts.values() if c == 1)
    rare_mols = sum(c for c in scaffold_counts.values() if 2 <= c <= 5)
    common_mols = sum(c for c in scaffold_counts.values() if 6 <= c <= 20)
    abundant_mols = sum(c for c in scaffold_counts.values() if c > 20)
    
    # Diversity metrics
    counts = list(scaffold_counts.values())
    gini = _calculate_gini_coefficient(counts)
    shannon = _calculate_shannon_entropy(counts)
    
    # Top scaffolds
    top_scaffolds = scaffold_counts.most_common(top_n)
    
    # Structural outliers (only for top 50 scaffolds to save time)
    print("Calculating structural outliers...")
    outliers = []
    top_50_scaffolds = [s for s, _ in scaffold_counts.most_common(50)]
    unique_scaffolds = list(scaffold_counts.keys())
    
    for scaffold in top_50_scaffolds:
        avg_sim = _calculate_scaffold_similarity(scaffold, unique_scaffolds)
        if avg_sim < outlier_threshold:
            count = scaffold_counts[scaffold]
            outliers.append({
                'scaffold': scaffold,
                'count': count,
                'avg_similarity': avg_sim
            })
    
    outliers.sort(key=lambda x: x['avg_similarity'])
    
    # Activity enrichment (optional)
    enrichment_results = None
    if activity_column is not None:
        print("Performing activity enrichment analysis...")
        enrichment_results = _perform_enrichment_analysis(
            df, 'scaffold', activity_column, activity_type
        )
    
    # Build report text
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SCAFFOLD ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Section 1: Overview
    report_lines.append("1. OVERVIEW")
    report_lines.append(f"   Total molecules: {n_molecules:,}")
    report_lines.append(f"   Unique scaffolds: {n_unique_scaffolds:,}")
    report_lines.append(f"   Diversity ratio: {diversity_ratio:.3f} ({diversity_ratio*100:.1f}% of molecules have unique scaffolds)")
    report_lines.append(f"   Molecules without scaffolds: {n_no_scaffold} ({n_no_scaffold/n_molecules*100:.1f}%)")
    report_lines.append("")
    
    # Section 2: Distribution
    report_lines.append("2. DISTRIBUTION")
    total_with_scaffold = n_molecules - n_no_scaffold
    report_lines.append(f"   Singleton scaffolds: {singleton:,} ({singleton/n_unique_scaffolds*100:.1f}% of scaffolds, {singleton_mols/total_with_scaffold*100:.1f}% of molecules)")
    report_lines.append(f"   Rare scaffolds (2-5): {rare:,} ({rare/n_unique_scaffolds*100:.1f}% of scaffolds, {rare_mols/total_with_scaffold*100:.1f}% of molecules)")
    report_lines.append(f"   Common scaffolds (6-20): {common:,} ({common/n_unique_scaffolds*100:.1f}% of scaffolds, {common_mols/total_with_scaffold*100:.1f}% of molecules)")
    report_lines.append(f"   Abundant scaffolds (>20): {abundant:,} ({abundant/n_unique_scaffolds*100:.1f}% of scaffolds, {abundant_mols/total_with_scaffold*100:.1f}% of molecules)")
    report_lines.append("")
    
    # Interpretation of Gini
    if gini < 0.3:
        gini_interp = "low inequality"
    elif gini < 0.6:
        gini_interp = "moderate inequality"
    else:
        gini_interp = "high inequality"
    
    # Interpretation of Shannon
    if shannon < 2:
        shannon_interp = "low diversity"
    elif shannon < 4:
        shannon_interp = "moderate diversity"
    else:
        shannon_interp = "high diversity"
    
    report_lines.append(f"   Gini coefficient: {gini:.2f} ({gini_interp})")
    report_lines.append(f"   Shannon entropy: {shannon:.2f} ({shannon_interp})")
    report_lines.append("")
    
    # Section 3: Top scaffolds
    report_lines.append(f"3. TOP {min(top_n, len(top_scaffolds))} MOST COMMON SCAFFOLDS")
    for i, (scaffold, count) in enumerate(top_scaffolds, 1):
        pct = count / n_molecules * 100
        report_lines.append(f"   {i}. {scaffold}: {count:,} molecules ({pct:.1f}%)")
    report_lines.append("")
    
    # Section 4: Structural outliers
    report_lines.append("4. STRUCTURAL OUTLIERS")
    if len(outliers) > 0:
        report_lines.append(f"   Found {len(outliers)} scaffolds with avg Tanimoto < {outlier_threshold}:")
        for outlier in outliers[:10]:  # Show top 10 outliers
            report_lines.append(f"   - {outlier['scaffold']}: {outlier['count']} molecules (Tanimoto={outlier['avg_similarity']:.2f})")
    else:
        report_lines.append(f"   No structural outliers found (all scaffolds have avg similarity >= {outlier_threshold})")
    report_lines.append("")
    
    # Section 5: Activity enrichment (optional)
    if enrichment_results is not None:
        if activity_type == 'classification':
            report_lines.append("5. ACTIVITY ENRICHMENT (CLASSIFICATION)")
            stats = enrichment_results['overall_stats']
            report_lines.append(f"   Overall activity rate: {stats['rate']*100:.1f}% ({stats['active']:,}/{stats['total']:,} active)")
            report_lines.append("")
            
            if len(enrichment_results['privileged_scaffolds']) > 0:
                report_lines.append("   Privileged scaffolds (>70% active):")
                for item in enrichment_results['privileged_scaffolds'][:10]:
                    report_lines.append(
                        f"   - {item['scaffold']}: {item['rate']*100:.0f}% active "
                        f"({item['active']}/{item['total']} molecules) {item['significance']} p={item['p_value']:.3e}"
                    )
                report_lines.append("")
            
            if len(enrichment_results['inactive_scaffolds']) > 0:
                report_lines.append("   Inactive scaffolds (<30% active):")
                for item in enrichment_results['inactive_scaffolds'][:10]:
                    report_lines.append(
                        f"   - {item['scaffold']}: {item['rate']*100:.0f}% active "
                        f"({item['active']}/{item['total']} molecules) {item['significance']} p={item['p_value']:.3e}"
                    )
        
        else:  # regression
            report_lines.append("5. ACTIVITY ENRICHMENT (REGRESSION)")
            stats = enrichment_results['overall_stats']
            report_lines.append(f"   Overall mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            report_lines.append(f"   Overall median: {stats['median']:.2f}")
            report_lines.append("")
            
            if len(enrichment_results['privileged_scaffolds']) > 0:
                report_lines.append("   High-activity scaffolds (mean > overall + 0.5*std):")
                for item in enrichment_results['privileged_scaffolds'][:10]:
                    report_lines.append(
                        f"   - {item['scaffold']}: mean={item['mean']:.2f} ± {item['std']:.2f} "
                        f"({item['total']} molecules) {item['significance']} p={item['p_value']:.3e}"
                    )
                report_lines.append("")
            
            if len(enrichment_results['inactive_scaffolds']) > 0:
                report_lines.append("   Low-activity scaffolds (mean < overall - 0.5*std):")
                for item in enrichment_results['inactive_scaffolds'][:10]:
                    report_lines.append(
                        f"   - {item['scaffold']}: mean={item['mean']:.2f} ± {item['std']:.2f} "
                        f"({item['total']} molecules) {item['significance']} p={item['p_value']:.3e}"
                    )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Build structured JSON report
    report_json = {
        'overview': {
            'total_molecules': n_molecules,
            'unique_scaffolds': n_unique_scaffolds,
            'diversity_ratio': diversity_ratio,
            'molecules_without_scaffolds': int(n_no_scaffold)
        },
        'distribution': {
            'singleton_scaffolds': {'count': singleton, 'molecules': singleton_mols},
            'rare_scaffolds': {'count': rare, 'molecules': rare_mols},
            'common_scaffolds': {'count': common, 'molecules': common_mols},
            'abundant_scaffolds': {'count': abundant, 'molecules': abundant_mols},
            'gini_coefficient': float(gini),
            'shannon_entropy': float(shannon)
        },
        'top_scaffolds': [
            {'scaffold': s, 'count': c, 'percentage': c/n_molecules*100} 
            for s, c in top_scaffolds
        ],
        'structural_outliers': outliers,
        'activity_enrichment': enrichment_results
    }
    
    # Store all outputs
    # 1. Text report
    report_text_id = _store_resource(
        report_text,
        project_manifest_path,
        output_filename + "_report",
        explanation + " (formatted text)",
        'txt'
    )
    
    # 2. JSON report
    report_json_id = _store_resource(
        report_json,
        project_manifest_path,
        output_filename + "_report",
        explanation + " (structured JSON)",
        'json'
    )
    
    # 3. Dataset with scaffolds
    dataset_with_scaffolds_id = _store_resource(
        df,
        project_manifest_path,
        output_filename + "_with_scaffolds",
        explanation + " (dataset with scaffold column)",
        'csv'
    )
    
    # Create summary
    summary = (
        f"Analyzed {n_molecules:,} molecules with {n_unique_scaffolds:,} unique scaffolds "
        f"({diversity_ratio*100:.1f}% diversity). "
    )
    if enrichment_results is not None:
        n_priv = len(enrichment_results['privileged_scaffolds'])
        n_inact = len(enrichment_results['inactive_scaffolds'])
        summary += f"Found {n_priv} privileged and {n_inact} inactive scaffolds."
    
    # Print report to console
    print("\n" + report_text)
    
    return {
        'report_text_filename': report_text_id,
        'report_json_filename': report_json_id,
        'dataset_with_scaffolds_filename': dataset_with_scaffolds_id,
        'n_molecules': n_molecules,
        'n_unique_scaffolds': n_unique_scaffolds,
        'n_no_scaffold': int(n_no_scaffold),
        'diversity_ratio': diversity_ratio,
        'gini_coefficient': float(gini),
        'shannon_entropy': float(shannon),
        'summary': summary,
        'report': report_text
    }


def get_all_scaffold_report_tools():
    """
    Returns a list of MCP-exposed scaffold report functions for server registration.
    """
    return [
        generate_scaffold_report,
    ]
