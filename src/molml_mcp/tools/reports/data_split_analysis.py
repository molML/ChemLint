"""
Data splitting quality report generation for molecular datasets.

Looks for the following issues:
- Imbalanced splits (size-wise)
- Data leakage between splits
  - Identical molecules in different splits
  - Highly similar molecules in different splits
  - Functional group overlap between splits (approached leniently, as many datasets
    naturally have common functional groups across splits). Functional groups that
    only appear in one split could be highlighted.
  - Scaffold leakage/overlap between splits
  - Physicochemical property distribution overlap between splits
  - Activity distribution overlap between splits
  - Activity cliffs between splits (similar molecules across splits with large
    activity differences)
- Low data warning in any split
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from collections import Counter
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

from molml_mcp.infrastructure.resources import _load_resource


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _analyze_split_characteristics(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    label_col: str,
    min_split_size: int = 50,
    imbalance_threshold: float = 0.1
) -> Dict:
    """
    Compute fundamental split characteristics.
    
    Compares train/test splits, and train/val & test/val if validation split exists.
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    label_col : str
        Name of label column
    min_split_size : int
        Minimum acceptable molecules per split (default: 50)
    imbalance_threshold : float
        Threshold for flagging imbalanced splits (default: 0.1 = 10%)
        
    Returns
    -------
    Dict with split characteristics including:
        - sizes: absolute and relative sizes
        - ratios: size ratios between splits
        - flags: warnings for empty/small/imbalanced splits
        - class_distribution: for classification tasks
        - value_distribution: for regression tasks
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Compute sizes
    n_train = len(df_train)
    n_test = len(df_test)
    n_val = len(df_val) if df_val is not None else 0
    n_total = n_train + n_test + n_val
    
    # Compute relative sizes
    pct_train = (n_train / n_total * 100) if n_total > 0 else 0
    pct_test = (n_test / n_total * 100) if n_total > 0 else 0
    pct_val = (n_val / n_total * 100) if n_total > 0 and n_val > 0 else 0
    
    # Check for empty splits
    empty_splits = []
    if n_train == 0:
        empty_splits.append('train')
    if n_test == 0:
        empty_splits.append('test')
    if df_val is not None and n_val == 0:
        empty_splits.append('val')
    
    # Check for small splits
    small_splits = []
    if 0 < n_train < min_split_size:
        small_splits.append(f'train ({n_train} < {min_split_size})')
    if 0 < n_test < min_split_size:
        small_splits.append(f'test ({n_test} < {min_split_size})')
    if df_val is not None and 0 < n_val < min_split_size:
        small_splits.append(f'val ({n_val} < {min_split_size})')
    
    # Calculate size ratios
    ratios = {}
    if n_train > 0 and n_test > 0:
        ratios['train_test'] = round(n_train / n_test, 3)
        ratios['test_train'] = round(n_test / n_train, 3)
    
    if df_val is not None and n_val > 0:
        if n_train > 0:
            ratios['train_val'] = round(n_train / n_val, 3)
            ratios['val_train'] = round(n_val / n_train, 3)
        if n_test > 0:
            ratios['test_val'] = round(n_test / n_val, 3)
            ratios['val_test'] = round(n_val / n_test, 3)
    
    # Flag imbalanced splits
    imbalanced_flags = []
    
    # Check train/test balance (typical is 80/20, so ratio ~4.0)
    if 'train_test' in ratios:
        # Flag if train is too small compared to test (ratio < 1.5) or too large (ratio > 10)
        if ratios['train_test'] < 1.5:
            imbalanced_flags.append(f"Train too small relative to test (ratio={ratios['train_test']})")
        elif ratios['train_test'] > 10:
            imbalanced_flags.append(f"Train too large relative to test (ratio={ratios['train_test']})")
    
    # Check if any split represents less than threshold of total
    if pct_train < imbalance_threshold * 100 and n_train > 0:
        imbalanced_flags.append(f"Train is only {pct_train:.1f}% of total data")
    if pct_test < imbalance_threshold * 100 and n_test > 0:
        imbalanced_flags.append(f"Test is only {pct_test:.1f}% of total data")
    if df_val is not None and pct_val < imbalance_threshold * 100 and n_val > 0:
        imbalanced_flags.append(f"Val is only {pct_val:.1f}% of total data")
    
    # Determine if classification or regression
    # Check if labels are binary (0/1) or categorical
    all_labels = []
    if label_col in df_train.columns:
        all_labels.extend(df_train[label_col].dropna().tolist())
    if label_col in df_test.columns:
        all_labels.extend(df_test[label_col].dropna().tolist())
    if df_val is not None and label_col in df_val.columns:
        all_labels.extend(df_val[label_col].dropna().tolist())
    
    unique_labels = set(all_labels)
    # Classification: exactly 2 unique values (binary classification only)
    is_classification = (
        len(unique_labels) == 2 and 
        all(isinstance(x, (int, np.integer)) or float(x).is_integer() for x in unique_labels)
    )
    
    result = {
        'sizes': {
            'train': n_train,
            'test': n_test,
            'val': n_val if df_val is not None else None,
            'total': n_total
        },
        'percentages': {
            'train': round(pct_train, 2),
            'test': round(pct_test, 2),
            'val': round(pct_val, 2) if df_val is not None else None
        },
        'ratios': ratios,
        'flags': {
            'empty_splits': empty_splits,
            'small_splits': small_splits,
            'imbalanced_splits': imbalanced_flags
        },
        'task_type': 'classification' if is_classification else 'regression'
    }
    
    # Add class distribution for classification
    if is_classification:
        class_dist = {}
        
        for split_name, df in [('train', df_train), ('test', df_test), ('val', df_val)]:
            if df is None:
                continue
            
            if label_col not in df.columns:
                class_dist[split_name] = {'error': f'Column {label_col} not found'}
                continue
            
            labels = df[label_col].dropna()
            counts = Counter(labels)
            n_samples = len(labels)
            
            class_dist[split_name] = {
                'counts': dict(counts),
                'proportions': {
                    cls: round(count / n_samples, 4) 
                    for cls, count in counts.items()
                } if n_samples > 0 else {},
                'n_classes': len(counts),
                'n_samples': n_samples
            }
        
        result['class_distribution'] = class_dist
    
    # Add value distribution for regression
    else:
        value_dist = {}
        
        for split_name, df in [('train', df_train), ('test', df_test), ('val', df_val)]:
            if df is None:
                continue
            
            if label_col not in df.columns:
                value_dist[split_name] = {'error': f'Column {label_col} not found'}
                continue
            
            values = df[label_col].dropna().values
            
            if len(values) == 0:
                value_dist[split_name] = {'error': 'No valid values'}
                continue
            
            value_dist[split_name] = {
                'n_samples': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        result['value_distribution'] = value_dist
    
    return result


def _detect_exact_duplicates(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    max_examples: int = 10
) -> Dict:
    """
    Find identical molecules across splits (exact SMILES matches).
    
    This is a CRITICAL issue if found - indicates data leakage that will
    inflate performance metrics.
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    max_examples : int
        Maximum number of duplicate examples to include (default: 10)
        
    Returns
    -------
    Dict with:
        - train_test_duplicates: duplicates between train and test
        - train_val_duplicates: duplicates between train and val (if val exists)
        - test_val_duplicates: duplicates between test and val (if val exists)
        - severity: 'CRITICAL' if any duplicates found, 'OK' otherwise
        - total_duplicate_molecules: total unique molecules found in multiple splits
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    result = {
        'severity': 'OK',
        'total_duplicate_molecules': 0,
        'train_test_duplicates': None,
        'train_val_duplicates': None,
        'test_val_duplicates': None
    }
    
    # Helper to find duplicates between two splits
    def find_split_duplicates(df1, df2, name1, name2):
        if smiles_col not in df1.columns or smiles_col not in df2.columns:
            return {
                'error': f'SMILES column "{smiles_col}" not found in one or both splits',
                'n_duplicates': 0,
                'examples': []
            }
        
        # Get SMILES from each split (drop NaN)
        smiles1 = df1[smiles_col].dropna()
        smiles2 = df2[smiles_col].dropna()
        
        # Find intersection
        duplicates = set(smiles1) & set(smiles2)
        n_duplicates = len(duplicates)
        
        if n_duplicates == 0:
            return {
                'n_duplicates': 0,
                'examples': []
            }
        
        # Collect examples with indices
        examples = []
        for smi in list(duplicates)[:max_examples]:
            # Get indices in both splits
            indices1 = df1[df1[smiles_col] == smi].index.tolist()
            indices2 = df2[df2[smiles_col] == smi].index.tolist()
            
            examples.append({
                'smiles': smi[:100],  # Truncate long SMILES
                f'{name1}_indices': indices1[:5],  # Limit to 5 indices per split
                f'{name2}_indices': indices2[:5],
                f'{name1}_count': len(indices1),
                f'{name2}_count': len(indices2)
            })
        
        return {
            'n_duplicates': n_duplicates,
            'pct_of_split1': round(n_duplicates / len(smiles1) * 100, 2) if len(smiles1) > 0 else 0,
            'pct_of_split2': round(n_duplicates / len(smiles2) * 100, 2) if len(smiles2) > 0 else 0,
            'examples': examples,
            'showing_n_examples': min(n_duplicates, max_examples)
        }
    
    # Check train/test duplicates
    train_test_dups = find_split_duplicates(df_train, df_test, 'train', 'test')
    result['train_test_duplicates'] = train_test_dups
    
    if train_test_dups['n_duplicates'] > 0:
        result['severity'] = 'CRITICAL'
        result['total_duplicate_molecules'] += train_test_dups['n_duplicates']
    
    # Check train/val duplicates
    if df_val is not None:
        train_val_dups = find_split_duplicates(df_train, df_val, 'train', 'val')
        result['train_val_duplicates'] = train_val_dups
        
        if train_val_dups['n_duplicates'] > 0:
            result['severity'] = 'CRITICAL'
            result['total_duplicate_molecules'] += train_val_dups['n_duplicates']
    
    # Check test/val duplicates
    if df_val is not None:
        test_val_dups = find_split_duplicates(df_test, df_val, 'test', 'val')
        result['test_val_duplicates'] = test_val_dups
        
        if test_val_dups['n_duplicates'] > 0:
            result['severity'] = 'CRITICAL'
            result['total_duplicate_molecules'] += test_val_dups['n_duplicates']
    
    return result


def _detect_similarity_leakage(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    label_col: str,
    similarity_threshold: float = 0.9,
    activity_cliff_similarity: float = 0.8,
    activity_cliff_fold_diff: float = 10.0,
    max_examples: int = 10
) -> Dict:
    """
    Find highly similar molecules across splits and detect activity cliffs.
    
    Uses ECFP4 (Morgan fingerprints, radius=2) with Tanimoto similarity.
    
    High similarity (>0.9 Tanimoto) between splits indicates potential data leakage.
    Activity cliffs (>0.8 similarity + large activity difference) indicate problematic
    cases where similar molecules have very different labels.
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    label_col : str
        Name of label column
    similarity_threshold : float
        Threshold for flagging high similarity leakage (default: 0.9)
    activity_cliff_similarity : float
        Similarity threshold for activity cliff detection (default: 0.8)
    activity_cliff_fold_diff : float
        Fold difference for regression activity cliffs (default: 10.0)
        For classification, looks for different class labels
    max_examples : int
        Maximum number of examples to include (default: 10)
        
    Returns
    -------
    Dict with:
        - within_split_similarity: avg and max similarity within each split
        - between_split_similarity: avg and max similarity between splits
        - leakage statistics per split comparison
        - activity cliff counts and examples
        - similarity distribution statistics
        - severity flags
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Helper to compute fingerprints
    def compute_fingerprints(df, smiles_col):
        """Compute ECFP4 fingerprints for all valid SMILES."""
        fps = []
        indices = []
        smiles_list = []
        
        for idx, smi in enumerate(df[smiles_col]):
            if pd.isna(smi) or smi == '':
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)
            indices.append(idx)
            smiles_list.append(smi)
        
        return fps, indices, smiles_list
    
    # Helper to determine if classification or regression
    def is_classification_task(df, label_col):
        if label_col not in df.columns:
            return False
        labels = df[label_col].dropna()
        unique = set(labels)
        return len(unique) <= 20 and all(
            isinstance(x, (int, np.integer)) or x in [0, 1, 0.0, 1.0] 
            for x in unique
        )
    
    # Helper to check for activity cliff
    def is_activity_cliff(label1, label2, is_classification):
        """Check if two labels represent an activity cliff."""
        if pd.isna(label1) or pd.isna(label2):
            return False
        
        if is_classification:
            # For classification: different classes
            return label1 != label2
        else:
            # For regression: fold difference >= threshold
            if label1 <= 0 or label2 <= 0:
                return False
            ratio = max(label1, label2) / min(label1, label2)
            return ratio >= activity_cliff_fold_diff
    
    # Helper to analyze similarity between two splits
    def analyze_split_similarity(df1, df2, name1, name2, fps1, indices1, smiles1, fps2, indices2, smiles2):
        """Compute similarity metrics between two splits."""
        if len(fps1) == 0 or len(fps2) == 0:
            return {
                'error': 'No valid fingerprints computed for one or both splits',
                'n_high_similarity': 0,
                'n_activity_cliffs': 0,
                'similarity_stats': {}
            }
        
        # Get labels for activity cliff detection
        is_clf = is_classification_task(df1, label_col) or is_classification_task(df2, label_col)
        
        labels1 = []
        labels2 = []
        if label_col in df1.columns and label_col in df2.columns:
            labels1 = df1.iloc[indices1][label_col].values
            labels2 = df2.iloc[indices2][label_col].values
        
        # For each molecule in split2, find most similar in split1
        max_similarities = []
        high_similarity_pairs = []
        activity_cliff_pairs = []
        
        for i2, fp2 in enumerate(fps2):
            # Compute similarities to all molecules in split1
            similarities = [DataStructs.TanimotoSimilarity(fp2, fp1) for fp1 in fps1]
            max_sim = max(similarities)
            max_sim_idx = similarities.index(max_sim)
            
            max_similarities.append(max_sim)
            
            # Check for high similarity leakage
            if max_sim >= similarity_threshold:
                pair = {
                    'similarity': round(max_sim, 4),
                    f'{name2}_index': indices2[i2],
                    f'{name2}_smiles': smiles2[i2][:100],
                    f'{name1}_index': indices1[max_sim_idx],
                    f'{name1}_smiles': smiles1[max_sim_idx][:100]
                }
                
                # Add labels if available
                if len(labels1) > 0 and len(labels2) > 0:
                    pair[f'{name2}_label'] = float(labels2[i2]) if not pd.isna(labels2[i2]) else None
                    pair[f'{name1}_label'] = float(labels1[max_sim_idx]) if not pd.isna(labels1[max_sim_idx]) else None
                
                high_similarity_pairs.append(pair)
            
            # Check for activity cliffs
            if max_sim >= activity_cliff_similarity and len(labels1) > 0 and len(labels2) > 0:
                if is_activity_cliff(labels2[i2], labels1[max_sim_idx], is_clf):
                    cliff_pair = {
                        'similarity': round(max_sim, 4),
                        f'{name2}_index': indices2[i2],
                        f'{name2}_smiles': smiles2[i2][:100],
                        f'{name2}_label': float(labels2[i2]) if not pd.isna(labels2[i2]) else None,
                        f'{name1}_index': indices1[max_sim_idx],
                        f'{name1}_smiles': smiles1[max_sim_idx][:100],
                        f'{name1}_label': float(labels1[max_sim_idx]) if not pd.isna(labels1[max_sim_idx]) else None
                    }
                    
                    if not is_clf and cliff_pair[f'{name2}_label'] and cliff_pair[f'{name1}_label']:
                        fold_diff = max(cliff_pair[f'{name2}_label'], cliff_pair[f'{name1}_label']) / \
                                    min(cliff_pair[f'{name2}_label'], cliff_pair[f'{name1}_label'])
                        cliff_pair['fold_difference'] = round(fold_diff, 2)
                    
                    activity_cliff_pairs.append(cliff_pair)
        
        # Compute similarity statistics
        similarity_stats = {
            'mean': round(float(np.mean(max_similarities)), 4),
            'median': round(float(np.median(max_similarities)), 4),
            'std': round(float(np.std(max_similarities)), 4),
            'min': round(float(np.min(max_similarities)), 4),
            'max': round(float(np.max(max_similarities)), 4),
            'q25': round(float(np.percentile(max_similarities, 25)), 4),
            'q75': round(float(np.percentile(max_similarities, 75)), 4)
        }
        
        return {
            'n_high_similarity': len(high_similarity_pairs),
            'pct_high_similarity': round(len(high_similarity_pairs) / len(fps2) * 100, 2) if len(fps2) > 0 else 0,
            'n_activity_cliffs': len(activity_cliff_pairs),
            'pct_activity_cliffs': round(len(activity_cliff_pairs) / len(fps2) * 100, 2) if len(fps2) > 0 else 0,
            'similarity_stats': similarity_stats,
            'high_similarity_examples': high_similarity_pairs[:max_examples],
            'activity_cliff_examples': activity_cliff_pairs[:max_examples],
            'showing_similarity_examples': min(len(high_similarity_pairs), max_examples),
            'showing_cliff_examples': min(len(activity_cliff_pairs), max_examples)
        }
    
    # Helper to compute within-split similarity statistics
    def compute_within_split_similarity(fps, name):
        """Compute average and max similarity within a single split."""
        if len(fps) < 2:
            return {
                'error': 'Insufficient molecules for within-split similarity',
                'n_molecules': len(fps)
            }
        
        # Sample pairs to avoid O(n²) computation for large datasets
        # For datasets > 1000, sample 1000 random molecules
        if len(fps) > 1000:
            import random
            sampled_fps = random.sample(fps, 1000)
        else:
            sampled_fps = fps
        
        similarities = []
        for i in range(len(sampled_fps)):
            for j in range(i + 1, len(sampled_fps)):
                sim = DataStructs.TanimotoSimilarity(sampled_fps[i], sampled_fps[j])
                similarities.append(sim)
        
        if len(similarities) == 0:
            return {
                'error': 'Could not compute similarities',
                'n_molecules': len(fps)
            }
        
        return {
            'n_molecules': len(fps),
            'n_comparisons': len(similarities),
            'avg_similarity': round(float(np.mean(similarities)), 4),
            'max_similarity': round(float(np.max(similarities)), 4),
            'median_similarity': round(float(np.median(similarities)), 4),
            'sampled': len(fps) > 1000
        }
    
    # Compute fingerprints for all splits
    train_fps, train_indices, train_smiles = compute_fingerprints(df_train, smiles_col)
    test_fps, test_indices, test_smiles = compute_fingerprints(df_test, smiles_col)
    val_fps, val_indices, val_smiles = None, None, None
    if df_val is not None:
        val_fps, val_indices, val_smiles = compute_fingerprints(df_val, smiles_col)
    
    result = {
        'similarity_threshold': similarity_threshold,
        'activity_cliff_similarity_threshold': activity_cliff_similarity,
        'activity_cliff_fold_threshold': activity_cliff_fold_diff,
        'within_split_similarity': {},
        'between_split_similarity': {},
        'test_vs_train': None,
        'val_vs_train': None,
        'val_vs_test': None,
        'overall_severity': 'OK'
    }
    
    # Compute within-split similarities
    result['within_split_similarity']['train'] = compute_within_split_similarity(train_fps, 'train')
    result['within_split_similarity']['test'] = compute_within_split_similarity(test_fps, 'test')
    if df_val is not None and val_fps is not None:
        result['within_split_similarity']['val'] = compute_within_split_similarity(val_fps, 'val')
    
    # Analyze test vs train
    test_train_analysis = analyze_split_similarity(
        df_train, df_test, 'train', 'test',
        train_fps, train_indices, train_smiles,
        test_fps, test_indices, test_smiles
    )
    result['test_vs_train'] = test_train_analysis
    
    # Store between-split similarity summary
    result['between_split_similarity']['test_vs_train'] = {
        'avg_max_similarity': test_train_analysis['similarity_stats']['mean'],
        'max_similarity': test_train_analysis['similarity_stats']['max']
    }
    
    # Determine severity
    if test_train_analysis['n_high_similarity'] > 0:
        result['overall_severity'] = 'HIGH'
    if test_train_analysis['n_activity_cliffs'] > 0:
        if result['overall_severity'] == 'OK':
            result['overall_severity'] = 'MEDIUM'
    
    # Analyze val vs train (if val exists)
    if df_val is not None and val_fps is not None:
        val_train_analysis = analyze_split_similarity(
            df_train, df_val, 'train', 'val',
            train_fps, train_indices, train_smiles,
            val_fps, val_indices, val_smiles
        )
        result['val_vs_train'] = val_train_analysis
        
        result['between_split_similarity']['val_vs_train'] = {
            'avg_max_similarity': val_train_analysis['similarity_stats']['mean'],
            'max_similarity': val_train_analysis['similarity_stats']['max']
        }
        
        if val_train_analysis['n_high_similarity'] > 0:
            result['overall_severity'] = 'HIGH'
        if val_train_analysis['n_activity_cliffs'] > 0 and result['overall_severity'] == 'OK':
            result['overall_severity'] = 'MEDIUM'
    
    # Analyze val vs test (if val exists)
    if df_val is not None and val_fps is not None:
        val_test_analysis = analyze_split_similarity(
            df_test, df_val, 'test', 'val',
            test_fps, test_indices, test_smiles,
            val_fps, val_indices, val_smiles
        )
        result['val_vs_test'] = val_test_analysis
        
        result['between_split_similarity']['val_vs_test'] = {
            'avg_max_similarity': val_test_analysis['similarity_stats']['mean'],
            'max_similarity': val_test_analysis['similarity_stats']['max']
        }
        
        if val_test_analysis['n_high_similarity'] > 0:
            result['overall_severity'] = 'HIGH'
        if val_test_analysis['n_activity_cliffs'] > 0 and result['overall_severity'] == 'OK':
            result['overall_severity'] = 'MEDIUM'
    
    return result


def _detect_scaffold_leakage(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    max_examples: int = 10
) -> Dict:
    """
    Detect scaffold overlap between splits.
    
    Uses Murcko scaffolds to identify structural similarity at the scaffold level.
    Scaffold-based splitting should ideally result in zero scaffold overlap between
    train and test/val splits to prevent data leakage.
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    max_examples : int
        Maximum number of shared scaffold examples to include (default: 10)
        
    Returns
    -------
    Dict with:
        - scaffold overlap counts per split comparison
        - percentage of test/val scaffolds found in train
        - examples of shared scaffolds with molecule counts
        - severity: HIGH if significant overlap, MEDIUM if some overlap, OK if none
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Helper to compute scaffolds
    def compute_scaffolds(df, smiles_col):
        """Compute Murcko scaffolds for all valid SMILES."""
        scaffold_dict = {}  # scaffold_smiles -> list of molecule indices
        failed_count = 0
        
        for idx, smi in enumerate(df[smiles_col]):
            if pd.isna(smi) or smi == '':
                failed_count += 1
                continue
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed_count += 1
                continue
            
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                
                if scaffold_smiles not in scaffold_dict:
                    scaffold_dict[scaffold_smiles] = []
                scaffold_dict[scaffold_smiles].append(idx)
            except:
                failed_count += 1
                continue
        
        return scaffold_dict, failed_count
    
    # Helper to analyze scaffold overlap between two splits
    def analyze_scaffold_overlap(scaffolds1, scaffolds2, name1, name2):
        """Compare scaffold sets between two splits."""
        set1 = set(scaffolds1.keys())
        set2 = set(scaffolds2.keys())
        
        shared_scaffolds = set1 & set2
        n_shared = len(shared_scaffolds)
        
        if n_shared == 0:
            return {
                'n_shared_scaffolds': 0,
                'n_scaffolds_split1': len(set1),
                'n_scaffolds_split2': len(set2),
                'pct_split2_in_split1': 0.0,
                'pct_split1_in_split2': 0.0,
                'examples': []
            }
        
        # Calculate percentages
        pct_split2_in_split1 = (n_shared / len(set2) * 100) if len(set2) > 0 else 0
        pct_split1_in_split2 = (n_shared / len(set1) * 100) if len(set1) > 0 else 0
        
        # Collect examples with molecule counts
        examples = []
        for scaffold_smi in list(shared_scaffolds)[:max_examples]:
            examples.append({
                'scaffold_smiles': scaffold_smi[:100],  # Truncate long SMILES
                f'n_molecules_{name1}': len(scaffolds1[scaffold_smi]),
                f'n_molecules_{name2}': len(scaffolds2[scaffold_smi]),
                f'{name1}_indices': scaffolds1[scaffold_smi][:5],  # Limit to 5 examples
                f'{name2}_indices': scaffolds2[scaffold_smi][:5]
            })
        
        # Sort by total molecule count (most common scaffolds first)
        examples.sort(
            key=lambda x: x[f'n_molecules_{name1}'] + x[f'n_molecules_{name2}'],
            reverse=True
        )
        
        return {
            'n_shared_scaffolds': n_shared,
            'n_scaffolds_split1': len(set1),
            'n_scaffolds_split2': len(set2),
            'pct_split2_in_split1': round(pct_split2_in_split1, 2),
            'pct_split1_in_split2': round(pct_split1_in_split2, 2),
            'examples': examples,
            'showing_n_examples': min(n_shared, max_examples)
        }
    
    # Compute scaffolds for all splits
    train_scaffolds, train_failed = compute_scaffolds(df_train, smiles_col)
    test_scaffolds, test_failed = compute_scaffolds(df_test, smiles_col)
    val_scaffolds, val_failed = None, 0
    if df_val is not None:
        val_scaffolds, val_failed = compute_scaffolds(df_val, smiles_col)
    
    result = {
        'computation_stats': {
            'train_scaffolds_computed': len(train_scaffolds),
            'train_failed': train_failed,
            'test_scaffolds_computed': len(test_scaffolds),
            'test_failed': test_failed
        },
        'train_test_overlap': None,
        'train_val_overlap': None,
        'test_val_overlap': None,
        'overall_severity': 'OK'
    }
    
    if df_val is not None:
        result['computation_stats']['val_scaffolds_computed'] = len(val_scaffolds)
        result['computation_stats']['val_failed'] = val_failed
    
    # Analyze train/test overlap
    train_test_overlap = analyze_scaffold_overlap(train_scaffolds, test_scaffolds, 'train', 'test')
    result['train_test_overlap'] = train_test_overlap
    
    # Determine severity for test vs train
    # High severity if >50% of test scaffolds are in train
    # Medium severity if >20% of test scaffolds are in train
    if train_test_overlap['pct_split2_in_split1'] > 50:
        result['overall_severity'] = 'HIGH'
    elif train_test_overlap['pct_split2_in_split1'] > 20:
        result['overall_severity'] = 'MEDIUM'
    elif train_test_overlap['n_shared_scaffolds'] > 0:
        result['overall_severity'] = 'LOW'
    
    # Analyze train/val overlap
    if df_val is not None and val_scaffolds is not None:
        train_val_overlap = analyze_scaffold_overlap(train_scaffolds, val_scaffolds, 'train', 'val')
        result['train_val_overlap'] = train_val_overlap
        
        # Update severity
        if train_val_overlap['pct_split2_in_split1'] > 50:
            result['overall_severity'] = 'HIGH'
        elif train_val_overlap['pct_split2_in_split1'] > 20 and result['overall_severity'] in ['OK', 'LOW']:
            result['overall_severity'] = 'MEDIUM'
        elif train_val_overlap['n_shared_scaffolds'] > 0 and result['overall_severity'] == 'OK':
            result['overall_severity'] = 'LOW'
    
    # Analyze test/val overlap
    if df_val is not None and val_scaffolds is not None:
        test_val_overlap = analyze_scaffold_overlap(test_scaffolds, val_scaffolds, 'test', 'val')
        result['test_val_overlap'] = test_val_overlap
        
        # Note: test/val overlap is less critical than train/test or train/val
        # Only flag if very high overlap
        if test_val_overlap['pct_split2_in_split1'] > 70 and result['overall_severity'] == 'OK':
            result['overall_severity'] = 'LOW'
    
    return result


def _detect_stereoisomer_tautomer_leakage(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    max_examples: int = 10
) -> Dict:
    """
    Find molecules that differ only in stereochemistry or tautomeric form across splits.
    
    Stereoisomers: Same connectivity but different 3D arrangement (e.g., R vs S)
    Tautomers: Structural isomers that readily interconvert (e.g., keto-enol)
    
    These represent subtle forms of data leakage that can inflate model performance.
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    max_examples : int
        Maximum number of examples to include per category (default: 10)
        
    Returns
    -------
    Dict with:
        - stereoisomer_pairs: molecules differing only in stereochemistry
        - tautomer_pairs: molecules differing only in tautomeric form
        - counts per split comparison
        - severity level
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Initialize tautomer enumerator
    tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
    
    # Helper to create canonical forms
    def create_canonical_forms(df, smiles_col):
        """
        Create canonical representations for stereoisomer and tautomer detection.
        
        Returns:
            stereo_free_dict: canonical SMILES without stereochemistry -> list of (idx, original_smiles)
            tautomer_dict: canonical tautomer SMILES -> list of (idx, original_smiles)
        """
        stereo_free_dict = {}
        tautomer_dict = {}
        
        for idx, smi in enumerate(df[smiles_col]):
            if pd.isna(smi) or smi == '':
                continue
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            # Create stereochemistry-free representation
            try:
                # Remove stereochemistry by converting to SMILES without stereo info
                Chem.RemoveStereochemistry(mol)
                stereo_free_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                
                if stereo_free_smiles not in stereo_free_dict:
                    stereo_free_dict[stereo_free_smiles] = []
                stereo_free_dict[stereo_free_smiles].append((idx, smi))
            except:
                pass
            
            # Create canonical tautomer representation
            try:
                # Re-parse original SMILES for tautomer analysis
                mol_taut = Chem.MolFromSmiles(smi)
                if mol_taut is not None:
                    canonical_tautomer = tautomer_enumerator.Canonicalize(mol_taut)
                    tautomer_smiles = Chem.MolToSmiles(canonical_tautomer)
                    
                    if tautomer_smiles not in tautomer_dict:
                        tautomer_dict[tautomer_smiles] = []
                    tautomer_dict[tautomer_smiles].append((idx, smi))
            except:
                pass
        
        return stereo_free_dict, tautomer_dict
    
    # Helper to find matches between splits
    def find_isomer_matches(dict1, dict2, name1, name2, match_type='stereoisomer'):
        """Find molecules that share canonical form but have different original SMILES."""
        matches = []
        
        # Find shared canonical forms
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        shared_keys = keys1 & keys2
        
        for canonical_form in shared_keys:
            entries1 = dict1[canonical_form]
            entries2 = dict2[canonical_form]
            
            # Check if any original SMILES differ
            # (same canonical form but different original = stereoisomer/tautomer)
            smiles_set1 = {smi for _, smi in entries1}
            smiles_set2 = {smi for _, smi in entries2}
            
            # Find different SMILES that map to same canonical form
            for idx1, smi1 in entries1:
                for idx2, smi2 in entries2:
                    if smi1 != smi2:  # Different original SMILES
                        matches.append({
                            'canonical_form': canonical_form[:100],
                            f'{name1}_index': idx1,
                            f'{name1}_smiles': smi1[:100],
                            f'{name2}_index': idx2,
                            f'{name2}_smiles': smi2[:100],
                            'type': match_type
                        })
        
        return matches
    
    # Compute canonical forms for all splits
    train_stereo, train_taut = create_canonical_forms(df_train, smiles_col)
    test_stereo, test_taut = create_canonical_forms(df_test, smiles_col)
    val_stereo, val_taut = None, None
    if df_val is not None:
        val_stereo, val_taut = create_canonical_forms(df_val, smiles_col)
    
    result = {
        'train_test_stereoisomers': None,
        'train_test_tautomers': None,
        'train_val_stereoisomers': None,
        'train_val_tautomers': None,
        'test_val_stereoisomers': None,
        'test_val_tautomers': None,
        'overall_severity': 'OK'
    }
    
    total_stereoisomer_pairs = 0
    total_tautomer_pairs = 0
    
    # Analyze train/test
    train_test_stereo = find_isomer_matches(train_stereo, test_stereo, 'train', 'test', 'stereoisomer')
    train_test_taut = find_isomer_matches(train_taut, test_taut, 'train', 'test', 'tautomer')
    
    result['train_test_stereoisomers'] = {
        'n_pairs': len(train_test_stereo),
        'examples': train_test_stereo[:max_examples],
        'showing_n_examples': min(len(train_test_stereo), max_examples)
    }
    result['train_test_tautomers'] = {
        'n_pairs': len(train_test_taut),
        'examples': train_test_taut[:max_examples],
        'showing_n_examples': min(len(train_test_taut), max_examples)
    }
    
    total_stereoisomer_pairs += len(train_test_stereo)
    total_tautomer_pairs += len(train_test_taut)
    
    # Analyze train/val
    if df_val is not None and val_stereo is not None:
        train_val_stereo = find_isomer_matches(train_stereo, val_stereo, 'train', 'val', 'stereoisomer')
        train_val_taut = find_isomer_matches(train_taut, val_taut, 'train', 'val', 'tautomer')
        
        result['train_val_stereoisomers'] = {
            'n_pairs': len(train_val_stereo),
            'examples': train_val_stereo[:max_examples],
            'showing_n_examples': min(len(train_val_stereo), max_examples)
        }
        result['train_val_tautomers'] = {
            'n_pairs': len(train_val_taut),
            'examples': train_val_taut[:max_examples],
            'showing_n_examples': min(len(train_val_taut), max_examples)
        }
        
        total_stereoisomer_pairs += len(train_val_stereo)
        total_tautomer_pairs += len(train_val_taut)
    
    # Analyze test/val
    if df_val is not None and val_stereo is not None:
        test_val_stereo = find_isomer_matches(test_stereo, val_stereo, 'test', 'val', 'stereoisomer')
        test_val_taut = find_isomer_matches(test_taut, val_taut, 'test', 'val', 'tautomer')
        
        result['test_val_stereoisomers'] = {
            'n_pairs': len(test_val_stereo),
            'examples': test_val_stereo[:max_examples],
            'showing_n_examples': min(len(test_val_stereo), max_examples)
        }
        result['test_val_tautomers'] = {
            'n_pairs': len(test_val_taut),
            'examples': test_val_taut[:max_examples],
            'showing_n_examples': min(len(test_val_taut), max_examples)
        }
        
        total_stereoisomer_pairs += len(test_val_stereo)
        total_tautomer_pairs += len(test_val_taut)
    
    # Determine severity
    # Stereoisomers and tautomers are subtle leakage - flag as MEDIUM if found
    if total_stereoisomer_pairs > 0 or total_tautomer_pairs > 0:
        result['overall_severity'] = 'MEDIUM'
    
    result['total_stereoisomer_pairs'] = total_stereoisomer_pairs
    result['total_tautomer_pairs'] = total_tautomer_pairs
    
    return result


def _test_property_distributions(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    alpha: float = 0.05
) -> Dict:
    """
    Test if physicochemical property distributions differ significantly between splits.
    
    Uses Kolmogorov-Smirnov (KS) test to compare distributions. Significant differences
    may indicate biased splitting that could affect model generalization.
    
    Properties tested:
    - Molecular Weight (MolWt)
    - LogP (octanol-water partition coefficient)
    - TPSA (Topological Polar Surface Area)
    - Number of H-bond donors (NumHDonors)
    - Number of H-bond acceptors (NumHAcceptors)
    - Number of rotatable bonds (NumRotatableBonds)
    - Number of aromatic rings (NumAromaticRings)
    - Number of heavy atoms (NumHeavyAtoms)
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    alpha : float
        Significance level for statistical tests (default: 0.05)
        
    Returns
    -------
    Dict with:
        - KS test results per property per split comparison
        - p-values, test statistics, interpretations
        - severity flag if significant differences found
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Define properties to compute
    property_functions = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'TPSA': Descriptors.TPSA,
        'NumHDonors': Lipinski.NumHDonors,
        'NumHAcceptors': Lipinski.NumHAcceptors,
        'NumRotatableBonds': Lipinski.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumHeavyAtoms': Lipinski.HeavyAtomCount
    }
    
    # Helper to compute properties for a dataset
    def compute_properties(df, smiles_col):
        """Compute all properties for valid molecules."""
        properties = {prop: [] for prop in property_functions.keys()}
        n_computed = 0
        n_failed = 0
        
        for smi in df[smiles_col]:
            if pd.isna(smi) or smi == '':
                n_failed += 1
                continue
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_failed += 1
                continue
            
            try:
                for prop_name, prop_func in property_functions.items():
                    value = prop_func(mol)
                    properties[prop_name].append(value)
                n_computed += 1
            except:
                n_failed += 1
                continue
        
        return properties, n_computed, n_failed
    
    # Helper to run KS test between two property distributions
    def compare_distributions(props1, props2, name1, name2):
        """Run KS tests for all properties between two splits."""
        results = {}
        
        for prop_name in property_functions.keys():
            values1 = np.array(props1[prop_name])
            values2 = np.array(props2[prop_name])
            
            if len(values1) < 3 or len(values2) < 3:
                results[prop_name] = {
                    'error': 'Insufficient data for statistical test',
                    'n_samples_split1': len(values1),
                    'n_samples_split2': len(values2)
                }
                continue
            
            # Run KS test
            ks_stat, p_value = stats.ks_2samp(values1, values2)
            
            # Interpret results
            significant = p_value < alpha
            interpretation = 'DIFFERENT' if significant else 'SIMILAR'
            
            # Compute summary statistics for context
            results[prop_name] = {
                'ks_statistic': round(float(ks_stat), 4),
                'p_value': round(float(p_value), 6),
                'significant': significant,
                'interpretation': interpretation,
                f'{name1}_mean': round(float(np.mean(values1)), 3),
                f'{name1}_std': round(float(np.std(values1)), 3),
                f'{name1}_median': round(float(np.median(values1)), 3),
                f'{name2}_mean': round(float(np.mean(values2)), 3),
                f'{name2}_std': round(float(np.std(values2)), 3),
                f'{name2}_median': round(float(np.median(values2)), 3),
                'n_samples_split1': len(values1),
                'n_samples_split2': len(values2)
            }
        
        return results
    
    # Compute properties for all splits
    train_props, train_computed, train_failed = compute_properties(df_train, smiles_col)
    test_props, test_computed, test_failed = compute_properties(df_test, smiles_col)
    val_props, val_computed, val_failed = None, 0, 0
    if df_val is not None:
        val_props, val_computed, val_failed = compute_properties(df_val, smiles_col)
    
    result = {
        'alpha': alpha,
        'properties_tested': list(property_functions.keys()),
        'computation_stats': {
            'train_computed': train_computed,
            'train_failed': train_failed,
            'test_computed': test_computed,
            'test_failed': test_failed
        },
        'train_vs_test': None,
        'train_vs_val': None,
        'test_vs_val': None,
        'overall_severity': 'OK'
    }
    
    if df_val is not None:
        result['computation_stats']['val_computed'] = val_computed
        result['computation_stats']['val_failed'] = val_failed
    
    # Compare train vs test
    train_test_results = compare_distributions(train_props, test_props, 'train', 'test')
    result['train_vs_test'] = train_test_results
    
    # Count significant differences
    n_significant_train_test = sum(
        1 for prop_result in train_test_results.values() 
        if isinstance(prop_result, dict) and prop_result.get('significant', False)
    )
    
    if n_significant_train_test > 0:
        result['overall_severity'] = 'MEDIUM'
    
    # Compare train vs val
    if df_val is not None and val_props is not None:
        train_val_results = compare_distributions(train_props, val_props, 'train', 'val')
        result['train_vs_val'] = train_val_results
        
        n_significant_train_val = sum(
            1 for prop_result in train_val_results.values() 
            if isinstance(prop_result, dict) and prop_result.get('significant', False)
        )
        
        if n_significant_train_val > 0:
            result['overall_severity'] = 'MEDIUM'
    
    # Compare test vs val
    if df_val is not None and val_props is not None:
        test_val_results = compare_distributions(test_props, val_props, 'test', 'val')
        result['test_vs_val'] = test_val_results
        
        n_significant_test_val = sum(
            1 for prop_result in test_val_results.values() 
            if isinstance(prop_result, dict) and prop_result.get('significant', False)
        )
        
        # Test/val differences less critical, only flag if many properties differ
        if n_significant_test_val > len(property_functions) / 2:
            if result['overall_severity'] == 'OK':
                result['overall_severity'] = 'LOW'
    
    # Add summary counts
    result['summary'] = {
        'n_properties_tested': len(property_functions),
        'n_significant_train_test': n_significant_train_test if 'train_vs_test' in result else 0
    }
    
    if df_val is not None:
        result['summary']['n_significant_train_val'] = n_significant_train_val if 'train_vs_val' in result else 0
        result['summary']['n_significant_test_val'] = n_significant_test_val if 'test_vs_val' in result else 0
    
    return result


def _test_activity_distributions(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    label_col: str,
    alpha: float = 0.05,
    imbalance_threshold: float = 0.3
) -> Dict:
    """
    Compare activity/label distributions across splits using appropriate statistical tests.
    
    For regression: Uses Kolmogorov-Smirnov (KS) test to compare distributions
    For classification: Uses Chi-square test for proportion differences and checks class balance
    
    Significant differences may indicate biased splitting that could affect model performance.
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    label_col : str
        Name of label column
    alpha : float
        Significance level for statistical tests (default: 0.05)
    imbalance_threshold : float
        Threshold for flagging class imbalance (default: 0.3)
        If minority class < threshold * majority class, flag as imbalanced
        
    Returns
    -------
    Dict with:
        - Statistical test results per split comparison
        - For classification: class balance metrics per split
        - p-values, test statistics, interpretations
        - severity flags
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Check if label column exists
    if label_col not in df_train.columns:
        return {'error': f'Label column "{label_col}" not found in training data'}
    
    # Get all labels to determine task type
    all_labels = []
    if label_col in df_train.columns:
        all_labels.extend(df_train[label_col].dropna().tolist())
    if label_col in df_test.columns:
        all_labels.extend(df_test[label_col].dropna().tolist())
    if df_val is not None and label_col in df_val.columns:
        all_labels.extend(df_val[label_col].dropna().tolist())
    
    if len(all_labels) == 0:
        return {'error': 'No valid labels found in any split'}
    
    # Determine if classification or regression
    unique_labels = set(all_labels)
    is_classification = len(unique_labels) <= 20 and all(
        isinstance(x, (int, np.integer)) or x in [0, 1, 0.0, 1.0] 
        for x in unique_labels
    )
    
    result = {
        'task_type': 'classification' if is_classification else 'regression',
        'alpha': alpha,
        'train_vs_test': None,
        'train_vs_val': None,
        'test_vs_val': None,
        'overall_severity': 'OK'
    }
    
    if is_classification:
        # Classification: use Chi-square test and check class balance
        
        def analyze_classification_splits(df1, df2, name1, name2):
            """Compare class distributions between two splits using Chi-square test."""
            labels1 = df1[label_col].dropna() if label_col in df1.columns else pd.Series([])
            labels2 = df2[label_col].dropna() if label_col in df2.columns else pd.Series([])
            
            if len(labels1) < 5 or len(labels2) < 5:
                return {
                    'error': 'Insufficient data for statistical test',
                    'n_samples_split1': len(labels1),
                    'n_samples_split2': len(labels2)
                }
            
            # Get class counts for both splits
            counts1 = Counter(labels1)
            counts2 = Counter(labels2)
            
            # Get all classes
            all_classes = sorted(set(counts1.keys()) | set(counts2.keys()))
            
            # Create contingency table
            observed = []
            for cls in all_classes:
                observed.append([counts1.get(cls, 0), counts2.get(cls, 0)])
            observed = np.array(observed)
            
            # Check if chi-square test is valid (all expected frequencies >= 5)
            if observed.sum() < 10 or np.any(observed.sum(axis=1) < 2):
                return {
                    'error': 'Insufficient data for Chi-square test',
                    'n_samples_split1': len(labels1),
                    'n_samples_split2': len(labels2)
                }
            
            # Run Chi-square test
            try:
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed.T)
            except:
                return {
                    'error': 'Chi-square test failed',
                    'n_samples_split1': len(labels1),
                    'n_samples_split2': len(labels2)
                }
            
            # Interpret results
            significant = p_value < alpha
            interpretation = 'DIFFERENT' if significant else 'SIMILAR'
            
            # Compute class proportions
            props1 = {cls: counts1[cls] / len(labels1) for cls in all_classes}
            props2 = {cls: counts2[cls] / len(labels2) for cls in all_classes}
            
            return {
                'chi2_statistic': round(float(chi2_stat), 4),
                'p_value': round(float(p_value), 6),
                'degrees_of_freedom': int(dof),
                'significant': significant,
                'interpretation': interpretation,
                'n_classes': len(all_classes),
                'classes': [str(cls) for cls in all_classes],
                f'{name1}_counts': {str(k): int(v) for k, v in counts1.items()},
                f'{name2}_counts': {str(k): int(v) for k, v in counts2.items()},
                f'{name1}_proportions': {str(k): round(v, 4) for k, v in props1.items()},
                f'{name2}_proportions': {str(k): round(v, 4) for k, v in props2.items()},
                'n_samples_split1': len(labels1),
                'n_samples_split2': len(labels2)
            }
        
        # Helper to check class balance within a split
        def check_class_balance(df, split_name):
            """Check if classes are balanced within a single split."""
            labels = df[label_col].dropna() if label_col in df.columns else pd.Series([])
            
            if len(labels) == 0:
                return {'error': 'No valid labels'}
            
            counts = Counter(labels)
            
            if len(counts) < 2:
                return {
                    'n_classes': len(counts),
                    'imbalanced': False,
                    'message': 'Only one class present'
                }
            
            # Check imbalance
            max_count = max(counts.values())
            min_count = min(counts.values())
            ratio = min_count / max_count if max_count > 0 else 0
            
            imbalanced = ratio < imbalance_threshold
            
            # Calculate proportions
            proportions = {str(k): round(v / len(labels), 4) for k, v in counts.items()}
            
            return {
                'n_samples': len(labels),
                'n_classes': len(counts),
                'class_counts': {str(k): int(v) for k, v in counts.items()},
                'class_proportions': proportions,
                'imbalanced': imbalanced,
                'min_class_ratio': round(ratio, 4),
                'majority_class': str(max(counts, key=counts.get)),
                'minority_class': str(min(counts, key=counts.get))
            }
        
        # Analyze train vs test
        result['train_vs_test'] = analyze_classification_splits(df_train, df_test, 'train', 'test')
        if result['train_vs_test'].get('significant', False):
            result['overall_severity'] = 'MEDIUM'
        
        # Analyze train vs val
        if df_val is not None:
            result['train_vs_val'] = analyze_classification_splits(df_train, df_val, 'train', 'val')
            if result['train_vs_val'].get('significant', False):
                result['overall_severity'] = 'MEDIUM'
        
        # Analyze test vs val
        if df_val is not None:
            result['test_vs_val'] = analyze_classification_splits(df_test, df_val, 'test', 'val')
            if result['test_vs_val'].get('significant', False) and result['overall_severity'] == 'OK':
                result['overall_severity'] = 'LOW'
        
        # Check class balance within each split
        result['class_balance'] = {
            'train': check_class_balance(df_train, 'train'),
            'test': check_class_balance(df_test, 'test')
        }
        
        if df_val is not None:
            result['class_balance']['val'] = check_class_balance(df_val, 'val')
        
        # Flag if any split is imbalanced
        if any(balance.get('imbalanced', False) for balance in result['class_balance'].values()):
            if result['overall_severity'] == 'OK':
                result['overall_severity'] = 'LOW'
    
    else:
        # Regression: use KS test
        
        def analyze_regression_splits(df1, df2, name1, name2):
            """Compare activity distributions between two splits using KS test."""
            values1 = df1[label_col].dropna().values if label_col in df1.columns else np.array([])
            values2 = df2[label_col].dropna().values if label_col in df2.columns else np.array([])
            
            if len(values1) < 3 or len(values2) < 3:
                return {
                    'error': 'Insufficient data for statistical test',
                    'n_samples_split1': len(values1),
                    'n_samples_split2': len(values2)
                }
            
            # Run KS test
            ks_stat, p_value = stats.ks_2samp(values1, values2)
            
            # Interpret results
            significant = p_value < alpha
            interpretation = 'DIFFERENT' if significant else 'SIMILAR'
            
            # Compute summary statistics
            return {
                'ks_statistic': round(float(ks_stat), 4),
                'p_value': round(float(p_value), 6),
                'significant': significant,
                'interpretation': interpretation,
                f'{name1}_mean': round(float(np.mean(values1)), 3),
                f'{name1}_std': round(float(np.std(values1)), 3),
                f'{name1}_median': round(float(np.median(values1)), 3),
                f'{name1}_min': round(float(np.min(values1)), 3),
                f'{name1}_max': round(float(np.max(values1)), 3),
                f'{name2}_mean': round(float(np.mean(values2)), 3),
                f'{name2}_std': round(float(np.std(values2)), 3),
                f'{name2}_median': round(float(np.median(values2)), 3),
                f'{name2}_min': round(float(np.min(values2)), 3),
                f'{name2}_max': round(float(np.max(values2)), 3),
                'n_samples_split1': len(values1),
                'n_samples_split2': len(values2)
            }
        
        # Analyze train vs test
        result['train_vs_test'] = analyze_regression_splits(df_train, df_test, 'train', 'test')
        if result['train_vs_test'].get('significant', False):
            result['overall_severity'] = 'MEDIUM'
        
        # Analyze train vs val
        if df_val is not None:
            result['train_vs_val'] = analyze_regression_splits(df_train, df_val, 'train', 'val')
            if result['train_vs_val'].get('significant', False):
                result['overall_severity'] = 'MEDIUM'
        
        # Analyze test vs val
        if df_val is not None:
            result['test_vs_val'] = analyze_regression_splits(df_test, df_val, 'test', 'val')
            if result['test_vs_val'].get('significant', False) and result['overall_severity'] == 'OK':
                result['overall_severity'] = 'LOW'
    
    return result


def _analyze_functional_group_distribution(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    min_occurrence_threshold: int = 2
) -> Dict:
    """
    Analyze functional group distribution across splits and identify unique groups.
    
    This analysis is intentionally lenient - many functional groups will naturally
    be shared across splits. The focus is on identifying groups that appear ONLY
    in one split, which could indicate biased splitting or limited chemical diversity.
    
    Functional groups detected:
    - Aromatic rings, Carbonyl, Carboxylic acid, Amide, Ether, Ester
    - Alcohol, Ketone, Aldehyde, Amine (primary/secondary/tertiary)
    - Halogen (with F/Cl/Br/I breakdown), Nitro, Sulfur-containing
    
    Parameters
    ----------
    train_path : str
        Filename of training split resource
    test_path : str
        Filename of test split resource
    val_path : Optional[str]
        Filename of validation split resource (if exists)
    project_manifest_path : str
        Path to manifest.json
    smiles_col : str
        Name of SMILES column
    min_occurrence_threshold : int
        Minimum occurrences to report a unique group (default: 2)
        Groups appearing only once may be noise
        
    Returns
    -------
    Dict with:
        - functional group counts per split
        - groups unique to each split
        - groups shared across splits
        - overall distribution metrics
    """
    # Load datasets
    df_train = _load_resource(project_manifest_path, train_path)
    df_test = _load_resource(project_manifest_path, test_path)
    df_val = _load_resource(project_manifest_path, val_path) if val_path else None
    
    # Define functional groups to detect
    functional_groups = {
        'Aromatic rings': lambda mol: Descriptors.NumAromaticRings(mol),
        'Carbonyl': lambda mol: Fragments.fr_C_O(mol),
        'Carboxylic acid': lambda mol: Fragments.fr_COO(mol) + Fragments.fr_COO2(mol),
        'Amide': lambda mol: Fragments.fr_amide(mol),
        'Ether': lambda mol: Fragments.fr_ether(mol),
        'Ester': lambda mol: Fragments.fr_ester(mol),
        'Alcohol': lambda mol: Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol),
        'Ketone': lambda mol: Fragments.fr_ketone(mol),
        'Aldehyde': lambda mol: Fragments.fr_aldehyde(mol),
        'Primary amine': lambda mol: Fragments.fr_NH2(mol),
        'Secondary amine': lambda mol: Fragments.fr_NH1(mol),
        'Tertiary amine': lambda mol: Fragments.fr_NH0(mol),
        'Halogen': lambda mol: Fragments.fr_halogen(mol),
        'Fluorine': lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9]),
        'Chlorine': lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17]),
        'Bromine': lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 35]),
        'Iodine': lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 53]),
        'Nitro': lambda mol: Fragments.fr_nitro(mol),
        'Sulfur-containing': lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16])
    }
    
    # Helper to detect functional groups in a dataset
    def detect_functional_groups(df, smiles_col):
        """Detect all functional groups and count molecules containing each."""
        group_counts = {group: 0 for group in functional_groups.keys()}
        n_molecules = 0
        n_failed = 0
        
        for smi in df[smiles_col]:
            if pd.isna(smi) or smi == '':
                n_failed += 1
                continue
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_failed += 1
                continue
            
            n_molecules += 1
            
            # Check each functional group
            for group_name, group_func in functional_groups.items():
                try:
                    count = group_func(mol)
                    if count > 0:
                        group_counts[group_name] += 1
                except:
                    continue
        
        return group_counts, n_molecules, n_failed
    
    # Detect functional groups in all splits
    train_groups, train_n_mol, train_failed = detect_functional_groups(df_train, smiles_col)
    test_groups, test_n_mol, test_failed = detect_functional_groups(df_test, smiles_col)
    val_groups, val_n_mol, val_failed = None, 0, 0
    if df_val is not None:
        val_groups, val_n_mol, val_failed = detect_functional_groups(df_val, smiles_col)
    
    # Compute percentages
    train_percentages = {
        group: round(count / train_n_mol * 100, 2) if train_n_mol > 0 else 0
        for group, count in train_groups.items()
    }
    test_percentages = {
        group: round(count / test_n_mol * 100, 2) if test_n_mol > 0 else 0
        for group, count in test_groups.items()
    }
    val_percentages = None
    if val_groups is not None:
        val_percentages = {
            group: round(count / val_n_mol * 100, 2) if val_n_mol > 0 else 0
            for group, count in val_groups.items()
        }
    
    # Identify groups unique to each split (present in one split but not others)
    unique_to_train = []
    unique_to_test = []
    unique_to_val = []
    
    for group in functional_groups.keys():
        train_has = train_groups[group] >= min_occurrence_threshold
        test_has = test_groups[group] >= min_occurrence_threshold
        val_has = val_groups and val_groups[group] >= min_occurrence_threshold
        
        if train_has and not test_has and not val_has:
            unique_to_train.append({
                'group': group,
                'count': train_groups[group],
                'pct_molecules': train_percentages[group]
            })
        
        if test_has and not train_has and not val_has:
            unique_to_test.append({
                'group': group,
                'count': test_groups[group],
                'pct_molecules': test_percentages[group]
            })
        
        if val_has and not train_has and not test_has:
            unique_to_val.append({
                'group': group,
                'count': val_groups[group],
                'pct_molecules': val_percentages[group]
            })
    
    # Identify groups shared across all splits
    shared_groups = []
    for group in functional_groups.keys():
        train_has = train_groups[group] > 0
        test_has = test_groups[group] > 0
        val_has = val_groups is None or val_groups[group] > 0
        
        if train_has and test_has and val_has:
            shared_groups.append(group)
    
    result = {
        'n_functional_groups_tested': len(functional_groups),
        'computation_stats': {
            'train_molecules': train_n_mol,
            'train_failed': train_failed,
            'test_molecules': test_n_mol,
            'test_failed': test_failed
        },
        'train': {
            'counts': train_groups,
            'percentages': train_percentages
        },
        'test': {
            'counts': test_groups,
            'percentages': test_percentages
        },
        'unique_to_train': unique_to_train,
        'unique_to_test': unique_to_test,
        'shared_across_all_splits': shared_groups,
        'n_shared_groups': len(shared_groups),
        'overall_severity': 'OK'
    }
    
    if df_val is not None:
        result['computation_stats']['val_molecules'] = val_n_mol
        result['computation_stats']['val_failed'] = val_failed
        result['val'] = {
            'counts': val_groups,
            'percentages': val_percentages
        }
        result['unique_to_val'] = unique_to_val
    
    # Determine severity
    # Flag if multiple groups are unique to test/val (potential bias)
    n_unique_test = len(unique_to_test)
    n_unique_val = len(unique_to_val) if df_val is not None else 0
    
    if n_unique_test > 3 or n_unique_val > 3:
        result['overall_severity'] = 'MEDIUM'
    elif n_unique_test > 0 or n_unique_val > 0:
        result['overall_severity'] = 'LOW'
    
    # Add summary
    result['summary'] = {
        'n_unique_to_train': len(unique_to_train),
        'n_unique_to_test': n_unique_test,
        'n_shared': len(shared_groups),
        'pct_shared': round(len(shared_groups) / len(functional_groups) * 100, 1)
    }
    
    if df_val is not None:
        result['summary']['n_unique_to_val'] = n_unique_val
    
    return result


# ============================================================================
# AGGREGATOR FUNCTION
# ============================================================================

def _analyze_split_quality(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    label_col: str,
    output_filename: str,
    explanation: str = "Data splitting quality analysis report",
    # Parameters for individual helpers
    min_split_size: int = 50,
    imbalance_threshold: float = 0.1,
    similarity_threshold: float = 0.9,
    activity_cliff_similarity: float = 0.8,
    activity_cliff_fold_diff: float = 10.0,
    alpha: float = 0.05,
    min_occurrence_threshold: int = 2,
    max_examples: int = 10
) -> Dict:
    """
    Comprehensive data splitting quality analysis combining all quality checks.
    
    Runs 8 quality checks: split characteristics, exact duplicates, similarity leakage,
    scaffold overlap, stereoisomer/tautomer leakage, property distributions,
    activity distributions, and functional group distributions.
    
    Parameters
    ----------
    train_path : str
        Training split filename.
    test_path : str
        Test split filename.
    val_path : Optional[str]
        Validation split filename (if exists).
    project_manifest_path : str
        Path to manifest.json.
    smiles_col : str
        SMILES column name.
    label_col : str
        Label column name.
    output_filename : str
        Output filename prefix.
    explanation : str
        Description for manifest.
    min_split_size : int, default=50
        Minimum acceptable split size.
    imbalance_threshold : float, default=0.1
        Class imbalance threshold.
    similarity_threshold : float, default=0.9
        Tanimoto similarity threshold for leakage.
    activity_cliff_similarity : float, default=0.8
        Similarity threshold for activity cliffs.
    activity_cliff_fold_diff : float, default=10.0
        Fold difference for regression activity cliffs.
    alpha : float, default=0.05
        Significance level for statistical tests.
    min_occurrence_threshold : int, default=2
        Minimum functional group occurrences to flag.
    max_examples : int, default=10
        Maximum examples per issue.
        
    Returns
    -------
    Dict
        Contains output_filename, overall_severity, severity_summary, split_characteristics,
        exact_duplicates, similarity_leakage, scaffold_leakage, stereoisomer_tautomer_leakage,
        property_distributions, activity_distributions, functional_groups, metadata.
    """
    import json
    from datetime import datetime
    from molml_mcp.infrastructure.resources import _store_resource
    
    # Store start time
    start_time = datetime.now()
    
    # Initialize result structure
    result = {
        'metadata': {
            'analysis_type': 'data_splitting_quality',
            'timestamp': start_time.isoformat(),
            'train_file': train_path,
            'test_file': test_path,
            'val_file': val_path,
            'smiles_column': smiles_col,
            'label_column': label_col,
            'parameters': {
                'min_split_size': min_split_size,
                'imbalance_threshold': imbalance_threshold,
                'similarity_threshold': similarity_threshold,
                'activity_cliff_similarity': activity_cliff_similarity,
                'activity_cliff_fold_diff': activity_cliff_fold_diff,
                'alpha': alpha,
                'min_occurrence_threshold': min_occurrence_threshold,
                'max_examples': max_examples
            }
        }
    }
    
    # 1. Analyze split characteristics
    print("Running split characteristics analysis...")
    result['split_characteristics'] = _analyze_split_characteristics(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, label_col, min_split_size, imbalance_threshold
    )
    
    # 2. Detect exact duplicates (CRITICAL if found)
    print("Detecting exact duplicates...")
    result['exact_duplicates'] = _detect_exact_duplicates(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, max_examples
    )
    
    # 3. Detect similarity-based leakage
    print("Detecting similarity-based leakage...")
    result['similarity_leakage'] = _detect_similarity_leakage(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, label_col, similarity_threshold,
        activity_cliff_similarity, activity_cliff_fold_diff, max_examples
    )
    
    # 4. Detect scaffold leakage
    print("Detecting scaffold overlap...")
    result['scaffold_leakage'] = _detect_scaffold_leakage(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, max_examples
    )
    
    # 5. Detect stereoisomer/tautomer leakage
    print("Detecting stereoisomer/tautomer leakage...")
    result['stereoisomer_tautomer_leakage'] = _detect_stereoisomer_tautomer_leakage(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, max_examples
    )
    
    # 6. Test property distributions
    print("Testing property distributions...")
    result['property_distributions'] = _test_property_distributions(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, alpha
    )
    
    # 7. Test activity distributions
    print("Testing activity distributions...")
    result['activity_distributions'] = _test_activity_distributions(
        train_path, test_path, val_path, project_manifest_path,
        label_col, alpha, imbalance_threshold
    )
    
    # 8. Analyze functional group distribution
    print("Analyzing functional group distributions...")
    result['functional_groups'] = _analyze_functional_group_distribution(
        train_path, test_path, val_path, project_manifest_path,
        smiles_col, min_occurrence_threshold
    )
    
    # Determine overall severity (highest severity found)
    severity_levels = ['OK', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    severities = []
    
    # Collect all severity values
    if 'severity' in result['split_characteristics']:
        severities.append(result['split_characteristics']['severity'])
    if 'severity' in result['exact_duplicates']:
        severities.append(result['exact_duplicates']['severity'])
    if 'overall_severity' in result['similarity_leakage']:
        severities.append(result['similarity_leakage']['overall_severity'])
    if 'overall_severity' in result['scaffold_leakage']:
        severities.append(result['scaffold_leakage']['overall_severity'])
    if 'overall_severity' in result['stereoisomer_tautomer_leakage']:
        severities.append(result['stereoisomer_tautomer_leakage']['overall_severity'])
    if 'overall_severity' in result['property_distributions']:
        severities.append(result['property_distributions']['overall_severity'])
    if 'overall_severity' in result['activity_distributions']:
        severities.append(result['activity_distributions']['overall_severity'])
    if 'overall_severity' in result['functional_groups']:
        severities.append(result['functional_groups']['overall_severity'])
    
    # Find highest severity
    severity_indices = [severity_levels.index(s) for s in severities if s in severity_levels]
    overall_severity_idx = max(severity_indices) if severity_indices else 0
    overall_severity = severity_levels[overall_severity_idx]
    
    result['overall_severity'] = overall_severity
    
    # Count issues by severity
    from collections import Counter
    severity_counts = Counter(severities)
    result['severity_summary'] = {
        'CRITICAL': severity_counts.get('CRITICAL', 0),
        'HIGH': severity_counts.get('HIGH', 0),
        'MEDIUM': severity_counts.get('MEDIUM', 0),
        'LOW': severity_counts.get('LOW', 0),
        'OK': severity_counts.get('OK', 0)
    }
    
    # Add execution time
    end_time = datetime.now()
    result['metadata']['execution_time_seconds'] = (end_time - start_time).total_seconds()
    result['metadata']['completed_at'] = end_time.isoformat()
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    result = convert_numpy_types(result)
    
    # Save report as JSON
    print(f"Saving report to {output_filename}...")
    output_file = _store_resource(
        result,
        project_manifest_path,
        output_filename,
        explanation,
        'json'
    )
    
    # Return summary
    return {
        'output_filename': output_file,
        'overall_severity': overall_severity,
        'severity_summary': result['severity_summary'],
        'n_checks_performed': 8,
        'execution_time_seconds': result['metadata']['execution_time_seconds'],
        'issues_found': {
            'exact_duplicates': result['exact_duplicates'].get('total_duplicate_molecules', 0),
            'high_similarity_pairs': sum(
                result['similarity_leakage'].get(k, {}).get('n_high_similarity', 0)
                for k in ['test_vs_train', 'val_vs_train', 'val_vs_test']
                if result['similarity_leakage'].get(k) is not None
            ),
            'activity_cliffs': sum(
                result['similarity_leakage'].get(k, {}).get('n_activity_cliffs', 0)
                for k in ['test_vs_train', 'val_vs_train', 'val_vs_test']
                if result['similarity_leakage'].get(k) is not None
            ),
            'scaffold_overlap_pct': result['scaffold_leakage'].get('train_test_overlap', {}).get('pct_split2_in_split1', 0),
            'stereoisomer_pairs': result['stereoisomer_tautomer_leakage'].get('total_stereoisomer_pairs', 0),
            'tautomer_pairs': result['stereoisomer_tautomer_leakage'].get('total_tautomer_pairs', 0),
            'significant_property_diffs': result['property_distributions'].get('summary', {}).get('n_significant_train_test', 0),
            'activity_distribution_different': result['activity_distributions'].get('train_vs_test', {}).get('significant', False),
            'unique_functional_groups_test': len(result['functional_groups'].get('unique_to_test', []))
        }
    }


# ============================================================================
# TEXT REPORT WRITER
# ============================================================================

def data_split_quality_analysis(
    train_path: str,
    test_path: str,
    val_path: Optional[str],
    project_manifest_path: str,
    smiles_col: str,
    label_col: str,
    output_filename: str,
    explanation: str = "Data splitting quality text report",
    # Parameters for quality analysis
    min_split_size: int = 50,
    imbalance_threshold: float = 0.1,
    similarity_threshold: float = 0.9,
    activity_cliff_similarity: float = 0.8,
    activity_cliff_fold_diff: float = 10.0,
    alpha: float = 0.05,
    min_occurrence_threshold: int = 2,
    max_examples: int = 10
) -> Dict:
    """
    🚀 PRIMARY TOOL FOR DATA SPLIT ANALYSIS & DATA LEAKAGE DETECTION 🚀
    
    Comprehensive train/test/val split quality analysis with 8 diagnostic checks:
    1. Split characteristics (sizes, ratios, class balance)
    2. Exact duplicate detection (CRITICAL leakage)
    3. Similarity-based leakage (high similarity pairs, activity cliffs)
    4. Scaffold overlap (structural leakage)
    5. Stereoisomer/tautomer leakage (subtle molecular variants)
    6. Physicochemical property distributions (bias detection)
    7. Activity/label distributions (target variable bias)
    8. Functional group distributions (chemical space coverage)
    
    Produces both a detailed JSON report and a human-readable text report with
    severity levels (OK, LOW, MEDIUM, HIGH, CRITICAL) for each issue found. 
    
    Parameters
    ----------
    train_path : str
        Training split filename.
    test_path : str
        Test split filename.
    val_path : Optional[str]
        Validation split filename (if exists).
    project_manifest_path : str
        Path to manifest.json.
    smiles_col : str
        SMILES column name.
    label_col : str
        Label column name.
    output_filename : str
        Output filename prefix for text report.
    explanation : str
        Description for manifest.
    min_split_size : int, default=50
        Minimum acceptable split size.
    imbalance_threshold : float, default=0.1
        Class imbalance threshold.
    similarity_threshold : float, default=0.9
        Tanimoto similarity threshold for leakage.
    activity_cliff_similarity : float, default=0.8
        Similarity threshold for activity cliffs.
    activity_cliff_fold_diff : float, default=10.0
        Fold difference for regression activity cliffs.
    alpha : float, default=0.05
        Significance level for statistical tests.
    min_occurrence_threshold : int, default=2
        Minimum functional group occurrences to flag.
    max_examples : int, default=10
        Maximum examples per issue.
        
    Returns
    -------
    Dict
        Contains output_filename, json_report_filename, n_lines, overall_severity,
        report_sections, issues_found.
    """
    from molml_mcp.infrastructure.resources import _load_resource, _store_resource
    
    # Step 1: Run comprehensive quality analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA SPLITTING QUALITY ANALYSIS")
    print("="*80)
    
    json_result = analyze_split_quality(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        project_manifest_path=project_manifest_path,
        smiles_col=smiles_col,
        label_col=label_col,
        output_filename=f"{output_filename}_json",
        explanation=f"JSON analysis for {explanation}",
        min_split_size=min_split_size,
        imbalance_threshold=imbalance_threshold,
        similarity_threshold=similarity_threshold,
        activity_cliff_similarity=activity_cliff_similarity,
        activity_cliff_fold_diff=activity_cliff_fold_diff,
        alpha=alpha,
        min_occurrence_threshold=min_occurrence_threshold,
        max_examples=max_examples
    )
    
    json_report_filename = json_result['output_filename']
    
    print(f"\n✓ Quality analysis complete")
    print(f"  Overall Severity: {json_result['overall_severity']}")
    print(f"  JSON Report: {json_report_filename}")
    
    # Step 2: Load the JSON report and generate text report
    print(f"\nGenerating human-readable text report...")
    
    # Load JSON report
    report = _load_resource(project_manifest_path, json_report_filename)
    
    # Start building text report
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("DATA SPLITTING QUALITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Metadata
    meta = report.get('metadata', {})
    lines.append(f"Analysis Date: {meta.get('timestamp', 'N/A')}")
    lines.append(f"Execution Time: {meta.get('execution_time_seconds', 0):.2f}s")
    lines.append("")
    lines.append(f"Train File: {meta.get('train_file', 'N/A')}")
    lines.append(f"Test File: {meta.get('test_file', 'N/A')}")
    if meta.get('val_file'):
        lines.append(f"Val File: {meta.get('val_file')}")
    lines.append("")
    lines.append(f"SMILES Column: {meta.get('smiles_column', 'N/A')}")
    lines.append(f"Label Column: {meta.get('label_column', 'N/A')}")
    lines.append("")
    
    # Overall Summary
    overall_severity = report.get('overall_severity', 'UNKNOWN')
    severity_summary = report.get('severity_summary', {})
    
    lines.append("=" * 80)
    lines.append("OVERALL ASSESSMENT")
    lines.append("=" * 80)
    lines.append("")
    
    # Severity icon
    severity_icons = {
        'OK': '✓',
        'LOW': '⚠',
        'MEDIUM': '⚠⚠',
        'HIGH': '⚠⚠⚠',
        'CRITICAL': '🔴'
    }
    icon = severity_icons.get(overall_severity, '?')
    lines.append(f"Overall Severity: {icon} {overall_severity}")
    lines.append("")
    lines.append("Severity Breakdown:")
    lines.append(f"  - CRITICAL: {severity_summary.get('CRITICAL', 0)} checks")
    lines.append(f"  - HIGH:     {severity_summary.get('HIGH', 0)} checks")
    lines.append(f"  - MEDIUM:   {severity_summary.get('MEDIUM', 0)} checks")
    lines.append(f"  - LOW:      {severity_summary.get('LOW', 0)} checks")
    lines.append(f"  - OK:       {severity_summary.get('OK', 0)} checks")
    lines.append("")
    
    # 1. Split Characteristics
    lines.append("=" * 80)
    lines.append("1. SPLIT CHARACTERISTICS")
    lines.append("=" * 80)
    lines.append("")
    
    split_char = report.get('split_characteristics', {})
    if 'error' in split_char:
        lines.append(f"Error: {split_char['error']}")
    else:
        train_size = split_char.get('train_size', 'N/A')
        test_size = split_char.get('test_size', 'N/A')
        val_size = split_char.get('val_size')
        
        lines.append(f"Train Size: {train_size} molecules" if train_size != 'N/A' else f"Train Size: {train_size}")
        lines.append(f"Test Size:  {test_size} molecules" if test_size != 'N/A' else f"Test Size:  {test_size}")
        if val_size:
            lines.append(f"Val Size:   {val_size} molecules" if val_size != 'N/A' else f"Val Size:   {val_size}")
        lines.append("")
        
        split_ratios = split_char.get('split_ratios')
        if split_ratios and split_ratios.get('train_pct'):
            ratios = split_char['split_ratios']
            lines.append("Split Ratios:")
            lines.append(f"  - Train: {ratios.get('train_pct', 0):.1f}%")
            lines.append(f"  - Test:  {ratios.get('test_pct', 0):.1f}%")
            if ratios.get('val_pct'):
                lines.append(f"  - Val:   {ratios.get('val_pct', 0):.1f}%")
            lines.append("")
        
        if split_char.get('task_type') == 'classification':
            n_classes = split_char.get('n_classes', '?')
            lines.append(f"Task Type: Classification ({n_classes} classes)")
            lines.append("")
            
            # Train class distribution
            if split_char.get('train_class_distribution'):
                lines.append("Train Class Distribution:")
                for cls, count in split_char['train_class_distribution'].items():
                    pct = split_char.get('train_class_percentages', {}).get(cls, 0)
                    lines.append(f"  - Class {cls}: {count} ({pct:.1f}%)")
                lines.append("")
            
            # Test class distribution
            if split_char.get('test_class_distribution'):
                lines.append("Test Class Distribution:")
                for cls, count in split_char['test_class_distribution'].items():
                    pct = split_char.get('test_class_percentages', {}).get(cls, 0)
                    lines.append(f"  - Class {cls}: {count} ({pct:.1f}%)")
                lines.append("")
        else:
            lines.append(f"Task Type: Regression")
            lines.append("")
    
    lines.append(f"Overall Severity: {split_char.get('severity', 'UNKNOWN')}")
    lines.append("")
    
    # 2. Exact Duplicates
    lines.append("=" * 80)
    lines.append("2. EXACT DUPLICATE DETECTION")
    lines.append("=" * 80)
    lines.append("")
    
    exact_dups = report.get('exact_duplicates', {})
    total_dups = exact_dups.get('total_duplicate_molecules', 0)
    
    if total_dups > 0:
        lines.append(f"🔴 CRITICAL: {total_dups} duplicate molecule(s) found across splits!")
        lines.append("")
        lines.append("This is a CRITICAL data leakage issue that will inflate performance metrics.")
        lines.append("")
        
        # Train/Test duplicates
        train_test = exact_dups.get('train_test_duplicates', {})
        if train_test.get('n_duplicates', 0) > 0:
            lines.append(f"Train/Test: {train_test['n_duplicates']} duplicates")
            if train_test.get('examples'):
                lines.append("  Examples:")
                for ex in train_test['examples'][:3]:
                    lines.append(f"    - {ex['smiles']}")
            lines.append("")
        
        # Train/Val duplicates
        if exact_dups.get('train_val_duplicates'):
            train_val = exact_dups['train_val_duplicates']
            if train_val.get('n_duplicates', 0) > 0:
                lines.append(f"Train/Val: {train_val['n_duplicates']} duplicates")
                lines.append("")
    else:
        lines.append("✓ No exact duplicates found between splits")
        lines.append("")
    
    lines.append(f"Overall Severity: {exact_dups.get('severity', 'UNKNOWN')}")
    lines.append("")
    
    # 3. Similarity Leakage
    lines.append("=" * 80)
    lines.append("3. SIMILARITY-BASED LEAKAGE")
    lines.append("=" * 80)
    lines.append("")
    
    sim_leak = report.get('similarity_leakage', {})
    lines.append(f"Similarity Threshold: {sim_leak.get('similarity_threshold', 0.9)}")
    lines.append(f"Activity Cliff Threshold: {sim_leak.get('activity_cliff_similarity_threshold', 0.8)}")
    lines.append("")
    
    # Test vs Train
    test_train = sim_leak.get('test_vs_train', {})
    if test_train:
        lines.append("Test vs Train:")
        lines.append(f"  - High similarity pairs: {test_train.get('n_high_similarity', 0)}")
        lines.append(f"  - Activity cliffs: {test_train.get('n_activity_cliffs', 0)}")
        if test_train.get('similarity_stats'):
            stats = test_train['similarity_stats']
            lines.append(f"  - Avg max similarity: {stats.get('mean', 0):.3f}")
            lines.append(f"  - Max similarity: {stats.get('max', 0):.3f}")
        lines.append("")
    
    lines.append(f"Overall Severity: {sim_leak.get('overall_severity', 'UNKNOWN')}")
    lines.append("")
    
    # 4. Scaffold Leakage
    lines.append("=" * 80)
    lines.append("4. SCAFFOLD OVERLAP")
    lines.append("=" * 80)
    lines.append("")
    
    scaffold = report.get('scaffold_leakage', {})
    train_test_overlap = scaffold.get('train_test_overlap', {})
    
    if train_test_overlap:
        n_shared = train_test_overlap.get('n_shared_scaffolds', 0)
        pct_test = train_test_overlap.get('pct_split2_in_split1', 0)
        
        lines.append(f"Train/Test Overlap:")
        lines.append(f"  - Shared scaffolds: {n_shared}")
        lines.append(f"  - % of test scaffolds in train: {pct_test:.1f}%")
        lines.append("")
        
        if n_shared > 0 and train_test_overlap.get('examples'):
            lines.append("  Top shared scaffolds:")
            for ex in train_test_overlap['examples'][:3]:
                lines.append(f"    - {ex['scaffold_smiles'][:50]}...")
                lines.append(f"      Train: {ex['n_molecules_train']} mols, Test: {ex['n_molecules_test']} mols")
            lines.append("")
    
    lines.append(f"Overall Severity: {scaffold.get('overall_severity', 'UNKNOWN')}")
    lines.append("")
    
    # 5. Stereoisomer/Tautomer Leakage
    lines.append("=" * 80)
    lines.append("5. STEREOISOMER & TAUTOMER LEAKAGE")
    lines.append("=" * 80)
    lines.append("")
    
    stereo_taut = report.get('stereoisomer_tautomer_leakage', {})
    n_stereo = stereo_taut.get('total_stereoisomer_pairs', 0)
    n_taut = stereo_taut.get('total_tautomer_pairs', 0)
    
    lines.append(f"Total Stereoisomer Pairs: {n_stereo}")
    lines.append(f"Total Tautomer Pairs: {n_taut}")
    lines.append("")
    
    if n_stereo > 0 or n_taut > 0:
        lines.append("These represent subtle data leakage that can inflate model performance.")
        lines.append("")
        
        # Train/Test details
        train_test_stereo = stereo_taut.get('train_test_stereoisomers', {})
        train_test_taut = stereo_taut.get('train_test_tautomers', {})
        
        if train_test_stereo.get('n_pairs', 0) > 0:
            lines.append(f"Train/Test Stereoisomers: {train_test_stereo['n_pairs']} pairs")
        if train_test_taut.get('n_pairs', 0) > 0:
            lines.append(f"Train/Test Tautomers: {train_test_taut['n_pairs']} pairs")
        lines.append("")
    else:
        lines.append("✓ No stereoisomer or tautomer leakage detected")
        lines.append("")
    
    lines.append(f"Overall Severity: {stereo_taut.get('overall_severity', 'UNKNOWN')}")
    lines.append("")
    
    # 6. Property Distributions
    lines.append("=" * 80)
    lines.append("6. PHYSICOCHEMICAL PROPERTY DISTRIBUTIONS")
    lines.append("=" * 80)
    lines.append("")
    
    prop_dist = report.get('property_distributions', {})
    lines.append(f"Statistical Test: Kolmogorov-Smirnov (α={prop_dist.get('alpha', 0.05)})")
    lines.append(f"Properties Tested: {prop_dist.get('summary', {}).get('n_properties_tested', 8)}")
    lines.append("")
    
    n_sig = prop_dist.get('summary', {}).get('n_significant_train_test', 0)
    if n_sig > 0:
        lines.append(f"⚠ {n_sig} properties show significant differences between train/test")
        lines.append("")
        
        # Show significant properties
        train_test = prop_dist.get('train_vs_test', {})
        if train_test:
            lines.append("Significant differences:")
            for prop, result in train_test.items():
                if isinstance(result, dict) and result.get('significant'):
                    train_mean = result.get('train_mean', 0)
                    test_mean = result.get('test_mean', 0)
                    p_val = result.get('p_value', 0)
                    lines.append(f"  - {prop}: train={train_mean:.2f}, test={test_mean:.2f} (p={p_val:.4f})")
            lines.append("")
    else:
        lines.append("✓ No significant property distribution differences detected")
        lines.append("")
    
    lines.append(f"Overall Severity: {prop_dist.get('overall_severity', 'UNKNOWN')}")
    lines.append("")
    
    # 7. Activity Distributions
    lines.append("=" * 80)
    lines.append("7. ACTIVITY/LABEL DISTRIBUTIONS")
    lines.append("=" * 80)
    lines.append("")
    
    act_dist = report.get('activity_distributions', {})
    task_type = act_dist.get('task_type', 'unknown')
    lines.append(f"Task Type: {task_type.capitalize()}")
    lines.append("")
    
    train_test = act_dist.get('train_vs_test', {})
    if train_test and not train_test.get('error'):
        if task_type == 'classification':
            lines.append(f"Chi-square test: p={train_test.get('p_value', 0):.4f}")
            lines.append(f"Interpretation: {train_test.get('interpretation', 'UNKNOWN')}")
            lines.append("")
            
            # Class balance
            if act_dist.get('class_balance'):
                lines.append("Class Balance:")
                for split_name, balance in act_dist['class_balance'].items():
                    if not balance.get('error'):
                        imbalanced = balance.get('imbalanced', False)
                        status = "⚠ Imbalanced" if imbalanced else "✓ Balanced"
                        ratio = balance.get('min_class_ratio', 0)
                        lines.append(f"  - {split_name.capitalize()}: {status} (ratio={ratio:.3f})")
                lines.append("")
        else:
            lines.append(f"KS test: p={train_test.get('p_value', 0):.4f}")
            lines.append(f"Interpretation: {train_test.get('interpretation', 'UNKNOWN')}")
            lines.append("")
            lines.append(f"Train: mean={train_test.get('train_mean', 0):.3f}, std={train_test.get('train_std', 0):.3f}")
            lines.append(f"Test:  mean={train_test.get('test_mean', 0):.3f}, std={train_test.get('test_std', 0):.3f}")
            lines.append("")
    
    lines.append(f"Overall Severity: {act_dist.get('overall_severity', 'UNKNOWN')}")
    lines.append("")
    
    # 8. Functional Groups
    lines.append("=" * 80)
    lines.append("8. FUNCTIONAL GROUP DISTRIBUTION")
    lines.append("=" * 80)
    lines.append("")
    
    func_groups = report.get('functional_groups', {})
    summary = func_groups.get('summary', {})
    
    lines.append(f"Functional Groups Tested: {func_groups.get('n_functional_groups_tested', 19)}")
    lines.append(f"Shared Across Splits: {summary.get('n_shared', 0)} ({summary.get('pct_shared', 0):.1f}%)")
    lines.append("")
    
    n_unique_test = summary.get('n_unique_to_test', 0)
    n_unique_train = summary.get('n_unique_to_train', 0)
    
    if n_unique_test > 0 or n_unique_train > 0:
        lines.append("Unique Groups:")
        if n_unique_train > 0:
            lines.append(f"  - Train only: {n_unique_train} groups")
            unique_train = func_groups.get('unique_to_train', [])
            for group_info in unique_train[:3]:
                lines.append(f"    * {group_info['group']}: {group_info['count']} molecules ({group_info['pct_molecules']:.1f}%)")
        
        if n_unique_test > 0:
            lines.append(f"  - Test only: {n_unique_test} groups")
            unique_test = func_groups.get('unique_to_test', [])
            for group_info in unique_test[:3]:
                lines.append(f"    * {group_info['group']}: {group_info['count']} molecules ({group_info['pct_molecules']:.1f}%)")
        lines.append("")
    else:
        lines.append("✓ No functional groups unique to a single split")
        lines.append("")
    
    lines.append(f"Overall Severity: {func_groups.get('overall_severity', 'UNKNOWN')}")
    lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Join lines into text
    report_text = "\n".join(lines)
    
    # Save as text resource
    output_file = _store_resource(
        report_text,
        project_manifest_path,
        output_filename,
        explanation,
        'txt'
    )
    
    print(f"✓ Text report generated: {output_file}")
    print(f"  Lines: {len(lines)}")
    print(f"  Sections: {len(['metadata', 'overall_assessment', 'split_characteristics', 'exact_duplicates', 'similarity_leakage', 'scaffold_overlap', 'stereoisomer_tautomer', 'property_distributions', 'activity_distributions', 'functional_groups'])}")
    print("="*80 + "\n")
    
    return {
        'output_filename': output_file,
        'json_report_filename': json_report_filename,
        'n_lines': len(lines),
        'overall_severity': overall_severity,
        'report_sections': [
            'metadata',
            'overall_assessment',
            'split_characteristics',
            'exact_duplicates',
            'similarity_leakage',
            'scaffold_overlap',
            'stereoisomer_tautomer',
            'property_distributions',
            'activity_distributions',
            'functional_groups'
        ],
        'issues_found': json_result['issues_found']
    }
