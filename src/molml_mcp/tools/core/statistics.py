"""
Statistical tests for dataset analysis.

All tests include appropriate effect size metrics:

Normality Tests:
- Shapiro-Wilk test: checks if data is normally distributed (best for small to medium samples)
- Kolmogorov-Smirnov test: alternative normality test (good for larger samples)
- Anderson-Darling test: another normality test (more sensitive to tails)

Paired Comparison Tests:
- Paired t-test: compares means of two related samples (assumes normality)
- Wilcoxon signed-rank test: non-parametric alternative to paired t-test

Correlation Tests:
- Pearson correlation: measures linear correlation (assumes normality)
- Spearman correlation: measures monotonic correlation (rank-based, non-parametric)

Independent Sample Tests:
- Independent t-test (Welch's): compares means of two independent samples [Cohen's d effect size]
- Mann-Whitney U test: non-parametric alternative to independent t-test [Cliff's delta effect size]
- Kolmogorov-Smirnov test: compares distributions of two independent samples

Multi-Group Tests:
- One-way ANOVA: compares means across multiple groups (assumes normality) [eta-squared effect size]
- Kruskal-Wallis test: non-parametric alternative to one-way ANOVA [epsilon-squared effect size]

Categorical Tests:
- Chi-square test of independence: tests association between categorical variables [Cramér's V effect size]
- Fisher's exact test: exact test for 2x2 tables (small samples) [odds ratio effect size]
- McNemar's test: tests for changes in paired categorical data [odds ratio effect size]
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from molml_mcp.infrastructure.resources import _load_resource


def test_shapiro_wilk(
    input_filename: str,
    project_manifest_path: str,
    column: str,
    alpha: float = 0.05
) -> Dict:
    """
    Shapiro-Wilk test for normality (best for n < 5000).
    H0: data is normally distributed.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        column: Column to test
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistic, p_value, alpha, is_normal, n_samples, interpretation, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get data and remove NaN values
    data = df[column].dropna()
    n_samples = len(data)
    
    if n_samples == 0:
        raise ValueError(f"No valid (non-NaN) data in column '{column}'")
    
    if n_samples < 3:
        raise ValueError(
            f"Shapiro-Wilk test requires at least 3 samples. "
            f"Found: {n_samples}"
        )
    
    # Perform Shapiro-Wilk test
    statistic, p_value = stats.shapiro(data)
    
    # Interpret result
    is_normal = p_value > alpha
    
    if is_normal:
        interpretation = (
            f"Data appears normally distributed (p={p_value:.4f} > α={alpha}). "
            f"Fail to reject null hypothesis."
        )
    else:
        interpretation = (
            f"Data does NOT appear normally distributed (p={p_value:.4f} ≤ α={alpha}). "
            f"Reject null hypothesis."
        )
    
    return {
        "test": "Shapiro-Wilk",
        "column": column,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_normal": is_normal,
        "n_samples": n_samples,
        "interpretation": interpretation,
        "summary": f"Shapiro-Wilk test: W={statistic:.4f}, p={p_value:.4f}, normal={is_normal}"
    }


def test_kolmogorov_smirnov_norm(
    input_filename: str,
    project_manifest_path: str,
    column: str,
    alpha: float = 0.05
) -> Dict:
    """
    K-S test for normality (compares empirical distribution to normal).
    H0: data follows normal distribution.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        column: Column to test
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistic, p_value, alpha, is_normal, n_samples, mean, std, interpretation, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get data and remove NaN values
    data = df[column].dropna()
    n_samples = len(data)
    
    if n_samples == 0:
        raise ValueError(f"No valid (non-NaN) data in column '{column}'")
    
    # Calculate mean and std for the normal distribution
    mean = data.mean()
    std = data.std()
    
    if std == 0:
        raise ValueError(f"Standard deviation is zero for column '{column}'")
    
    # Perform K-S test comparing data to normal distribution
    statistic, p_value = stats.kstest(data, 'norm', args=(mean, std))
    
    # Interpret result
    is_normal = p_value > alpha
    
    if is_normal:
        interpretation = (
            f"Data appears normally distributed (p={p_value:.4f} > α={alpha}). "
            f"Fail to reject null hypothesis."
        )
    else:
        interpretation = (
            f"Data does NOT appear normally distributed (p={p_value:.4f} ≤ α={alpha}). "
            f"Reject null hypothesis."
        )
    
    return {
        "test": "Kolmogorov-Smirnov",
        "column": column,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_normal": is_normal,
        "n_samples": n_samples,
        "mean": float(mean),
        "std": float(std),
        "interpretation": interpretation,
        "summary": f"K-S test: D={statistic:.4f}, p={p_value:.4f}, normal={is_normal}"
    }


def test_anderson_darling(
    input_filename: str,
    project_manifest_path: str,
    column: str,
    significance_level: str = "5%"
) -> Dict:
    """
    Anderson-Darling test for normality (more sensitive to tail deviations).
    H0: data follows normal distribution.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        column: Column to test
        significance_level: One of "15%", "10%", "5%", "2.5%", "1%" (default: "5%")
        
    Returns:
        Dictionary with statistic, critical_values, significance_levels, selected_alpha, 
        critical_value, is_normal, n_samples, interpretation, summary
    """
    # Validate significance level
    valid_levels = ["15%", "10%", "5%", "2.5%", "1%"]
    if significance_level not in valid_levels:
        raise ValueError(
            f"significance_level must be one of {valid_levels}. "
            f"Got: {significance_level}"
        )
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get data and remove NaN values
    data = df[column].dropna()
    n_samples = len(data)
    
    if n_samples == 0:
        raise ValueError(f"No valid (non-NaN) data in column '{column}'")
    
    # Perform Anderson-Darling test
    result = stats.anderson(data, dist='norm')
    
    statistic = result.statistic
    critical_values = result.critical_values
    significance_levels = result.significance_level
    
    # Map significance level to index
    level_map = {"15%": 0, "10%": 1, "5%": 2, "2.5%": 3, "1%": 4}
    idx = level_map[significance_level]
    
    critical_value = critical_values[idx]
    is_normal = statistic < critical_value
    
    # Create critical values dict
    critical_values_dict = {
        f"{int(sig)}%": float(cv) 
        for sig, cv in zip(significance_levels, critical_values)
    }
    
    if is_normal:
        interpretation = (
            f"Data appears normally distributed "
            f"(statistic={statistic:.4f} < critical_value={critical_value:.4f} at α={significance_level}). "
            f"Fail to reject null hypothesis."
        )
    else:
        interpretation = (
            f"Data does NOT appear normally distributed "
            f"(statistic={statistic:.4f} ≥ critical_value={critical_value:.4f} at α={significance_level}). "
            f"Reject null hypothesis."
        )
    
    return {
        "test": "Anderson-Darling",
        "column": column,
        "statistic": float(statistic),
        "critical_values": critical_values_dict,
        "significance_levels": [f"{int(s)}%" for s in significance_levels],
        "selected_alpha": significance_level,
        "critical_value": float(critical_value),
        "is_normal": is_normal,
        "n_samples": n_samples,
        "interpretation": interpretation,
        "summary": f"Anderson-Darling test: A²={statistic:.4f}, critical={critical_value:.4f}, normal={is_normal}"
    }


def test_paired_ttest(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Paired t-test for related samples (assumes normal distribution of differences).
    H0: mean difference between pairs is zero.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        alternative: "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary with statistic, p_value, alpha, alternative, is_significant, n_pairs,
        mean_diff, std_diff, interpretation, summary
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length for paired test. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_pairs = len(data_a_clean)
    
    if n_pairs == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    if n_pairs < 2:
        raise ValueError(f"Paired t-test requires at least 2 pairs. Found: {n_pairs}")
    
    # Calculate differences
    differences = data_a_clean - data_b_clean
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(data_a_clean, data_b_clean, alternative=alternative)
    
    # Interpret result
    is_significant = p_value <= alpha
    
    if alternative == "two-sided":
        if is_significant:
            direction = "greater" if mean_diff > 0 else "less"
            interpretation = (
                f"Significant difference detected (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean difference: {mean_diff:.4f}. Dataset A is {direction} than Dataset B."
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f} > α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Dataset A is significantly GREATER than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly greater than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Dataset A is significantly LESS than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly less than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
    
    return {
        "test": "Paired t-test",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "is_significant": bool(is_significant),
        "n_pairs": int(n_pairs),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "interpretation": interpretation,
        "summary": f"Paired t-test: t={statistic:.4f}, p={p_value:.4f}, significant={is_significant}"
    }


def test_wilcoxon_signed_rank(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Wilcoxon signed-rank test for related samples (non-parametric alternative to paired t-test).
    H0: median difference between pairs is zero.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        alternative: "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary with statistic, p_value, alpha, alternative, is_significant, n_pairs,
        median_diff, n_positive, n_negative, n_zero, interpretation, summary
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length for paired test. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_pairs = len(data_a_clean)
    
    if n_pairs == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    # Calculate differences
    differences = data_a_clean - data_b_clean
    median_diff = float(np.median(differences))
    n_positive = int(np.sum(differences > 0))
    n_negative = int(np.sum(differences < 0))
    n_zero = int(np.sum(differences == 0))
    
    # Perform Wilcoxon signed-rank test
    result = stats.wilcoxon(data_a_clean, data_b_clean, alternative=alternative)
    statistic = result.statistic
    p_value = result.pvalue
    
    # Interpret result
    is_significant = p_value <= alpha
    
    if alternative == "two-sided":
        if is_significant:
            direction = "greater" if median_diff > 0 else "less"
            interpretation = (
                f"Significant difference detected (p={p_value:.4f} ≤ α={alpha}). "
                f"Median difference: {median_diff:.4f}. Dataset A is {direction} than Dataset B."
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f} > α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Dataset A is significantly GREATER than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly greater than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Dataset A is significantly LESS than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly less than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
    
    return {
        "test": "Wilcoxon signed-rank",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "is_significant": bool(is_significant),
        "n_pairs": int(n_pairs),
        "median_diff": float(median_diff),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "n_zero": int(n_zero),
        "interpretation": interpretation,
        "summary": f"Wilcoxon test: W={statistic:.4f}, p={p_value:.4f}, significant={is_significant}"
    }


def test_pearson_correlation(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05
) -> Dict:
    """
    Pearson correlation for linear relationship (assumes normality). r ∈ [-1, 1].
    H0: no linear correlation (r = 0). Strength: |r| < 0.3 weak, 0.3-0.7 moderate, ≥0.7 strong.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with correlation, p_value, alpha, is_significant, n_samples,
        strength, direction, interpretation, summary
    """
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_samples = len(data_a_clean)
    
    if n_samples == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    if n_samples < 3:
        raise ValueError(f"Correlation requires at least 3 samples. Found: {n_samples}")
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(data_a_clean, data_b_clean)
    
    # Interpret correlation strength (using common thresholds)
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if correlation > 0 else "negative"
    
    # Interpret significance
    is_significant = p_value <= alpha
    
    if is_significant:
        interpretation = (
            f"Significant {strength} {direction} correlation detected "
            f"(r={correlation:.4f}, p={p_value:.4f} ≤ α={alpha}). "
            f"Variables show a linear relationship."
        )
    else:
        interpretation = (
            f"No significant correlation (r={correlation:.4f}, p={p_value:.4f} > α={alpha}). "
            f"Variables do not show a significant linear relationship."
        )
    
    return {
        "test": "Pearson correlation",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "is_significant": bool(is_significant),
        "n_samples": int(n_samples),
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "summary": f"Pearson r={correlation:.4f}, p={p_value:.4f}, {strength} {direction}, significant={is_significant}"
    }


def test_spearman_correlation(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05
) -> Dict:
    """
    Spearman rank correlation for monotonic relationship (non-parametric). ρ ∈ [-1, 1].
    H0: no monotonic correlation (ρ = 0). Strength: |ρ| < 0.3 weak, 0.3-0.7 moderate, ≥0.7 strong.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with correlation, p_value, alpha, is_significant, n_samples,
        strength, direction, interpretation, summary
    """
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_samples = len(data_a_clean)
    
    if n_samples == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    if n_samples < 3:
        raise ValueError(f"Correlation requires at least 3 samples. Found: {n_samples}")
    
    # Calculate Spearman correlation
    correlation, p_value = stats.spearmanr(data_a_clean, data_b_clean)
    
    # Interpret correlation strength (using common thresholds)
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if correlation > 0 else "negative"
    
    # Interpret significance
    is_significant = p_value <= alpha
    
    if is_significant:
        interpretation = (
            f"Significant {strength} {direction} correlation detected "
            f"(ρ={correlation:.4f}, p={p_value:.4f} ≤ α={alpha}). "
            f"Variables show a monotonic relationship."
        )
    else:
        interpretation = (
            f"No significant correlation (ρ={correlation:.4f}, p={p_value:.4f} > α={alpha}). "
            f"Variables do not show a significant monotonic relationship."
        )
    
    return {
        "test": "Spearman correlation",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "is_significant": bool(is_significant),
        "n_samples": int(n_samples),
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "summary": f"Spearman ρ={correlation:.4f}, p={p_value:.4f}, {strength} {direction}, significant={is_significant}"
    }


def test_independent_ttest(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Welch's t-test for independent samples (assumes normality, not equal variances).
    H0: groups have equal means. Effect size: Cohen's d.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        alternative: "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary with statistic, p_value, alpha, alternative, is_significant, n_a, n_b,
        mean_a, mean_b, std_a, std_b, cohens_d, effect_size, interpretation, summary
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data and remove NaN
    data_a = df_a[column_a].dropna().values
    data_b = df_b[column_b].dropna().values
    
    n_a = len(data_a)
    n_b = len(data_b)
    
    if n_a < 2 or n_b < 2:
        raise ValueError(f"Each group needs at least 2 samples. Group A: {n_a}, Group B: {n_b}")
    
    # Calculate statistics
    mean_a = float(np.mean(data_a))
    mean_b = float(np.mean(data_b))
    std_a = float(np.std(data_a, ddof=1))
    std_b = float(np.std(data_b, ddof=1))
    
    # Calculate Cohen's d (pooled standard deviation version)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
    
    # Interpret Cohen's d
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_size_interp = "negligible"
    elif abs_d < 0.5:
        effect_size_interp = "small"
    elif abs_d < 0.8:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Perform Welch's t-test (equal_var=False)
    statistic, p_value = stats.ttest_ind(data_a, data_b, equal_var=False, alternative=alternative)
    
    # Interpret result
    is_significant = p_value <= alpha
    mean_diff = mean_a - mean_b
    
    if alternative == "two-sided":
        if is_significant:
            direction = "greater" if mean_diff > 0 else "less"
            interpretation = (
                f"Significant difference detected (p={p_value:.4f} ≤ α={alpha}). "
                f"Group A mean ({mean_a:.4f}) is {direction} than Group B mean ({mean_b:.4f})."
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f} > α={alpha}). "
                f"Group A mean: {mean_a:.4f}, Group B mean: {mean_b:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Group A is significantly GREATER than Group B (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean A: {mean_a:.4f}, Mean B: {mean_b:.4f}."
            )
        else:
            interpretation = (
                f"Group A is NOT significantly greater than Group B (p={p_value:.4f} > α={alpha}). "
                f"Mean A: {mean_a:.4f}, Mean B: {mean_b:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Group A is significantly LESS than Group B (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean A: {mean_a:.4f}, Mean B: {mean_b:.4f}."
            )
        else:
            interpretation = (
                f"Group A is NOT significantly less than Group B (p={p_value:.4f} > α={alpha}). "
                f"Mean A: {mean_a:.4f}, Mean B: {mean_b:.4f}."
            )
    
    return {
        "test": "Independent t-test (Welch's)",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "is_significant": bool(is_significant),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "std_a": float(std_a),
        "std_b": float(std_b),
        "cohens_d": float(cohens_d),
        "effect_size": effect_size_interp,
        "interpretation": interpretation,
        "summary": f"Welch's t-test: t={statistic:.4f}, p={p_value:.4f}, d={cohens_d:.4f} ({effect_size_interp}), significant={is_significant}"
    }


def test_mann_whitney_u(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Mann-Whitney U test for independent samples (non-parametric). Effect size: Cliff's delta.
    H0: samples from same distribution.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        alternative: "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary with statistic, p_value, alpha, alternative, is_significant, n_a, n_b,
        median_a, median_b, cliffs_delta, effect_size, interpretation, summary
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data and remove NaN
    data_a = df_a[column_a].dropna().values
    data_b = df_b[column_b].dropna().values
    
    n_a = len(data_a)
    n_b = len(data_b)
    
    if n_a < 1 or n_b < 1:
        raise ValueError(f"Each group needs at least 1 sample. Group A: {n_a}, Group B: {n_b}")
    
    # Calculate medians
    median_a = float(np.median(data_a))
    median_b = float(np.median(data_b))
    
    # Calculate Cliff's delta (non-parametric effect size)
    # Counts pairs where A > B minus pairs where A < B, divided by total pairs
    n_greater = sum(a > b for a in data_a for b in data_b)
    n_less = sum(a < b for a in data_a for b in data_b)
    cliffs_delta = (n_greater - n_less) / (n_a * n_b)
    
    # Interpret Cliff's delta
    abs_delta = abs(cliffs_delta)
    if abs_delta < 0.147:
        effect_size_interp = "negligible"
    elif abs_delta < 0.33:
        effect_size_interp = "small"
    elif abs_delta < 0.474:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Perform Mann-Whitney U test
    result = stats.mannwhitneyu(data_a, data_b, alternative=alternative)
    statistic = result.statistic
    p_value = result.pvalue
    
    # Interpret result
    is_significant = p_value <= alpha
    median_diff = median_a - median_b
    
    if alternative == "two-sided":
        if is_significant:
            direction = "greater" if median_diff > 0 else "less"
            interpretation = (
                f"Significant difference detected (p={p_value:.4f} ≤ α={alpha}). "
                f"Group A median ({median_a:.4f}) is {direction} than Group B median ({median_b:.4f})."
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f} > α={alpha}). "
                f"Group A median: {median_a:.4f}, Group B median: {median_b:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Group A is significantly GREATER than Group B (p={p_value:.4f} ≤ α={alpha}). "
                f"Median A: {median_a:.4f}, Median B: {median_b:.4f}."
            )
        else:
            interpretation = (
                f"Group A is NOT significantly greater than Group B (p={p_value:.4f} > α={alpha}). "
                f"Median A: {median_a:.4f}, Median B: {median_b:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Group A is significantly LESS than Group B (p={p_value:.4f} ≤ α={alpha}). "
                f"Median A: {median_a:.4f}, Median B: {median_b:.4f}."
            )
        else:
            interpretation = (
                f"Group A is NOT significantly less than Group B (p={p_value:.4f} > α={alpha}). "
                f"Median A: {median_a:.4f}, Median B: {median_b:.4f}."
            )
    
    return {
        "test": "Mann-Whitney U",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "is_significant": bool(is_significant),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "median_a": float(median_a),
        "median_b": float(median_b),
        "cliffs_delta": float(cliffs_delta),
        "effect_size": effect_size_interp,
        "interpretation": interpretation,
        "summary": f"Mann-Whitney U={statistic:.4f}, p={p_value:.4f}, δ={cliffs_delta:.4f} ({effect_size_interp}), significant={is_significant}"
    }


def test_kolmogorov_smirnov_two_sample(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Two-sample K-S test comparing distributions (non-parametric).
    H0: samples from same distribution.
    
    Args:
        input_filename_a: First CSV dataset filename
        input_filename_b: Second CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: Column in dataset A
        column_b: Column in dataset B
        alpha: Significance level (default: 0.05)
        alternative: "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary with statistic, p_value, alpha, alternative, is_significant, n_a, n_b,
        interpretation, summary
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data and remove NaN
    data_a = df_a[column_a].dropna().values
    data_b = df_b[column_b].dropna().values
    
    n_a = len(data_a)
    n_b = len(data_b)
    
    if n_a < 1 or n_b < 1:
        raise ValueError(f"Each group needs at least 1 sample. Group A: {n_a}, Group B: {n_b}")
    
    # Perform two-sample K-S test
    statistic, p_value = stats.ks_2samp(data_a, data_b, alternative=alternative)
    
    # Interpret result
    is_significant = p_value <= alpha
    
    if alternative == "two-sided":
        if is_significant:
            interpretation = (
                f"Distributions are significantly different (p={p_value:.4f} ≤ α={alpha}). "
                f"K-S statistic: {statistic:.4f}."
            )
        else:
            interpretation = (
                f"No significant difference between distributions (p={p_value:.4f} > α={alpha}). "
                f"K-S statistic: {statistic:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Distribution A is stochastically GREATER than B (p={p_value:.4f} ≤ α={alpha}). "
                f"K-S statistic: {statistic:.4f}."
            )
        else:
            interpretation = (
                f"Distribution A is NOT stochastically greater than B (p={p_value:.4f} > α={alpha}). "
                f"K-S statistic: {statistic:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Distribution A is stochastically LESS than B (p={p_value:.4f} ≤ α={alpha}). "
                f"K-S statistic: {statistic:.4f}."
            )
        else:
            interpretation = (
                f"Distribution A is NOT stochastically less than B (p={p_value:.4f} > α={alpha}). "
                f"K-S statistic: {statistic:.4f}."
            )
    
    return {
        "test": "Two-sample Kolmogorov-Smirnov",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "is_significant": bool(is_significant),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "interpretation": interpretation,
        "summary": f"K-S test: D={statistic:.4f}, p={p_value:.4f}, significant={is_significant}"
    }


def test_one_way_anova(
    input_filenames: List[str],
    project_manifest_path: str,
    columns: List[str],
    alpha: float = 0.05
) -> Dict:
    """
    One-way ANOVA for comparing means across ≥3 groups (assumes normality, equal variances).
    H0: all group means are equal. Effect size: eta-squared.
    
    Args:
        input_filenames: List of CSV dataset filenames (one per group)
        project_manifest_path: Path to manifest.json
        columns: List of column names (one per dataset)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistic, p_value, alpha, is_significant, n_groups, group_sizes,
        group_means, group_stds, eta_squared, effect_size, interpretation, summary
    """
    if len(input_filenames) < 2:
        raise ValueError(f"Need at least 2 groups for ANOVA. Got: {len(input_filenames)}")
    
    if len(columns) != len(input_filenames):
        raise ValueError(
            f"Number of columns ({len(columns)}) must match number of datasets ({len(input_filenames)})"
        )
    
    # Load all groups
    groups = []
    group_means = []
    group_stds = []
    group_sizes = []
    
    for i, (filename, column) in enumerate(zip(input_filenames, columns)):
        df = _load_resource(project_manifest_path, filename)
        
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in dataset {i+1}. "
                f"Available: {list(df.columns)}"
            )
        
        data = df[column].dropna().values
        
        if len(data) < 2:
            raise ValueError(f"Group {i+1} needs at least 2 samples. Got: {len(data)}")
        
        groups.append(data)
        group_means.append(float(np.mean(data)))
        group_stds.append(float(np.std(data, ddof=1)))
        group_sizes.append(len(data))
    
    # Perform one-way ANOVA
    statistic, p_value = stats.f_oneway(*groups)
    
    # Calculate eta-squared (η²) - effect size for ANOVA
    # η² = SS_between / SS_total
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
    ss_total = sum((x - grand_mean)**2 for group in groups for x in group)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    
    # Interpret eta-squared
    if eta_squared < 0.01:
        effect_size_interp = "negligible"
    elif eta_squared < 0.06:
        effect_size_interp = "small"
    elif eta_squared < 0.14:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Interpret result
    is_significant = p_value <= alpha
    n_groups = len(groups)
    
    if is_significant:
        interpretation = (
            f"Significant differences detected among {n_groups} groups (p={p_value:.4f} ≤ α={alpha}). "
            f"At least one group mean differs from the others. "
            f"F-statistic: {statistic:.4f}."
        )
    else:
        interpretation = (
            f"No significant differences among {n_groups} groups (p={p_value:.4f} > α={alpha}). "
            f"All group means are statistically similar. "
            f"F-statistic: {statistic:.4f}."
        )
    
    return {
        "test": "One-way ANOVA",
        "datasets": input_filenames,
        "columns": columns,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "is_significant": bool(is_significant),
        "n_groups": int(n_groups),
        "group_sizes": group_sizes,
        "group_means": group_means,
        "group_stds": group_stds,
        "eta_squared": float(eta_squared),
        "effect_size": effect_size_interp,
        "interpretation": interpretation,
        "summary": f"ANOVA: F={statistic:.4f}, p={p_value:.4f}, η²={eta_squared:.4f} ({effect_size_interp}), {n_groups} groups, significant={is_significant}"
    }


def test_kruskal_wallis(
    input_filenames: List[str],
    project_manifest_path: str,
    columns: List[str],
    alpha: float = 0.05
) -> Dict:
    """
    Kruskal-Wallis H-test for comparing distributions across ≥2 groups (non-parametric).
    H0: all groups have same distribution. Effect size: epsilon-squared.
    
    Args:
        input_filenames: List of CSV dataset filenames (one per group)
        project_manifest_path: Path to manifest.json
        columns: List of column names (one per dataset)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistic, p_value, alpha, is_significant, n_groups, group_sizes,
        group_medians, epsilon_squared, effect_size, interpretation, summary
    """
    if len(input_filenames) < 2:
        raise ValueError(f"Need at least 2 groups for Kruskal-Wallis. Got: {len(input_filenames)}")
    
    if len(columns) != len(input_filenames):
        raise ValueError(
            f"Number of columns ({len(columns)}) must match number of datasets ({len(input_filenames)})"
        )
    
    # Load all groups
    groups = []
    group_medians = []
    group_sizes = []
    
    for i, (filename, column) in enumerate(zip(input_filenames, columns)):
        df = _load_resource(project_manifest_path, filename)
        
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in dataset {i+1}. "
                f"Available: {list(df.columns)}"
            )
        
        data = df[column].dropna().values
        
        if len(data) < 1:
            raise ValueError(f"Group {i+1} needs at least 1 sample. Got: {len(data)}")
        
        groups.append(data)
        group_medians.append(float(np.median(data)))
        group_sizes.append(len(data))
    
    # Perform Kruskal-Wallis test
    statistic, p_value = stats.kruskal(*groups)
    
    # Calculate epsilon-squared (ε²) - non-parametric effect size for Kruskal-Wallis
    # ε² = H / (n² - 1) / (n + 1) where H is the K-W statistic
    n_total = sum(group_sizes)
    epsilon_squared = statistic / ((n_total**2 - 1) / (n_total + 1))
    
    # Interpret epsilon-squared (similar thresholds to eta-squared)
    if epsilon_squared < 0.01:
        effect_size_interp = "negligible"
    elif epsilon_squared < 0.06:
        effect_size_interp = "small"
    elif epsilon_squared < 0.14:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Interpret result
    is_significant = p_value <= alpha
    n_groups = len(groups)
    
    if is_significant:
        interpretation = (
            f"Significant differences detected among {n_groups} groups (p={p_value:.4f} ≤ α={alpha}). "
            f"At least one group distribution differs from the others. "
            f"H-statistic: {statistic:.4f}."
        )
    else:
        interpretation = (
            f"No significant differences among {n_groups} groups (p={p_value:.4f} > α={alpha}). "
            f"All group distributions are statistically similar. "
            f"H-statistic: {statistic:.4f}."
        )
    
    return {
        "test": "Kruskal-Wallis H",
        "datasets": input_filenames,
        "columns": columns,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "is_significant": bool(is_significant),
        "n_groups": int(n_groups),
        "group_sizes": group_sizes,
        "group_medians": group_medians,
        "epsilon_squared": float(epsilon_squared),
        "effect_size": effect_size_interp,
        "interpretation": interpretation,
        "summary": f"Kruskal-Wallis: H={statistic:.4f}, p={p_value:.4f}, ε²={epsilon_squared:.4f} ({effect_size_interp}), {n_groups} groups, significant={is_significant}"
    }


def test_chi_square(
    input_filename: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05
) -> Dict:
    """
    Chi-square test of independence for categorical variables.
    H0: variables are independent. Effect size: Cramér's V.
    
    Args:
        input_filename: CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: First categorical column
        column_b: Second categorical column
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistic, p_value, degrees_of_freedom, alpha, is_significant,
        cramers_v, effect_size, contingency_table, interpretation, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    if column_a not in df.columns:
        raise ValueError(
            f"Column '{column_a}' not found. "
            f"Available: {list(df.columns)}"
        )
    if column_b not in df.columns:
        raise ValueError(
            f"Column '{column_b}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Remove rows with NaN in either column
    df_clean = df[[column_a, column_b]].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    # Create contingency table
    contingency_table = pd.crosstab(df_clean[column_a], df_clean[column_b])
    
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 categories in each variable. "
            f"Got {contingency_table.shape[0]} x {contingency_table.shape[1]}"
        )
    
    # Perform chi-square test
    chi2_result = stats.chi2_contingency(contingency_table)
    statistic = chi2_result[0]
    p_value = chi2_result[1]
    dof = chi2_result[2]
    
    # Calculate Cramér's V (effect size for chi-square)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    cramers_v = np.sqrt(statistic / (n * min_dim))
    
    # Interpret Cramér's V
    if cramers_v < 0.1:
        effect_size_interp = "negligible"
    elif cramers_v < 0.3:
        effect_size_interp = "small"
    elif cramers_v < 0.5:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Interpret result
    is_significant = p_value <= alpha
    
    if is_significant:
        interpretation = (
            f"Significant association detected (\u03c7\u00b2={statistic:.4f}, p={p_value:.4f} \u2264 \u03b1={alpha}). "
            f"The variables '{column_a}' and '{column_b}' are NOT independent. "
            f"Cramér's V={cramers_v:.4f} ({effect_size_interp} effect)."
        )
    else:
        interpretation = (
            f"No significant association (\u03c7\u00b2={statistic:.4f}, p={p_value:.4f} > \u03b1={alpha}). "
            f"The variables '{column_a}' and '{column_b}' appear independent. "
            f"Cramér's V={cramers_v:.4f} ({effect_size_interp} effect)."
        )
    
    # Convert contingency table to nested dict for JSON serialization
    contingency_dict = {
        str(idx): {str(col): int(val) for col, val in row.items()}
        for idx, row in contingency_table.to_dict('index').items()
    }
    
    return {
        "test": "Chi-square test of independence",
        "dataset": input_filename,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "alpha": float(alpha),
        "is_significant": bool(is_significant),
        "n_samples": int(n),
        "cramers_v": float(cramers_v),
        "effect_size": effect_size_interp,
        "contingency_table": contingency_dict,
        "interpretation": interpretation,
        "summary": f"Chi-square: \u03c7\u00b2={statistic:.4f}, p={p_value:.4f}, V={cramers_v:.4f} ({effect_size_interp}), significant={is_significant}"
    }


def test_fisher_exact(
    input_filename: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Fisher's exact test for 2x2 tables (exact test for small samples).
    H0: variables are independent. Effect size: odds ratio.
    
    Args:
        input_filename: CSV dataset filename
        project_manifest_path: Path to manifest.json
        column_a: First categorical column (2 categories)
        column_b: Second categorical column (2 categories)
        alpha: Significance level (default: 0.05)
        alternative: "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary with p_value, odds_ratio, alpha, alternative, is_significant,
        effect_size, contingency_table, interpretation, summary
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    if column_a not in df.columns:
        raise ValueError(
            f"Column '{column_a}' not found. "
            f"Available: {list(df.columns)}"
        )
    if column_b not in df.columns:
        raise ValueError(
            f"Column '{column_b}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Remove rows with NaN in either column
    df_clean = df[[column_a, column_b]].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    # Create contingency table
    contingency_table = pd.crosstab(df_clean[column_a], df_clean[column_b])
    
    # Fisher's exact test requires exactly 2x2 table
    if contingency_table.shape != (2, 2):
        raise ValueError(
            f"Fisher's exact test requires 2x2 contingency table. "
            f"Got {contingency_table.shape[0]} x {contingency_table.shape[1]}. "
            f"Both variables must have exactly 2 categories."
        )
    
    # Extract 2x2 table as numpy array
    table = contingency_table.values
    
    # Perform Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)
    
    # Interpret odds ratio as effect size
    # OR = 1: No association
    # OR > 1: Positive association
    # OR < 1: Negative association
    if odds_ratio == 0 or np.isinf(odds_ratio):
        effect_size_interp = "extreme (perfect or no overlap)"
    elif 0.9 <= odds_ratio <= 1.1:
        effect_size_interp = "negligible"
    elif (1.5 <= odds_ratio <= 3.5) or (1/3.5 <= odds_ratio <= 1/1.5):
        effect_size_interp = "small"
    elif (3.5 <= odds_ratio <= 9) or (1/9 <= odds_ratio <= 1/3.5):
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Interpret result
    is_significant = p_value <= alpha
    n = contingency_table.sum().sum()
    
    # Get category names for interpretation
    cat_a = list(contingency_table.index)
    cat_b = list(contingency_table.columns)
    
    if alternative == "two-sided":
        if is_significant:
            interpretation = (
                f"Significant association detected (p={p_value:.4f} ≤ α={alpha}). "
                f"The variables '{column_a}' and '{column_b}' are NOT independent. "
                f"Odds ratio={odds_ratio:.4f} ({effect_size_interp} effect)."
            )
        else:
            interpretation = (
                f"No significant association (p={p_value:.4f} > α={alpha}). "
                f"The variables '{column_a}' and '{column_b}' appear independent. "
                f"Odds ratio={odds_ratio:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Significant positive association (p={p_value:.4f} ≤ α={alpha}). "
                f"Odds of '{cat_b[1]}' are HIGHER for '{cat_a[0]}' vs '{cat_a[1]}'. "
                f"Odds ratio={odds_ratio:.4f} ({effect_size_interp} effect)."
            )
        else:
            interpretation = (
                f"No significant positive association (p={p_value:.4f} > α={alpha}). "
                f"Odds ratio={odds_ratio:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Significant negative association (p={p_value:.4f} ≤ α={alpha}). "
                f"Odds of '{cat_b[1]}' are LOWER for '{cat_a[0]}' vs '{cat_a[1]}'. "
                f"Odds ratio={odds_ratio:.4f} ({effect_size_interp} effect)."
            )
        else:
            interpretation = (
                f"No significant negative association (p={p_value:.4f} > α={alpha}). "
                f"Odds ratio={odds_ratio:.4f}."
            )
    
    # Convert contingency table to nested dict for JSON serialization
    contingency_dict = {
        str(idx): {str(col): int(val) for col, val in row.items()}
        for idx, row in contingency_table.to_dict('index').items()
    }
    
    return {
        "test": "Fisher's exact test",
        "dataset": input_filename,
        "column_a": column_a,
        "column_b": column_b,
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
        "alpha": float(alpha),
        "alternative": alternative,
        "is_significant": bool(is_significant),
        "n_samples": int(n),
        "effect_size": effect_size_interp,
        "contingency_table": contingency_dict,
        "interpretation": interpretation,
        "summary": f"Fisher's exact: p={p_value:.4f}, OR={odds_ratio:.4f} ({effect_size_interp}), significant={is_significant}"
    }


def test_mcnemar(
    input_filename_before: str,
    input_filename_after: str,
    project_manifest_path: str,
    column_before: str,
    column_after: str,
    alpha: float = 0.05
) -> Dict:
    """
    McNemar's test for paired binary categorical data (before/after changes).
    H0: marginal proportions are equal. Effect size: odds ratio.
    
    Args:
        input_filename_before: CSV dataset filename for "before"
        input_filename_after: CSV dataset filename for "after"
        project_manifest_path: Path to manifest.json
        column_before: Column in "before" dataset (binary)
        column_after: Column in "after" dataset (binary)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistic, p_value, alpha, is_significant, n_pairs, n_concordant,
        n_discordant, odds_ratio, effect_size, contingency_table, interpretation, summary
    """
    # Load datasets
    df_before = _load_resource(project_manifest_path, input_filename_before)
    df_after = _load_resource(project_manifest_path, input_filename_after)
    
    # Validate columns
    if column_before not in df_before.columns:
        raise ValueError(
            f"Column '{column_before}' not found in 'before' dataset. "
            f"Available: {list(df_before.columns)}"
        )
    if column_after not in df_after.columns:
        raise ValueError(
            f"Column '{column_after}' not found in 'after' dataset. "
            f"Available: {list(df_after.columns)}"
        )
    
    # Get data
    data_before = df_before[column_before].values
    data_after = df_after[column_after].values
    
    # Check equal lengths
    if len(data_before) != len(data_after):
        raise ValueError(
            f"Datasets must have equal length for paired test. "
            f"Before: {len(data_before)}, After: {len(data_after)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_before) | pd.isna(data_after))
    data_before_clean = data_before[valid_mask]
    data_after_clean = data_after[valid_mask]
    n_pairs = len(data_before_clean)
    
    if n_pairs == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    # Create contingency table for paired data
    contingency_table = pd.crosstab(data_before_clean, data_after_clean)
    
    # McNemar's test requires 2x2 table
    if contingency_table.shape != (2, 2):
        raise ValueError(
            f"McNemar's test requires 2x2 contingency table (binary variables). "
            f"Got {contingency_table.shape[0]} x {contingency_table.shape[1]}. "
            f"Before has {len(set(data_before_clean))} categories, "
            f"After has {len(set(data_after_clean))} categories."
        )
    
    # Extract 2x2 table as numpy array
    table = contingency_table.values
    
    # McNemar's test focuses on discordant pairs (b and c in the 2x2 table)
    # table layout:
    #               After=0  After=1
    # Before=0        a        b
    # Before=1        c        d
    
    a = table[0, 0]  # Both 0
    b = table[0, 1]  # Before=0, After=1 (changed to 1)
    c = table[1, 0]  # Before=1, After=0 (changed to 0)
    d = table[1, 1]  # Both 1
    
    n_concordant = a + d
    n_discordant = b + c
    
    # Calculate odds ratio for paired data (ratio of discordant pairs)
    if c == 0:
        odds_ratio = float('inf') if b > 0 else 1.0
    else:
        odds_ratio = b / c
    
    # Interpret odds ratio
    if np.isinf(odds_ratio):
        effect_size_interp = "extreme (all changes in one direction)"
    elif odds_ratio == 0:
        effect_size_interp = "extreme (all changes in opposite direction)"
    elif 0.9 <= odds_ratio <= 1.1:
        effect_size_interp = "negligible"
    elif (1.5 <= odds_ratio <= 3.5) or (1/3.5 <= odds_ratio <= 1/1.5):
        effect_size_interp = "small"
    elif (3.5 <= odds_ratio <= 9) or (1/9 <= odds_ratio <= 1/3.5):
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Perform McNemar's test
    result = mcnemar_test(table, exact=False)  # Use chi-square approximation
    statistic = result.statistic
    p_value = result.pvalue
    
    # Interpret result
    is_significant = p_value <= alpha
    
    # Get category names
    cat_before = list(contingency_table.index)
    cat_after = list(contingency_table.columns)
    
    if is_significant:
        if odds_ratio > 1:
            interpretation = (
                f"Significant change detected (p={p_value:.4f} ≤ α={alpha}). "
                f"More pairs changed from '{cat_before[0]}' to '{cat_after[1]}' (n={b}) "
                f"than from '{cat_before[1]}' to '{cat_after[0]}' (n={c}). "
                f"Odds ratio={odds_ratio:.4f} ({effect_size_interp} effect)."
            )
        elif odds_ratio < 1:
            interpretation = (
                f"Significant change detected (p={p_value:.4f} ≤ α={alpha}). "
                f"More pairs changed from '{cat_before[1]}' to '{cat_after[0]}' (n={c}) "
                f"than from '{cat_before[0]}' to '{cat_after[1]}' (n={b}). "
                f"Odds ratio={odds_ratio:.4f} ({effect_size_interp} effect)."
            )
        else:
            interpretation = (
                f"Significant change detected (p={p_value:.4f} ≤ α={alpha}), "
                f"but equal discordant pairs. Odds ratio={odds_ratio:.4f}."
            )
    else:
        interpretation = (
            f"No significant change (p={p_value:.4f} > α={alpha}). "
            f"Discordant pairs: {b} vs {c}. Odds ratio={odds_ratio:.4f}."
        )
    
    # Convert contingency table to nested dict for JSON serialization
    contingency_dict = {
        str(idx): {str(col): int(val) for col, val in row.items()}
        for idx, row in contingency_table.to_dict('index').items()
    }
    
    return {
        "test": "McNemar's test",
        "dataset_before": input_filename_before,
        "dataset_after": input_filename_after,
        "column_before": column_before,
        "column_after": column_after,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "is_significant": bool(is_significant),
        "n_pairs": int(n_pairs),
        "n_concordant": int(n_concordant),
        "n_discordant": int(n_discordant),
        "discordant_b": int(b),  # Before=0, After=1
        "discordant_c": int(c),  # Before=1, After=0
        "odds_ratio": float(odds_ratio),
        "effect_size": effect_size_interp,
        "contingency_table": contingency_dict,
        "interpretation": interpretation,
        "summary": f"McNemar's test: χ²={statistic:.4f}, p={p_value:.4f}, OR={odds_ratio:.4f} ({effect_size_interp}), significant={is_significant}"
    }


def get_all_statistical_test_tools():
    """
    Returns all MCP-exposed statistical test functions.
    """
    return [
        # Normality tests
        test_shapiro_wilk,
        test_kolmogorov_smirnov_norm,
        test_anderson_darling,
        # Paired comparison tests
        test_paired_ttest,
        test_wilcoxon_signed_rank,
        # Correlation tests
        test_pearson_correlation,
        test_spearman_correlation,
        # Independent sample tests
        test_independent_ttest,
        test_mann_whitney_u,
        test_kolmogorov_smirnov_two_sample,
        # Multi-group tests
        test_one_way_anova,
        test_kruskal_wallis,
        # Categorical tests
        test_chi_square,
        test_fisher_exact,
        test_mcnemar,
    ]