"""Tests for statistics.py functions with expected statistical outcomes."""
import pandas as pd
import numpy as np
import pytest


def test_shapiro_wilk_normal_data(session_workdir):
    """Test Shapiro-Wilk normality test with normally distributed data - should be not significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_shapiro_wilk

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create normal data - should pass normality test
    np.random.seed(42)
    df = pd.DataFrame({
        'values': np.random.normal(loc=50, scale=10, size=100)
    })

    input_filename = _store_resource(df, manifest_path, "normal_data", "Normal distribution", "csv")

    result = test_shapiro_wilk(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column='values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_normal" in result
    assert "statistic" in result
    # Should NOT reject normality (p > 0.05)
    assert result["is_normal"] == True
    assert result["p_value"] > 0.05


def test_kolmogorov_smirnov_norm_normal_data(session_workdir):
    """Test K-S normality test with normally distributed data - should be not significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_kolmogorov_smirnov_norm

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create normal data
    np.random.seed(42)
    df = pd.DataFrame({
        'values': np.random.normal(loc=100, scale=15, size=200)
    })

    input_filename = _store_resource(df, manifest_path, "ks_normal_data", "Normal distribution", "csv")

    result = test_kolmogorov_smirnov_norm(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column='values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_normal" in result
    assert "statistic" in result
    # Should pass normality test
    assert result["is_normal"] == True
    assert result["p_value"] > 0.05


def test_anderson_darling_normal_data(session_workdir):
    """Test Anderson-Darling normality test with normally distributed data - should be not significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_anderson_darling

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create normal data
    np.random.seed(42)
    df = pd.DataFrame({
        'values': np.random.normal(loc=75, scale=12, size=150)
    })

    input_filename = _store_resource(df, manifest_path, "ad_normal_data", "Normal distribution", "csv")

    result = test_anderson_darling(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column='values'
    )

    # Verify structure
    assert "statistic" in result
    assert "is_normal" in result
    assert "critical_value" in result
    # Should pass normality test (statistic < critical value)
    assert result["is_normal"] == True


def test_paired_ttest_significant_difference(session_workdir):
    """Test paired t-test with significant difference between before and after - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_paired_ttest

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create before/after data with clear difference (e.g., drug reduces values by 30 points)
    np.random.seed(42)
    n = 50
    before = np.random.normal(loc=100, scale=10, size=n)
    after = before - 30 + np.random.normal(loc=0, scale=5, size=n)  # Reduce by ~30

    df_before = pd.DataFrame({'score': before})
    df_after = pd.DataFrame({'score': after})

    filename_before = _store_resource(df_before, manifest_path, "before_treatment", "Before", "csv")
    filename_after = _store_resource(df_after, manifest_path, "after_treatment", "After", "csv")

    result = test_paired_ttest(
        input_filename_a=filename_before,
        input_filename_b=filename_after,
        project_manifest_path=manifest_path,
        column_a='score',
        column_b='score'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    # Should detect significant difference
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001
    # Large effect size (check for effect_size field)
    if "cohens_d" in result:
        assert abs(result["cohens_d"]) > 2.0
    elif "effect_size" in result:
        assert abs(result["effect_size"]) > 2.0


def test_wilcoxon_signed_rank_significant(session_workdir):
    """Test Wilcoxon signed-rank test with significant difference - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_wilcoxon_signed_rank

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create paired data with clear difference using exponential (non-normal) distribution
    np.random.seed(42)
    n = 50
    before = np.random.exponential(scale=10, size=n)
    after = before * 0.5 + np.random.exponential(scale=2, size=n)  # Reduce values

    df_before = pd.DataFrame({'values': before})
    df_after = pd.DataFrame({'values': after})

    filename_before = _store_resource(df_before, manifest_path, "wilcoxon_before", "Before", "csv")
    filename_after = _store_resource(df_after, manifest_path, "wilcoxon_after", "After", "csv")

    result = test_wilcoxon_signed_rank(
        input_filename_a=filename_before,
        input_filename_b=filename_after,
        project_manifest_path=manifest_path,
        column_a='values',
        column_b='values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    # Should detect significant difference
    assert result["is_significant"] == True
    assert result["p_value"] < 0.05


def test_pearson_correlation_strong_positive(session_workdir):
    """Test Pearson correlation with strong positive linear correlation - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_pearson_correlation

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create data with strong linear correlation
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=10, size=100)
    y = 2 * x + 10 + np.random.normal(loc=0, scale=5, size=100)  # y ≈ 2x + 10, r > 0.9

    df_x = pd.DataFrame({'x_values': x})
    df_y = pd.DataFrame({'y_values': y})

    filename_x = _store_resource(df_x, manifest_path, "pearson_x", "X data", "csv")
    filename_y = _store_resource(df_y, manifest_path, "pearson_y", "Y data", "csv")

    result = test_pearson_correlation(
        input_filename_a=filename_x,
        input_filename_b=filename_y,
        project_manifest_path=manifest_path,
        column_a='x_values',
        column_b='y_values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "correlation" in result
    # Should detect strong positive correlation
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001
    assert result["correlation"] > 0.9


def test_spearman_correlation_monotonic(session_workdir):
    """Test Spearman correlation with monotonic relationship - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_spearman_correlation

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create data with strong monotonic (but non-linear) relationship
    np.random.seed(42)
    x = np.linspace(1, 100, 100)
    y = np.log(x) * 10 + np.random.normal(loc=0, scale=1, size=100)  # Logarithmic relationship

    df_x = pd.DataFrame({'x_values': x})
    df_y = pd.DataFrame({'y_values': y})

    filename_x = _store_resource(df_x, manifest_path, "spearman_x", "X data", "csv")
    filename_y = _store_resource(df_y, manifest_path, "spearman_y", "Y data", "csv")

    result = test_spearman_correlation(
        input_filename_a=filename_x,
        input_filename_b=filename_y,
        project_manifest_path=manifest_path,
        column_a='x_values',
        column_b='y_values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "correlation" in result
    # Should detect strong monotonic correlation
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001
    assert result["correlation"] > 0.9


def test_independent_ttest_significant(session_workdir):
    """Test independent t-test with significant difference between groups - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_independent_ttest

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create two groups with large difference in means
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=50)
    group_b = np.random.normal(loc=130, scale=10, size=50)  # 30-point difference

    df_a = pd.DataFrame({'values': group_a})
    df_b = pd.DataFrame({'values': group_b})

    filename_a = _store_resource(df_a, manifest_path, "ttest_group_a", "Group A", "csv")
    filename_b = _store_resource(df_b, manifest_path, "ttest_group_b", "Group B", "csv")

    result = test_independent_ttest(
        input_filename_a=filename_a,
        input_filename_b=filename_b,
        project_manifest_path=manifest_path,
        column_a='values',
        column_b='values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "cohens_d" in result
    # Should detect significant difference
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001
    # Large effect size
    assert abs(result["cohens_d"]) > 2.0


def test_mann_whitney_u_significant(session_workdir):
    """Test Mann-Whitney U test with significant difference between groups - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_mann_whitney_u

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create two groups with clear difference using exponential distributions
    np.random.seed(42)
    group_a = np.random.exponential(scale=5, size=50)
    group_b = np.random.exponential(scale=20, size=50)  # Much larger values

    df_a = pd.DataFrame({'values': group_a})
    df_b = pd.DataFrame({'values': group_b})

    filename_a = _store_resource(df_a, manifest_path, "mwu_group_a", "Group A", "csv")
    filename_b = _store_resource(df_b, manifest_path, "mwu_group_b", "Group B", "csv")

    result = test_mann_whitney_u(
        input_filename_a=filename_a,
        input_filename_b=filename_b,
        project_manifest_path=manifest_path,
        column_a='values',
        column_b='values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "cliffs_delta" in result
    # Should detect significant difference
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001


def test_kolmogorov_smirnov_two_sample_different(session_workdir):
    """Test two-sample K-S test with different distributions - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_kolmogorov_smirnov_two_sample

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create two groups from different distributions
    np.random.seed(42)
    group_a = np.random.normal(loc=50, scale=10, size=100)  # Normal
    group_b = np.random.exponential(scale=30, size=100)  # Exponential (different shape)

    df_a = pd.DataFrame({'values': group_a})
    df_b = pd.DataFrame({'values': group_b})

    filename_a = _store_resource(df_a, manifest_path, "ks2_group_a", "Group A", "csv")
    filename_b = _store_resource(df_b, manifest_path, "ks2_group_b", "Group B", "csv")

    result = test_kolmogorov_smirnov_two_sample(
        input_filename_a=filename_a,
        input_filename_b=filename_b,
        project_manifest_path=manifest_path,
        column_a='values',
        column_b='values'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "statistic" in result
    # Should detect different distributions
    assert result["is_significant"] == True
    assert result["p_value"] < 0.05


def test_one_way_anova_significant(session_workdir):
    """Test one-way ANOVA with significant differences between groups - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_one_way_anova

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create three groups with different means
    np.random.seed(42)
    group_a = np.random.normal(loc=50, scale=10, size=40)
    group_b = np.random.normal(loc=70, scale=10, size=40)
    group_c = np.random.normal(loc=90, scale=10, size=40)

    df_a = pd.DataFrame({'values': group_a})
    df_b = pd.DataFrame({'values': group_b})
    df_c = pd.DataFrame({'values': group_c})

    filename_a = _store_resource(df_a, manifest_path, "anova_group_a", "Group A", "csv")
    filename_b = _store_resource(df_b, manifest_path, "anova_group_b", "Group B", "csv")
    filename_c = _store_resource(df_c, manifest_path, "anova_group_c", "Group C", "csv")

    result = test_one_way_anova(
        input_filenames=[filename_a, filename_b, filename_c],
        project_manifest_path=manifest_path,
        columns=['values', 'values', 'values']
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "eta_squared" in result
    # Should detect significant differences
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001


def test_kruskal_wallis_significant(session_workdir):
    """Test Kruskal-Wallis test with significant differences between groups - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_kruskal_wallis

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create three groups with different distributions (non-normal)
    np.random.seed(42)
    group_a = np.random.exponential(scale=5, size=40)
    group_b = np.random.exponential(scale=15, size=40)
    group_c = np.random.exponential(scale=30, size=40)

    df_a = pd.DataFrame({'values': group_a})
    df_b = pd.DataFrame({'values': group_b})
    df_c = pd.DataFrame({'values': group_c})

    filename_a = _store_resource(df_a, manifest_path, "kw_group_a", "Group A", "csv")
    filename_b = _store_resource(df_b, manifest_path, "kw_group_b", "Group B", "csv")
    filename_c = _store_resource(df_c, manifest_path, "kw_group_c", "Group C", "csv")

    result = test_kruskal_wallis(
        input_filenames=[filename_a, filename_b, filename_c],
        project_manifest_path=manifest_path,
        columns=['values', 'values', 'values']
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "epsilon_squared" in result
    # Should detect significant differences
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001


def test_chi_square_significant(session_workdir):
    """Test chi-square test with strong association - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_chi_square

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create data with strong association between treatment and outcome
    # Treatment A: mostly success, Treatment B: mostly failure
    df = pd.DataFrame({
        'treatment': ['A'] * 80 + ['B'] * 20 + ['A'] * 20 + ['B'] * 80,
        'outcome': ['success'] * 80 + ['success'] * 20 + ['failure'] * 20 + ['failure'] * 80
    })

    input_filename = _store_resource(df, manifest_path, "chi_square_data", "Categorical data", "csv")

    result = test_chi_square(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column_a='treatment',
        column_b='outcome'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "cramers_v" in result
    # Should detect significant association
    assert result["is_significant"] == True
    assert result["p_value"] < 0.001
    # Large effect size
    assert result["cramers_v"] > 0.5


def test_fisher_exact_significant(session_workdir):
    """Test Fisher's exact test with clear association in 2x2 table - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_fisher_exact

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create 2x2 data with clear association
    df = pd.DataFrame({
        'treatment': ['A'] * 18 + ['B'] * 2 + ['A'] * 2 + ['B'] * 18,
        'outcome': ['success'] * 18 + ['success'] * 2 + ['failure'] * 2 + ['failure'] * 18
    })

    input_filename = _store_resource(df, manifest_path, "fisher_data", "2x2 data", "csv")

    result = test_fisher_exact(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column_a='treatment',
        column_b='outcome'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "odds_ratio" in result
    # Should detect significant association
    assert result["is_significant"] == True
    assert result["p_value"] < 0.01
    # Odds ratio - either very large or very small indicates strong association
    assert result["odds_ratio"] < 0.1 or result["odds_ratio"] > 10


def test_mcnemar_significant_change(session_workdir):
    """Test McNemar's test with significant change in paired binary data - should be significant."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core.statistics import test_mcnemar

    manifest_path = str(session_workdir / "test_manifest.json")

    # Create before/after data showing significant improvement
    # Before: 70 negative, 30 positive
    # After: 30 negative, 70 positive (most negatives became positive)
    before_labels = ['negative'] * 50 + ['positive'] * 10 + ['negative'] * 20 + ['positive'] * 20
    after_labels = ['negative'] * 50 + ['positive'] * 10 + ['positive'] * 20 + ['positive'] * 20

    df_before = pd.DataFrame({'status': before_labels})
    df_after = pd.DataFrame({'status': after_labels})

    filename_before = _store_resource(df_before, manifest_path, "mcnemar_before", "Before", "csv")
    filename_after = _store_resource(df_after, manifest_path, "mcnemar_after", "After", "csv")

    result = test_mcnemar(
        input_filename_before=filename_before,
        input_filename_after=filename_after,
        project_manifest_path=manifest_path,
        column_before='status',
        column_after='status'
    )

    # Verify structure
    assert "p_value" in result
    assert "is_significant" in result
    assert "statistic" in result
    # Should detect significant change
    assert result["is_significant"] == True
    assert result["p_value"] < 0.05
