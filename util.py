import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ranksums, spearmanr, wilcoxon, hypergeom, kruskal
from statsmodels.stats.multitest import multipletests
import multiprocessing

equality_epsilon = 1e-6

def test_one_element_values(df, column, is_paired, groups=None, sorted_groups=None):
    if groups is None:
        groups = df.groupby('type')
        group_names, groups = zip(*groups)
    if sorted_groups is None and is_paired:
        sorted_groups = [group.sort_index() for group in groups]
    if is_paired:
        if len(groups) != 2:
            raise ValueError("Non-parametric paired data test is unimplemented except for two groups.")
        assert all([all(g.index.get_level_values(level=0) == t.index.get_level_values(level=0))
                    for (g, t) in zip(sorted_groups, sorted_groups[1:])])
        try:
            p = wilcoxon(*[g[column] for g in sorted_groups])[1]
        except Exception as e:
            print("Warning: nan value for wilcoxon test")
            p = np.nan
    else:
        # kruskal is non-parametric ANOVA, and returns the same p-values as ranksums (Wilcoxon rank-sum) for two
        # groups.
        try:
            p = kruskal(*[g[column] for g in groups])[1]
        except Exception as e:
            print("Warning: nan value for kruscal test of {}".format(column))
            p = np.nan
    return p

def get_dataframe_intersection(df1, df2):
    intersecting_columns = list(df1.columns.intersection(df2.columns))
    intersecting_index = df1.index.intersection(df2.index)
    if len(intersecting_columns) < max(len(df1.columns), len(df2.columns)):
        print("Warning: only {} elements in intersection (out of {} and {})".format(len(intersecting_columns),
                                                                                    len(df1.columns), len(df2.columns)))
    if len(intersecting_index) < max(len(df1.index), len(df2.index)):
        print("Warning: only {} samples in intersection (out of {} and {})".format(len(intersecting_index),
                                                                                   len(df1.index), len(df2.index)))
    df1_filtered = df1[intersecting_columns].filter(intersecting_index, axis=0)
    df2_filtered = df2[intersecting_columns].filter(intersecting_index, axis=0)

    return df1_filtered, df2_filtered


def omit_corr(x, y, corr_func):
    nan_indices = np.isnan(x) | np.isnan(y)
    x, y = x[~nan_indices], y[~nan_indices]
    # if x or y are constant, return nan
    if len(set(x)) == 1 or len(set(y)) == 1:
        return np.nan, np.nan
    return corr_func(x, y)


def run_correlation_tests(df1, df2, parallel=False, processes=30, corr_func=spearmanr):
    """
    Given two dataframes with similar index, runs the spearman rank correlation test between the two versions of all
    intersecting columns. Also runs a correlation test between the flattened data, by concatenating all columns.
    """
    df1, df2 = get_dataframe_intersection(df1, df2)
    cols = df1.columns

    test_pvalues = dict()
    test_correlations = dict()
    if parallel:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(omit_corr, [(df1[element].astype('float'), df2[element].astype('float'), corr_func)
                                               for element in cols])
        for i, element in enumerate(cols):
            corr, pvalue = results[i]
            test_pvalues[element] = pvalue
            test_correlations[element] = corr
    else:
        for element in cols:
            x, y = df1[element].astype('float'), df2[element].astype('float')
            nan_indices = np.isnan(x) | np.isnan(y)
            x, y = x[~nan_indices], y[~nan_indices]
            # if x or y are constant, return nan
            if len(set(x)) == 1 or len(set(y)) == 1:
                test_pvalues[element] = np.nan
                test_correlations[element] = np.nan
                continue
            corr, pvalue = corr_func(x, y)
            test_pvalues[element] = pvalue
            test_correlations[element] = corr

    # flatten dataframes and run a full test on all data.
    df1_relevant_flattened = pd.concat([df1[e] for e in cols])
    df2_relevant_flattened = pd.concat([df2[e] for e in cols])
    assert all(df1_relevant_flattened.index == df2_relevant_flattened.index)
    nan_indices = np.isnan(df1_relevant_flattened) | np.isnan(df2_relevant_flattened)

    full_corr, full_pvalue = corr_func(df1_relevant_flattened.astype('float')[~nan_indices],
                                       df2_relevant_flattened.astype('float')[~nan_indices], nan_policy='omit')

    return test_correlations, test_pvalues, full_corr, full_pvalue


def run_tests_for_vectors(data, is_paired, verbose=True, multiple_hypothesis_correction=True):
    """
    Given some data (fluxes / activities / buildup) of shape patient X elements (reactions / metabolites),
    with a two-level index of sample ids and types, defining different tumor groups (and possibly a healthy one)
    run statistical tests, per element, for whether the means of the data for this element are
    the same across all groups.
    If is_paired, assumes the indice labels of the groups
    match.
    For elements with no variance, ignore and do not include in returned dictionary keys.
    """
    elements = data.keys()
    test_pvalues = dict()
    empty = 0
    no_group_difference_elements = set()

    grouped_data = data.groupby('type')
    group_names, grouped_data = zip(*grouped_data)
    sorted_groups = [group.sort_index() for group in grouped_data] if is_paired else None
    if verbose:
        print("#elements to run over: {}".format(len(data.keys())))
    for i, element in enumerate(elements):
        # if i and not (i % 5000):
        #     if verbose:
        #         print("done {}".format(i))
        if (data[element].var() < equality_epsilon) or (data[element].isnull().sum() > 0):
            empty += 1
            if empty and not (empty % 1000):
                if verbose:
                    print("{} empty so far".format(empty))
            continue
        elif len(set(data[element].groupby('type').mean())) == 1:  # TODO: replace with approximate version?
            no_group_difference_elements.add(element)
            continue
        p = test_one_element_values(data, element, is_paired, groups=grouped_data, sorted_groups=sorted_groups)
        test_pvalues[element] = p

    if len(test_pvalues) == 0:
        print("Warning: no elements with variance found for testing!")
        return dict()

    element_list, pval_list = zip(*test_pvalues.items())
    if multiple_hypothesis_correction:
        # Benjamini/Yekutieli correction (controls family-wise FDR with no independence assumptions on tests)
        significant_truthvals, corrected_pval_list = multipletests(pval_list, method='fdr_by')[:2]
    else:
        significant_truthvals, corrected_pval_list = (np.array([pval <= 0.05 for element, pval in zip(element_list, pval_list)]),
                                                      np.array([pval for element, pval in zip(element_list, pval_list)]))
    test_pvalues = dict(zip(element_list, corrected_pval_list))
    significants = np.array(element_list)[significant_truthvals]

    if len(no_group_difference_elements) > 0:
        print("Warning! Identical values between groups for {} elements (skipped).".format(
            len(no_group_difference_elements)))

    return {e: v for (e, v) in test_pvalues.items() if e in significants}

