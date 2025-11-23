import pandas as pd
import numpy as np
import time
import scipy
from enum import Enum
import re
from sympy.parsing.sympy_parser import parse_expr
import sympy
def filter_infrequent_types(df, min_samples):
    counts = df.index.get_level_values(1).value_counts()
    frequent_types = counts[counts >= min_samples].index
    return df.loc[(df.index.get_level_values(0), frequent_types), :]


ArithmetizationMethod = Enum('ArithmetizationMethod', ['sumprod', 'maxmin', 'sumgeomean', 'arithmeangeomean'])
SampleNormalizationMethod = Enum("SampleNormalizationMethod", ['none', 'rank', 'quantile', 'ternary'])

def process_full_ge(cobra_model, ge_data_df,
                  verbose=True,
                  fixed_range_activities=False,
                  model_suffix="_AT[0-9]+",
                  gene_zero_fraction_threshold=0,
                  sample_normalization_method=SampleNormalizationMethod.rank,
                  post_processing_centering=True,
                arithmetization_mode=ArithmetizationMethod.arithmeangeomean,
                  ):
    print("Full GE ({} samples) normalization started.".format(len(ge_data_df)))
    start_time = time.time()

    ## Process GE

    ####
    # discrete_model.NegativeBinomial(endog=list(filtered_ge_data[filtered_ge_data.columns[9]]),
    #                                 exog=add_constant(pd.get_dummies(filtered_ge_data.index.get_level_values(1))),
    #                                 missing=False).fit_regularized().summary()
    if gene_zero_fraction_threshold > 0:
        ge_data_df = ge_data_df.loc[:, ge_data_df[ge_data_df == 0].count() <= len(ge_data_df) * gene_zero_fraction_threshold]
    # print(ge_data_df)
    if str(sample_normalization_method) == str(SampleNormalizationMethod.rank):
        ge_data_df = ge_data_df.rank(method='min', pct=True, axis=1)
    elif str(sample_normalization_method) == str(SampleNormalizationMethod.ternary):
        ge_data_df = ge_data_df.apply(discretize_values, axis=0, num_bins=3,
                                binarize_extremes=True, center=False)
    elif str(sample_normalization_method) == str(SampleNormalizationMethod.quantile):
        # take means of each gene as reference distribution, convert every sample to reference distribution based on ranks.
        rank_mean = ge_data_df.stack().groupby(ge_data_df.rank(method='first', axis=1).stack().astype(int)).mean()
        ge_data_df = ge_data_df.rank(method='min', axis=1).stack().astype(int).map(rank_mean).unstack()
        # assert all(ge_data_df.var(axis=1) != 0)
        assert not ge_data_df.isna().all().all()
    elif str(sample_normalization_method) == str(SampleNormalizationMethod.none):
        ge_data_df = ge_data_df
    else:
        raise ValueError("Unknown sample normalization method {}".format(sample_normalization_method))
    def map_activities(row):
        return pd.Series(ge_to_reaction_activities(cobra_model, row.to_dict(),
                                                   discretize=False, default_value_for_nan=np.nan,
                                                   arithmetization_mode=arithmetization_mode, model_suffix=model_suffix))

    activities_df = ge_data_df.astype(float).swifter.apply(map_activities, axis=1)
    # print(activities_df)
    if not all(activities_df.var(axis=1) != 0):
        print("Warning: some samples have zero variance in activities. ")
        print("Samples with zero variance:\n{}".format(
            activities_df.loc[activities_df.var(axis=1) == 0].index.tolist()))

    # assert all(activities_df.var(axis=1) != 0)
    assert not activities_df.isna().all().all()
    if fixed_range_activities:
        if post_processing_centering:
            raise ValueError("Post processing centering is inconsistent with fixed_range activities")
        m, M = activities_df.min(axis=1), activities_df.max(axis=1)
        activities_df = (2 * activities_df).subtract(m + M, axis=0).divide((M - m), axis=0)
    elif post_processing_centering:
        activities_df = activities_df.transpose().apply(scipy.stats.zscore, nan_policy='omit').transpose()
    else:
        pass

    if verbose:
        print("GE to reaction activities time: {:.2f}m".format((time.time() - start_time) / 60))
        print("{:.2f} fraction of activity values are null after mapping.".format(
            activities_df.isna().mean(axis=1).mean()))
    return activities_df


def ge_to_reaction_activities(model, gene_levels_map, discretize=True, discretize_quantiles=None,
                              discretize_binarize_extremes=True, arithmetization_mode=ArithmetizationMethod.sumprod,
                              allow_missing_genes=True, allow_missing_reactions=True,
                              default_value_for_nan=None,
                              model_suffix="_AT[0-9]+"):
    """
    Transforms gene activity levels to reaction activity levels, by arithmetization of logic rules.
    TODO: do rules make sense? Should 'and' be multiplication? Should there be different rules for [0,1] and [0,\inf]?
    :param model: A cobrapy model.
    :param gene_levels_map: A mapping between gene ids to activity levels. Can be in range [0,1] or [0,\inf)
    :param discretize: If true, computes discretize_quantiles of activity_values
    :param discretize_quantiles: #quantiles to use if discretize. i bin gets a value of (i+1)/(discretize_quantiles +1)
    If None defaults to the number of genes (should be similar to rank normalization, up to the binary extremes).
    :param discretize_binarize_extremes: If True and discretize is True, first and last bin values are set as 0 and 1.
    :param allow_missing_genes: If True, missing genes are replaced with a neutral value according to arithmetization.
    Otherwise raises error when missing a gene.
    :param allow_missing_reactions: If false, reactions with no corresponding gene value raise an exception.
    :param default_value_for_nan: If None, missing reactions and reactions with nan GE for their genes aren't
    included in the output. Otherwise, this is the value all they will all get.
    Ohterwise raises error when a reaction has no corresponding gene value.
    :param arithmetization_mode: either of ['sumprod', 'maxmin', 'sumgeomean', 'arithmeangeomean']. On sumprod, OR is
    transformed to + and AND to *, on maxmin OR is max() and AND is min(), on sumgeomean OR is + and AND is the
    geometric mean, and on arithmeangeomean OR is arithmetic mean, and AND is geometric mean.
    In all options, 'not', if discretize=True (or if levels already in range [0,1]), corresponds to 1's complement.
    Error otherwise.
    :model_suffix: if not None, a regex that is added as a suffix to any GE id to match with (possibly many) model genes

    :return:
    """

    gene_ids, gene_levels = list(zip(*list(gene_levels_map.items())))
    if discretize:
        if discretize_quantiles is None:
            # rank normalization.
            gene_levels = scipy.stats.rankdata(gene_levels, method='max') / (len(gene_levels) + 1)
            if discretize_binarize_extremes:
                gene_levels = np.where(gene_levels == len(gene_levels) / float(len(gene_levels) + 1), 1,
                                       gene_levels)
                gene_levels = np.where(gene_levels == 1.0 / (len(gene_levels) + 1), 0,
                                       gene_levels)
        else:
            gene_levels = discretize_values(gene_levels, discretize_quantiles, discretize_binarize_extremes)
    gene_levels_map = dict(list(zip(gene_ids, gene_levels)))
    is_normalized = discretize or all(0 <= level <= 1 for level in gene_levels_map.values())
    reaction_activities_map = dict()
    # We later add a version of all gene ids with _ as a prefix, since sympy otherwise might treat gene ids as nums
    gene_levels_map.update({"_" + g: v for g, v in gene_levels_map.items()})
    if model_suffix is None:
        model_gene_names = set([g.id for g in model.genes])
    else:
        # Some models have finer gene distinctions than GE ids, remove those if needed
        model_gene_names = set([re.sub(model_suffix, "", g.id) for g in model.genes])

    if min(gene_levels_map.values()) < 0:
        print("Warning: negative gene values - might not make sense for some arithmetization rules.")

    for reaction in model.reactions:
        rule_expression = reaction.gene_reaction_rule
        if rule_expression == "":
            if not allow_missing_reactions:
                raise ValueError("Reaction {} does not have any gene-reaction-rule".format(reaction.name))
            reaction_activities_map[reaction.id] = np.nan
            continue
        # if "not" in reaction.gene_reaction_rule and not is_normalized:
        #     raise ValueError("Can't arithmetize not with values outside [0,1]")
        if "not" in rule_expression:
            raise ValueError("cobrapy doesn't support 'not' rules.")

        # Some models have finer gene distinctions than GE ids, remove those if needed
        if model_suffix is not None:
            rule_expression = re.sub(model_suffix, "", rule_expression)

        # bypass sympy parsing (sometimes _very_ slow) if the rule is trivial
        if " or " not in rule_expression or " and " not in rule_expression:
            if " or " in rule_expression:
                value = _arithmetize_flat_ors(rule_expression, gene_levels_map, arithmetization_mode,
                                              cap=is_normalized, allow_missing_genes=allow_missing_genes)
            elif " and " in rule_expression:
                value = _arithmetize_flat_ands(rule_expression, gene_levels_map, arithmetization_mode,
                                               allow_missing_genes=allow_missing_genes)
            elif re.sub("[()]", "", rule_expression) in model_gene_names:
                if not allow_missing_genes and rule_expression not in gene_levels_map:
                    raise ValueError("Reaction {} is missing gene values for genes {}".format(
                        reaction.id, rule_expression))
                value = gene_levels_map.get(rule_expression, np.nan)
            else:
                raise ValueError("Unrecognized reaction rule {}".format(rule_expression))
        else:
            rule_expression = rule_expression.replace(" or ", " | ").replace(" and ", " & ").replace(" not ", " ~ ")
            # sympy evaluates stuff that begins with a numeral as a number, so some gene ids need to get a preceeding _
            for gene in reaction.genes:
                no_suffix_id = re.sub(model_suffix, "", gene.id) if model_suffix is not None else gene.id
                rule_expression = re.sub("(?P<prefix> |^|\\()" + no_suffix_id + "(?P<suffix> |$|\\))",
                                         "\\g<prefix>_" + no_suffix_id + "\\g<suffix>", rule_expression)

            logic_expr = parse_expr(rule_expression)
            reaction_missing_genes = {str(a) for a in logic_expr.atoms()}.difference(gene_levels_map)

            if len(reaction_missing_genes) > 0 and not allow_missing_genes:
                raise ValueError("Reaction {} is missing gene values for genes {}".format(reaction.id,
                                                                                          reaction_missing_genes))
            value = _arithmetize_sympy_expr(logic_expr, gene_levels_map, arithmetization_mode,
                                            cap=is_normalized)
        reaction_activities_map[reaction.id] = value

    if default_value_for_nan is None:
        reaction_activities_map = {r: val for r, val in reaction_activities_map.items() if not np.isnan(val)}
    else:
        reaction_activities_map = {r: (val if not np.isnan(val) else default_value_for_nan)
                                   for r, val in reaction_activities_map.items()}

    return reaction_activities_map

def _arithmetize_flat_ors(expr, symbol_values_map, arithmetization_mode, cap, allow_missing_genes):
    expr = expr.replace("(", " ").replace(")", " ")
    args = expr.split(" or ")
    vals = [symbol_values_map[s.strip()] for s in args if s.strip() in symbol_values_map]
    if len(vals) != len(args) and not allow_missing_genes:
        raise ValueError("Missing gene values for expression {}".format(expr))
    if len(vals) == 0:
        return np.nan
    if arithmetization_mode.value in [ArithmetizationMethod.sumprod.value, ArithmetizationMethod.sumgeomean.value]:
        s = np.nansum(vals)
        return min(1, s) if cap else s
    elif arithmetization_mode.value == ArithmetizationMethod.arithmeangeomean.value:
        return np.nanmean(vals)
    elif arithmetization_mode.value == ArithmetizationMethod.maxmin.value:
        return np.nanmax(vals)
    else:
        raise ValueError("Unknown arithmetization mode {}".format(arithmetization_mode))


def _arithmetize_flat_ands(expr, symbol_values_map, arithmetization_mode, allow_missing_genes):
    expr = expr.replace("(", " ").replace(")", " ")
    args = expr.split(" and ")
    vals = [symbol_values_map[s.strip()] for s in args if s.strip() in symbol_values_map]
    if len(vals) != len(args) and not allow_missing_genes:
        raise ValueError("Missing gene values for expression {}".format(expr))
    if len(vals) == 0:
        return np.nan
    if arithmetization_mode.value == ArithmetizationMethod.sumprod.value:
        return np.nanprod(vals)
    elif arithmetization_mode.value in [ArithmetizationMethod.sumgeomean.value, ArithmetizationMethod.arithmeangeomean.value]:
        return np.exp(np.nanmean(np.log(vals))) if (np.nanprod(vals) != 0) else 0
    elif arithmetization_mode.value == ArithmetizationMethod.maxmin.value:
        return np.nanmin(vals)
    else:
        raise ValueError("Unknown arithmetization mode {}".format(arithmetization_mode))


def _arithmetize_sympy_expr(expr, symbol_values_map, arithmetization_mode, cap):
    """
    Traverses the expression tree of a logic symbol expression, arithmetizing and substituting to get its value.
    :param arithmetization_mode: either of ['sumprod', 'maxmin', 'sumgeomean', 'arithmeangeomean']. On sumprod, OR is
    transformed to + and AND to *, on maxmin OR is max() and AND is min(), on sumgeomean OR is + and AND is the
    geometric mean, and on arithmeangeomean OR is arithmetic mean, and AND is geometric mean.
    Note that cobra doesn't suppoert 'not' rules, so we won't either.
    :param cap: if True, "or" result is capped by 1.
    :param expr: a sympy expression containing symbols and logical operations (and, not, or)
    :param symbol_values_map: a map from strings of the terminal symbols in the expression to their substituted values
    :return:
    """
    if expr.is_symbol:
        return symbol_values_map.get(str(expr), np.nan)
    else:
        arg_values = [_arithmetize_sympy_expr(argument, symbol_values_map, arithmetization_mode,
                                              cap) for argument in expr.args]
        arg_values = [val if val is not None else np.nan for val in arg_values]
        if np.isnan(arg_values).all():
            return np.nan
        if isinstance(expr, sympy.Or):
            if arithmetization_mode.value in [ArithmetizationMethod.sumprod.value, ArithmetizationMethod.sumgeomean.value]:
                s = np.nansum(arg_values)
                return min(1, s) if cap else s
            if arithmetization_mode.value == ArithmetizationMethod.arithmeangeomean.value:
                return np.nanmean(arg_values)
            if arithmetization_mode.value == ArithmetizationMethod.maxmin.value:
                return np.nanmax(arg_values)
        elif isinstance(expr, sympy.And):
            if arithmetization_mode.value == ArithmetizationMethod.sumprod.value:
                return np.nanprod(arg_values)
            if arithmetization_mode.value in [ArithmetizationMethod.sumgeomean.value,
                                              ArithmetizationMethod.arithmeangeomean.value]:
                return np.exp(np.nanmean(np.log(arg_values))) if (np.nanprod(arg_values) != 0) else 0
            if arithmetization_mode.value == ArithmetizationMethod.maxmin.value:
                return np.nanmin(arg_values)
        elif isinstance(expr, sympy.Not):
            raise ValueError("cobrapy doesn't support 'not' rules")
            # return 1 - arg_values[0]
        else:
            raise ValueError("unkown sympy expression type for {}".format(expr))


def discretize_values(values, num_bins, binarize_extremes=True, center=False):
    """
    Discretize a set of values using quantiles. Divides the values to num_bins bins of almost equal number of
    values in them, defined by (num_bins - 1) quantiles, and represent each value by the bin it belongs to,
    with quantiles themselves assigned the bin to their right.
    A number is assigned to the bin of the largest quantile it's smaller or equal to.
    If a value serves as multiple quantiles, the highest one is chosen.
    Bin values are equidistant over the range [0,1], such that for k bins, the corresponding values are
    {1/(k+1), 2/(k+1), ..., k/(k+1)}.
    See en.wikipedia.org/wiki/Quantile for how quantiles were defined, especially w.r.t. repeating values.
    If binarize_extreme is True, the bins are instead {0, 1/(k-1), ..., 1}.
    If center is True, all values are shifted by -0.5 and multiplied by 2.
    :param values: List-like, values assumed in range [0,\inf)
    :param num_bins:
    :param binarize_extremes:
    :return:
    """
    assert num_bins > 1
    assert int(num_bins) == num_bins
    # TODO: discretize to fit a normal, rather than uniform, distribution.
    quantile_values = np.quantile(values,
                                  [float(i + 1) / num_bins for i in range(num_bins - 1)],
                                  interpolation='lower')
    bins = np.digitize(values, quantile_values)
    index_to_val = lambda k: (k + 1) / (num_bins + 1.0)
    discrete_values = np.array([index_to_val(bin_index) for bin_index in bins])
    if binarize_extremes:
        discrete_values -= 1 / (num_bins + 1)
        discrete_values *= (num_bins + 1) / (num_bins - 1)
    if center:
        discrete_values -= 0.5
        discrete_values *= 2
    return discrete_values
