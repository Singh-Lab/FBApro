import pandas as pd
import numpy as np
import time
import scipy
from enum import Enum

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

