# FBApro
A linear-transformation based framework for integrating data with constraint based metabolic models. 
Implemented in Python as pytorch.nn.Module subclasses.
Includes three main methods: FBAprojection, FBAprojectionLowMid (aka FBAproPartial), FBAprojectionHighMid (akaFBAproFIxed).

# Usage snippet

% a cobrapy metabolic model with a stoichiometric matrix, 

% alternatively can feed a metabolites X reactions stoichiometric matrix (numpy array / torch tensor).

model = SOME_MODEL_FILE

% a samplex X reactions (numpy array / torch tensor)

data = SOME_DATA_MATRIX 

projection = FBAprojection(model)

% samples X reactions, each row is the closest row in ker(S) to the corresponding row of data.

steadied_states = projection.forward(data) 

# See "example" for toy models and a notebook with examples of running FBApro variants on different inputs on these models.

# Real and simulated data reproduction
## Synthetic data
synthetic_exact_noisy_data.ipynb, synthetic_noisy_missing_data.ipynb, synthetic_projections_timing.ipynb generate synthetic steady-state fluxes from given metabolic models, and analyze the runtime and performance of FBApro variants and benchmarks on them. To recreate paper figures, models need to be sourced and placed in synthetic_data_experiment_files/data (see instructions there).

## Real data
real_data_run.ipynb runs FBApro variants and benchmark on a given model, GE data and flux data and outputs predictions. real_data_plot.ipynb reads predictions and cached processed data and plots performance of methods. To recreate paper figures, data need to be sourced and placed in real_data_experiment_files/data (see instructions there).
