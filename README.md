# FBApro
A linear-transformation based framework for integrating data with constraint based metabolic models. 
Implemented in Python as pytorch.nn.Module subclasses.
All methods implemented in one class, with name generated depending on special cases of input values: FBAproFull, FBAproPartial, FBAproFixed, FBAproBasic.

# Usage snippet

% a cobrapy metabolic model with a stoichiometric matrix, 

% alternatively can feed a metabolites X reactions stoichiometric matrix (numpy array / torch tensor).

model = SOME_MODEL_FILE

% a samplex X reactions (numpy array / torch tensor)

data = SOME_DATA_MATRIX 
unknown_indices = LIST_OF_REACTION_INDICES
measured_indices = DISJOINT_LIST_OF_REACTION_INDICES

projection = FBAprojection(model, measured_indices=measured_indices, unknown_indices=unknown_indices)

% samples X reactions, each row is the closest row in ker(S) to the corresponding row of data.

steadied_states = projection.forward(data) 

# See "example" for toy models and a notebook with examples of running FBApro variants on different inputs on these models.

# Real and simulated data reproduction
## Synthetic data
synthetic_data_basis.ipynb, synthetic_data_cobrapy.ipynb, synthetic_data_randomfba.ipynb all generate synthetic steady-state fluxes from given metabolic models, and analyze the runtime and performance of FBApro variants and benchmarks on them, with different data generation methods. To recreate paper figures, models need to be sourced and placed in synthetic_data_experiment_files/data (see instructions there).

## Real data
real_data_run.ipynb runs FBApro variants and benchmark on a given model, GE data and flux data and outputs predictions. real_data_plot.ipynb reads predictions and cached processed data and plots performance of methods. To recreate paper figures, data need to be sourced and placed in real_data_experiment_files/data (see instructions there). 

paper_figures.ipynb generates figures separately using cached outputs from both synthetic and real data experiments used in the analysis described in the paper.
