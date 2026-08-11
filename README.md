# FBApro
A linear-transformation based framework for integrating data with constraint based metabolic models. 
Implemented in Python as pytorch.nn.Module subclasses.
All methods implemented in one class, with name generated depending on special cases of input values: FBAproFull, FBAproPartial, FBAproFixed, FBAproBasic.

# Usage snippet

```python
import cobra
import torch
from projection_methods import FBApro

# FBApro takes a metabolites X reactions stoichiometric matrix (numpy array / torch tensor),
# From a cobrapy model, get it with create_stoichiometric_matrix.
model = cobra.io.read_sbml_model(SOME_MODEL_FILE)
S = cobra.util.create_stoichiometric_matrix(model)

# samples X reactions, dtype matching the projection's (torch.float64 by default)
data = torch.tensor(SOME_DATA_MATRIX, dtype=torch.float64)

unknown_indices = LIST_OF_REACTION_INDICES           # reactions to be ignored
measured_indices = DISJOINT_LIST_OF_REACTION_INDICES # reactions to be fixed

projection = FBApro(stoichiometric_matrix=S, measured_indices=measured_indices,
                    unknown_indices=unknown_indices, device=torch.device('cpu'), acond=1e-5)
print(projection.name)  # FBAproFull for this combination of arguments

# samples X reactions
steadied_states = projection(data)
```

The variant is chosen by which index lists are passed: neither gives FBAproBasic, `unknown_indices`
alone gives FBAproPartial, `measured_indices` alone gives FBAproFixed, and both give FBAproFull.

FBApro's internal matrices and linear algebra method calls are conditioned with an absolute threshold acond, and a relative threshold rcond: entries smaller than acond are zeroed, and similarly matrix singular values with relative weight of rcond compared to the maximal singular value. The default is acond=0, rcond=1e-3, appropriate values may need to be empirically determined for different models and data sets, typically a small nonzero acond (significantly smaller than all stoichiometric coefficients) is useful for measured_indices leaving zero degrees of freedom, and a small nonzero rcond for big models and numerically unstable stoichiometric matrices.

# See "example" for toy models and a notebook with examples of running FBApro variants on different inputs on these models.

# Real and simulated data reproduction
## Synthetic data
synthetic_data_basis.ipynb, synthetic_data_cobrapy.ipynb, synthetic_data_randomfba.ipynb all generate synthetic steady-state fluxes from given metabolic models, and analyze the runtime and performance of FBApro variants and benchmarks on them, with different data generation methods. To recreate paper figures, models need to be sourced and placed in synthetic_data_experiment_files/data (see instructions there).

## Real data
real_data_run.ipynb runs FBApro variants and benchmark on a given model, GE data and flux data and outputs predictions. real_data_plot.ipynb reads predictions and cached processed data and plots performance of methods. To recreate paper figures, data need to be sourced and placed in real_data_experiment_files/data (see instructions there). 

paper_figures.ipynb generates figures separately using cached outputs from both synthetic and real data experiments used in the analysis described in the paper.
