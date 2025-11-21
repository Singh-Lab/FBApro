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

## See "example" for toy models and a notebook of running FBApro variants on them.
