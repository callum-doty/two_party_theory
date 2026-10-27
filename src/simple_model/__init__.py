"""
A deliberately small replacement for the margin/ceiling/optimizer stack,
split into the two jobs the old model tangled together:

1. baseline.py -- how likely each race is to go Democratic before any
   change in party money. A forecasting problem, judged only by
   calibration on elections it was not fit to.
2. spending.py -- how much a party dollar moves a race's margin. A causal
   question public data cannot settle; here it is an explicit, swappable
   assumption with two parameters, reported across a range.

allocate.py combines them: p_i(x) = Phi((mu_i + f_i(x_i) - f_i(x_obs_i)) / sigma),
anchored so that DCCC's observed allocation reproduces the baseline forecast
exactly, and spends the budget greedily where the next chunk buys the most
expected seats. scripts/run_simple_model.py is the single entry point.
"""
