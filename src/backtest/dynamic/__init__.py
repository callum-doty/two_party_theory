"""
Paper II — sequential/dynamic capital allocation architecture.

This package is purely additive to the static, retrospective backtest in
src/backtest/{data,estimation,model,optimizer,comparison,validation,outputs}
(Paper I). Nothing in those packages is modified; every module here reuses
Paper I's estimator, sigma model, and optimizer as fixed subroutines. See
docs/paper2_draft.md for the conceptual framework this package implements.
"""
