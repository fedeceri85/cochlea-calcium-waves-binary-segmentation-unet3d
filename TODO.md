# TODO


## 1. Standardize intensity normalization.
- Currently, inference and validation use different methods (global vs per-patch normalisation)

## 2. Add event-level evaluation

- Measure the fraction of annotated waves detected.
- Count false-positive wave instances.


## 3. Improve patch sampling

- Test a non-zero `min_foreground_fraction`.
- Add explicit sampling of wave boundaries.


## 4. Review the loss function

- Resolve the unused `loss.pos_weight` configuration value.
- Do not enable the current value of `20.0` without a controlled comparison.
- Compare the current Dice + BCE loss with measured class weighting and focal BCE.
- Consider a boundary-aware loss if jagged edges remain after normalization is corrected.



