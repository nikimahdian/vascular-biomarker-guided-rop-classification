**Primary paired statistics on the canonical test set (paired class-stratified bootstrap, seed 42, 10,000 replicates, N = 1,331). C vs B is from Task 7; E vs B and G vs E are the 10,000-replicate Task 8B closure.**

| Comparison | Metric | Delta | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|---|
| C vs B (biomarkers on RGB embedding) | AUC | 0.0011 | [-0.001691, +0.003868] | 0.4313 | no |
| C vs B (biomarkers on RGB embedding) | Balanced accuracy | -0.0030 | [-0.019805, +0.012895] | 0.7171 | no |
| C vs B (biomarkers on RGB embedding) | Macro F1 | 0.0007 | [-0.015127, +0.015549] | 0.9258 | no |
| C vs B (biomarkers on RGB embedding) | Brier | -0.0073 | [-0.015464, +0.000865] | 0.0819 | no |
| C vs B (biomarkers on RGB embedding) | ECE | -0.0043 | [-0.014093, +0.005995] | 0.4030 | no |
| E vs B (vessel on RGB embedding) | AUC | 0.0079 | [+0.002630, +0.013131] | 0.0031 | yes |
| E vs B (vessel on RGB embedding) | Balanced accuracy | 0.0116 | [-0.011994, +0.034846] | 0.3278 | no |
| E vs B (vessel on RGB embedding) | Macro F1 | 0.0159 | [-0.006211, +0.037655] | 0.1530 | no |
| E vs B (vessel on RGB embedding) | Brier | -0.0277 | [-0.043954, -0.011471] | 0.0011 | yes |
| E vs B (vessel on RGB embedding) | ECE | -0.0107 | [-0.026072, +0.003869] | 0.1642 | no |
| G vs E (biomarkers after vessel) | AUC | 0.0020 | [-0.000274, +0.004397] | 0.0900 | no |
| G vs E (biomarkers after vessel) | Balanced accuracy | 0.0038 | [-0.012053, +0.020061] | 0.6383 | no |
| G vs E (biomarkers after vessel) | Macro F1 | 0.0046 | [-0.010536, +0.019581] | 0.5440 | no |
| G vs E (biomarkers after vessel) | Brier | -0.0021 | [-0.009236, +0.004947] | 0.5707 | no |
| G vs E (biomarkers after vessel) | ECE | -0.0001 | [-0.009202, +0.011057] | 0.9906 | no |
