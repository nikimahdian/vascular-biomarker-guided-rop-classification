**Architecture robustness of the spatial-vessel contribution. Each row adds the same frozen 1792-d vessel embedding to a different RGB representation, with its own matched RGB-only control; the two rows are independent experiments, not a single model comparison.**

| RGB representation | RGB-only AUC | RGB + vessel AUC | Delta AUC | 95% CI | p | Delta macro F1 | Delta Brier |
|---|---|---|---|---|---|---|---|
| Original single-backbone (EfficientNet-B5) | 0.9249 | 0.9328 | 0.0079 | [+0.002630, +0.013131] | 0.0031 | 0.0159 | -0.0277 |
| Dual-backbone attention (ResNet50 + EfficientNet-B4) | 0.9060 | 0.9269 | 0.0209 | [+0.011426, +0.030828] | <0.0001 | 0.0404 | -0.0492 |
