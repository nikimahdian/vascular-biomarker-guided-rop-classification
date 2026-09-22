**Secondary exploratory neural experiments that did not improve on the frozen concatenation models. Values are canonical test-set metrics; none of these models is a primary result.**

| Model | AUC | Balanced accuracy | Macro F1 | Brier | ECE | Interpretation |
|---|---|---|---|---|---|---|
| Joint RGB-vessel | 0.9258 | 0.6574 | 0.6840 | 0.2759 | 0.1159 | learned cross-modal interaction; selected stage-2 epoch 4 |
| Joint RGB-vessel + biomarkers | 0.9162 | 0.6730 | 0.6978 | 0.2786 | 0.1152 | as H plus biomarker branch; selected epoch 6 |
| Frozen fusion control | 0.9298 | 0.7954 | 0.6852 | 0.3770 | 0.1676 | frozen-embedding MLP control; selected epoch 1 |
| Biomarker-conditioned FiLM | 0.9276 | 0.7897 | 0.6885 | 0.3614 | 0.1536 | bounded residual FiLM; selected epoch 1 |
