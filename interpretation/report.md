# Model Interpretation Report

This report summarizes the performance and interpretability analyses of the top models.

## Metrics Summary

| model               |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|
| 01_RandomForest     |   0.983929 |    0.974937 | 1        | 0.98731  |  0.999523 |
| 02_GradientBoosting |   0.859783 |    0.84824  | 0.94473  | 0.893889 |  0.928054 |
| 03_AdaBoost         |   0.832061 |    0.817522 | 0.941517 | 0.875149 |  0.899351 |

## ROC Curves

![ROC Curves](roc_curves.png)

## 01_RandomForest

### Permutation Importance

![01_RandomForest Permutation Importance](01_RandomForest_perm_importance.png)

### SHAP Summary Plot

![01_RandomForest SHAP Summary](01_RandomForest_shap_summary.png)

### Partial Dependence Plot

![01_RandomForest PDP](01_RandomForest_pdp.png)

## 02_GradientBoosting

### Permutation Importance

![02_GradientBoosting Permutation Importance](02_GradientBoosting_perm_importance.png)

### Partial Dependence Plot

![02_GradientBoosting PDP](02_GradientBoosting_pdp.png)

## 03_AdaBoost

### Permutation Importance

![03_AdaBoost Permutation Importance](03_AdaBoost_perm_importance.png)

### Partial Dependence Plot

![03_AdaBoost PDP](03_AdaBoost_pdp.png)

