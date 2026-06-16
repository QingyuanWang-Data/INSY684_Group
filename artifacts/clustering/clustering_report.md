# Clustering Report

## PCA Information Retention
- PCA components closest to or reaching 90% explained variance: `123`
- PCA components used for clustering: `2`
- Explained variance kept by chosen PCA: `0.1445`

## Method Scores
| Method | Cluster n | Noise rate | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: | ---: |
| kmeans | 2 | 0.0000 | 0.6540 | 3750.8502 | 0.7991 |
| hdbscan | 2 | 0.3940 | 0.5613 | 565.4348 | 0.3240 |
| gaussian_mixture | 2 | 0.0000 | 0.4908 | 2773.1138 | 1.0580 |

## Best Silhouette Method
`kmeans` has the best available silhouette score among the three methods.

## Cluster Meanings
These meanings are based on the largest standardized differences from the overall numeric feature means.

- `gaussian_mixture` cluster `0`: lower CC_CNT_DRAWINGS_ATM_CURRENT_max; lower CC_AMT_BALANCE_max; lower CC_AMT_TOTAL_RECEIVABLE_max; lower CC_AMT_RECIVABLE_max
- `gaussian_mixture` cluster `1`: higher INST_COUNT; higher INST_AMT_INSTALMENT_sum; higher INST_AMT_PAYMENT_sum; higher PREV_AMT_CREDIT_max
- `hdbscan` cluster `-1`: higher PREV_DAYS_LAST_DUE_1ST_VERSION_max; higher PREV_DAYS_LAST_DUE_1ST_VERSION_sum; lower INST_NUM_INSTALMENT_VERSION_min; higher PREV_DAYS_LAST_DUE_1ST_VERSION_mean
- `hdbscan` cluster `0`: higher CC_AMT_CREDIT_LIMIT_ACTUAL_min; higher CC_AMT_CREDIT_LIMIT_ACTUAL_mean; lower CC_CNT_DRAWINGS_ATM_CURRENT_max; higher CC_AMT_CREDIT_LIMIT_ACTUAL_max
- `hdbscan` cluster `1`: higher CC_SK_DPD_max; higher CC_SK_DPD_std; higher CC_SK_DPD_sum; higher CC_SK_DPD_mean
- `kmeans` cluster `0`: lower CC_AMT_TOTAL_RECEIVABLE_max; lower CC_AMT_RECIVABLE_max; lower CC_AMT_BALANCE_max; lower CC_AMT_RECEIVABLE_PRINCIPAL_max
- `kmeans` cluster `1`: lower INST_NUM_INSTALMENT_VERSION_min; lower PREV_DAYS_FIRST_DRAWING_min; higher INST_COUNT; higher PREV_DAYS_FIRST_DRAWING_std

## Picture Meaning
- PCA scatter plots show applicants projected into two dimensions; nearby points have similar engineered feature patterns.
- Color is the cluster assignment. HDBSCAN label `-1` means noise/outlier applicants.
- The PCA information-lost plot shows how much original feature variance is discarded as component count changes.
