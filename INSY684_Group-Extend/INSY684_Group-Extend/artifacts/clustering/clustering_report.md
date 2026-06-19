# Clustering Report

## PCA Information Retention
- PCA components closest to or reaching 90% explained variance: `215`
- PCA components used for clustering: `2`
- Explained variance kept by chosen PCA: `0.1211`

## Method Scores
| Method | Cluster n | Noise rate | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: | ---: |
| kmeans | 2 | 0.0000 | 0.5210 | 37901.8945 | 0.9473 |
| hdbscan | 8 | 0.3364 | 0.1977 | 528.8057 | 0.5124 |
| gaussian_mixture | 5 | 0.0000 | 0.3844 | 21457.3077 | 1.0462 |

## Best Silhouette Method
`kmeans` has the best available silhouette score among the three methods.

## Cluster Meanings
These meanings are based on the largest standardized differences from the overall numeric feature means.

- `gaussian_mixture` cluster `0`: higher PREV_CAT_NAME_CLIENT_TYPE_NEW_SHARE; lower PREV_CAT_NAME_CLIENT_TYPE_REPEATER_SHARE; higher PREV_CAT_NAME_PORTFOLIO_POS_SHARE; lower PREV_CAT_NAME_GOODS_CATEGORY_XNA_SHARE
- `gaussian_mixture` cluster `1`: lower CC_CNT_DRAWINGS_ATM_CURRENT_max; lower CC_AMT_BALANCE_max; lower CC_AMT_RECIVABLE_max; lower CC_AMT_TOTAL_RECEIVABLE_max
- `gaussian_mixture` cluster `2`: lower CC_AMT_BALANCE_max; lower CC_AMT_TOTAL_RECEIVABLE_max; lower CC_AMT_RECIVABLE_max; lower CC_AMT_RECEIVABLE_PRINCIPAL_max
- `gaussian_mixture` cluster `3`: higher INST_AMT_INSTALMENT_sum; higher INST_AMT_PAYMENT_sum; higher PREV_CAT_NAME_GOODS_CATEGORY_XNA_COUNT; higher PREV_CAT_NAME_CLIENT_TYPE_REPEATER_COUNT
- `gaussian_mixture` cluster `4`: lower PREV_DAYS_FIRST_DRAWING_mean; lower PREV_DAYS_FIRST_DRAWING_min; lower INST_NUM_INSTALMENT_VERSION_min; higher PREV_DAYS_FIRST_DRAWING_std
- `hdbscan` cluster `-1`: higher PREV_DAYS_LAST_DUE_1ST_VERSION_max; higher PREV_DAYS_LAST_DUE_1ST_VERSION_sum; lower INST_NUM_INSTALMENT_VERSION_min; lower PREV_DAYS_FIRST_DRAWING_min
- `hdbscan` cluster `0`: lower INST_NUM_INSTALMENT_VERSION_min; lower PREV_DAYS_FIRST_DRAWING_min; higher INST_COUNT; higher INST_NUM_INSTALMENT_NUMBER_max
- `hdbscan` cluster `1`: lower INST_NUM_INSTALMENT_VERSION_min; lower PREV_DAYS_FIRST_DRAWING_min; higher PREV_DAYS_FIRST_DRAWING_std; lower PREV_DAYS_FIRST_DRAWING_mean
- `hdbscan` cluster `2`: lower PREV_DAYS_FIRST_DRAWING_mean; higher PREV_DAYS_LAST_DUE_1ST_VERSION_mean; lower PREV_DAYS_FIRST_DRAWING_min; higher PREV_DAYS_FIRST_DRAWING_std
- `hdbscan` cluster `3`: higher INST_NUM_INSTALMENT_NUMBER_mean; lower INST_NUM_INSTALMENT_VERSION_min; lower PREV_DAYS_FIRST_DRAWING_min; higher PREV_DAYS_FIRST_DRAWING_std
- `hdbscan` cluster `4`: higher PREV_CAT_NAME_CASH_LOAN_PURPOSE_XNA_COUNT; higher PREV_CAT_NAME_CONTRACT_TYPE_CASH_LOANS_COUNT; higher PREV_CAT_NAME_PORTFOLIO_CASH_COUNT; higher PREV_CAT_NAME_GOODS_CATEGORY_XNA_COUNT
- `hdbscan` cluster `5`: higher PREV_DAYS_FIRST_DRAWING_sum; higher POS_COUNT; higher POS_CAT_NAME_CONTRACT_STATUS_ACTIVE_COUNT; higher PREV_CAT_NAME_PORTFOLIO_CASH_COUNT
- `hdbscan` cluster `6`: lower CC_CNT_DRAWINGS_ATM_CURRENT_max; lower CC_AMT_DRAWINGS_ATM_CURRENT_max; lower CC_AMT_BALANCE_max; lower CC_AMT_RECIVABLE_max
- `hdbscan` cluster `7`: higher PREV_CAT_NAME_PRODUCT_TYPE_X_SELL_COUNT; higher PREV_CAT_NAME_PORTFOLIO_CASH_COUNT; higher PREV_AMT_CREDIT_max; higher PREV_AMT_APPLICATION_max
- `kmeans` cluster `0`: lower CC_AMT_TOTAL_RECEIVABLE_max; lower CC_AMT_RECIVABLE_max; lower CC_AMT_BALANCE_max; lower CC_AMT_RECEIVABLE_PRINCIPAL_max
- `kmeans` cluster `1`: higher PREV_CAT_NAME_GOODS_CATEGORY_XNA_COUNT; higher PREV_CAT_NAME_CLIENT_TYPE_REPEATER_COUNT; higher PREV_COUNT; higher PREV_CAT_NAME_PRODUCT_TYPE_X_SELL_COUNT

## Picture Meaning
- PCA scatter plots show applicants projected into two dimensions; nearby points have similar engineered feature patterns.
- Color is the cluster assignment. HDBSCAN label `-1` means noise/outlier applicants.
- The PCA information-lost plot shows how much original feature variance is discarded as component count changes.
