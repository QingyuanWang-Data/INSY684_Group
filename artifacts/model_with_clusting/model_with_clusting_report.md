# Model With Clusting Report

## Best Overall By Validation ROC-AUC
- Feature set: `selected_original_features`
- Model: `lightgbm_focal`
- Validation ROC-AUC: `0.761652`
- Test ROC-AUC: `0.769211`
- Train ROC-AUC: `0.948843`
- Best iteration: `227`

## Split Counts
- Train rows: `35000`
- Validation rows: `10000`
- Test rows: `5000`

## LightGBM Feature Selection
LightGBM focal-loss feature importance was fit on the training split with early stopping on validation data. Top 20, 50, 70, and 100 subsets were selected by validation ROC-AUC, matching the original train/valid/test logic.

| Top K | Train AUC | Validation AUC | Test AUC | Best iteration |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.878530 | 0.745029 | 0.749787 | 165 |
| 50 | 0.914481 | 0.754302 | 0.762022 | 185 |
| 70 | 0.948598 | 0.758625 | 0.764880 | 271 |
| 100 | 0.961494 | 0.757926 | 0.768197 | 303 |
| 125 | 0.966408 | 0.758914 | 0.766116 | 318 |
| 150 | 0.963517 | 0.757865 | 0.766491 | 294 |
| 200 | 0.970880 | 0.757736 | 0.767552 | 320 |
| 250 | 0.964486 | 0.760854 | 0.769900 | 287 |
| 300 | 0.948843 | 0.761652 | 0.769211 | 227 |
| 400 | 0.970677 | 0.757622 | 0.771810 | 320 |
| 450 | 0.955831 | 0.761289 | 0.768791 | 253 |

- Selected top-k: `300`
- Selected features:

  - `EXT_SOURCE_2`
  - `EXT_SOURCE_3`
  - `EXT_SOURCE_1`
  - `CREDIT_ANNUITY_RATIO`
  - `DAYS_EMPLOYED`
  - `DAYS_BIRTH`
  - `INST_AMT_PAYMENT_min`
  - `DAYS_LAST_PHONE_CHANGE`
  - `NAME_EDUCATION_TYPE`
  - `AMT_ANNUITY`
  - `AMT_CREDIT`
  - `DAYS_EMPLOYED_PCT`
  - `ANNUITY_INCOME_RATIO`
  - `BUREAU_AMT_CREDIT_SUM_DEBT_mean`
  - `PREV_HOUR_APPR_PROCESS_START_mean`
  - `AMT_GOODS_PRICE`
  - `CODE_GENDER`
  - `CREDIT_INCOME_RATIO`
  - `BUREAU_DAYS_CREDIT_ENDDATE_sum`
  - `DAYS_ID_PUBLISH`
  - `INST_DAYS_INSTALMENT_std`
  - `POS_CNT_INSTALMENT_FUTURE_mean`
  - `OWN_CAR_AGE`
  - `PREV_CNT_PAYMENT_mean`
  - `BUREAU_DAYS_CREDIT_ENDDATE_mean`
  - `PREV_CNT_PAYMENT_std`
  - `PREV_SELLERPLACE_AREA_mean`
  - `POS_CNT_INSTALMENT_FUTURE_std`
  - `POS_MONTHS_BALANCE_std`
  - `PREV_HOUR_APPR_PROCESS_START_std`
  - `BUREAU_AMT_CREDIT_SUM_max`
  - `BUREAU_AMT_CREDIT_SUM_std`
  - `BUREAU_AMT_CREDIT_SUM_DEBT_std`
  - `POS_SK_DPD_DEF_mean`
  - `BUREAU_DAYS_CREDIT_ENDDATE_std`
  - `BUREAU_DAYS_CREDIT_ENDDATE_max`
  - `BUREAU_DAYS_CREDIT_max`
  - `BUREAU_DAYS_CREDIT_std`
  - `BUREAU_AMT_CREDIT_SUM_min`
  - `PREV_DAYS_DECISION_max`
  - `INST_AMT_PAYMENT_sum`
  - `BUREAU_AMT_CREDIT_SUM_mean`
  - `INST_NUM_INSTALMENT_VERSION_std`
  - `PREV_RATE_DOWN_PAYMENT_mean`
  - `DAYS_REGISTRATION`
  - `PREV_HOUR_APPR_PROCESS_START_min`
  - `PREV_DAYS_TERMINATION_std`
  - `POS_CNT_INSTALMENT_std`
  - `INST_NUM_INSTALMENT_VERSION_mean`
  - `PREV_SELLERPLACE_AREA_min`
  - `PREV_CNT_PAYMENT_sum`
  - `INST_AMT_INSTALMENT_max`
  - `BUREAU_AMT_CREDIT_SUM_sum`
  - `PREV_SELLERPLACE_AREA_sum`
  - `BUREAU_DAYS_ENDDATE_FACT_max`
  - `REGION_POPULATION_RELATIVE`
  - `NAME_INCOME_TYPE`
  - `INST_AMT_INSTALMENT_std`
  - `PREV_AMT_ANNUITY_min`
  - `INST_AMT_PAYMENT_max`
  - `PREV_DAYS_LAST_DUE_std`
  - `CC_CNT_DRAWINGS_ATM_CURRENT_std`
  - `PREV_DAYS_DECISION_mean`
  - `BUREAU_DAYS_CREDIT_mean`
  - `PREV_RATE_DOWN_PAYMENT_std`
  - `BUREAU_DAYS_CREDIT_UPDATE_std`
  - `PREV_DAYS_DECISION_std`
  - `ORGANIZATION_TYPE`
  - `BUREAU_AMT_CREDIT_SUM_DEBT_sum`
  - `BUREAU_AMT_CREDIT_MAX_OVERDUE_mean`
  - `OCCUPATION_TYPE`
  - `POS_CNT_INSTALMENT_mean`
  - `BUREAU_DAYS_CREDIT_UPDATE_max`
  - `PREV_AMT_ANNUITY_std`
  - `BUREAU_DAYS_CREDIT_min`
  - `BUREAU_DAYS_CREDIT_UPDATE_sum`
  - `CC_CNT_DRAWINGS_ATM_CURRENT_mean`
  - `INST_NUM_INSTALMENT_NUMBER_std`
  - `INST_AMT_PAYMENT_mean`
  - `BUREAU_DAYS_CREDIT_UPDATE_mean`
  - `PREV_SELLERPLACE_AREA_max`
  - `INST_AMT_PAYMENT_std`
  - `INST_DAYS_ENTRY_PAYMENT_std`
  - `PREV_AMT_GOODS_PRICE_min`
  - `BUREAU_DAYS_CREDIT_UPDATE_min`
  - `BUREAU_DAYS_CREDIT_sum`
  - `INST_DAYS_ENTRY_PAYMENT_max`
  - `BUREAU_DAYS_CREDIT_ENDDATE_min`
  - `PREV_RATE_DOWN_PAYMENT_sum`
  - `INST_AMT_INSTALMENT_min`
  - `CC_CNT_DRAWINGS_CURRENT_std`
  - `BUREAU_DAYS_ENDDATE_FACT_std`
  - `INST_NUM_INSTALMENT_VERSION_sum`
  - `POS_MONTHS_BALANCE_mean`
  - `NAME_FAMILY_STATUS`
  - `PREV_DAYS_FIRST_DUE_max`
  - `INST_AMT_INSTALMENT_mean`
  - `BUREAU_DAYS_ENDDATE_FACT_mean`
  - `POS_SK_DPD_mean`
  - `PREV_AMT_ANNUITY_max`
  - `AMT_INCOME_TOTAL`
  - `FLAG_DOCUMENT_3`
  - `INST_NUM_INSTALMENT_NUMBER_mean`
  - `PREV_HOUR_APPR_PROCESS_START_sum`
  - `PREV_AMT_CREDIT_mean`
  - `INST_NUM_INSTALMENT_NUMBER_sum`
  - `PREV_SELLERPLACE_AREA_std`
  - `PREV_RATE_DOWN_PAYMENT_max`
  - `YEARS_BEGINEXPLUATATION_MEDI`
  - `PREV_DAYS_FIRST_DUE_std`
  - `BUREAU_AMT_CREDIT_MAX_OVERDUE_sum`
  - `PREV_AMT_GOODS_PRICE_std`
  - `PREV_DAYS_LAST_DUE_1ST_VERSION_sum`
  - `PREV_DAYS_LAST_DUE_mean`
  - `INST_AMT_INSTALMENT_sum`
  - `PREV_AMT_DOWN_PAYMENT_std`
  - `PREV_AMT_ANNUITY_sum`
  - `PREV_CNT_PAYMENT_max`
  - `PREV_AMT_ANNUITY_mean`
  - `PREV_HOUR_APPR_PROCESS_START_max`
  - `POS_CNT_INSTALMENT_FUTURE_sum`
  - `POS_SK_DPD_DEF_std`
  - `BUREAU_AMT_CREDIT_SUM_DEBT_max`
  - `POS_MONTHS_BALANCE_sum`
  - `BUREAU_BAL_STATUS_NUM_std`
  - `POS_COUNT`
  - `PREV_AMT_DOWN_PAYMENT_sum`
  - `PREV_DAYS_FIRST_DUE_sum`
  - `INST_DAYS_INSTALMENT_mean`
  - `PREV_AMT_CREDIT_sum`
  - `BUREAU_AMT_CREDIT_MAX_OVERDUE_max`
  - `NONLIVINGAREA_MODE`
  - `POS_CNT_INSTALMENT_sum`
  - `BUREAU_DAYS_ENDDATE_FACT_sum`
  - `PREV_AMT_DOWN_PAYMENT_mean`
  - `BUREAU_BAL_COUNT`
  - `BUREAU_DAYS_ENDDATE_FACT_min`
  - `FLAG_OWN_CAR`
  - `INST_DAYS_INSTALMENT_max`
  - `PREV_RATE_DOWN_PAYMENT_min`
  - `PREV_AMT_DOWN_PAYMENT_max`
  - `PREV_AMT_CREDIT_min`
  - `INST_DAYS_ENTRY_PAYMENT_sum`
  - `INST_DAYS_ENTRY_PAYMENT_mean`
  - `INST_NUM_INSTALMENT_NUMBER_max`
  - `POS_CNT_INSTALMENT_min`
  - `PREV_AMT_APPLICATION_min`
  - `PREV_AMT_CREDIT_max`
  - `HOUR_APPR_PROCESS_START`
  - `BUREAU_BAL_MONTHS_BALANCE_mean`
  - `PREV_DAYS_TERMINATION_sum`
  - `NONLIVINGAREA_MEDI`
  - `PREV_DAYS_LAST_DUE_sum`
  - `INST_COUNT`
  - `PREV_DAYS_LAST_DUE_1ST_VERSION_std`
  - `TOTALAREA_MODE`
  - `PREV_DAYS_FIRST_DUE_mean`
  - `PREV_AMT_GOODS_PRICE_mean`
  - `PREV_AMT_CREDIT_std`
  - `PREV_DAYS_LAST_DUE_min`
  - `BUREAU_AMT_CREDIT_MAX_OVERDUE_std`
  - `LIVINGAREA_AVG`
  - `PREV_DAYS_TERMINATION_min`
  - `NONLIVINGAREA_AVG`
  - `PREV_DAYS_DECISION_sum`
  - `INST_DAYS_INSTALMENT_sum`
  - `PREV_AMT_APPLICATION_std`
  - `PREV_DAYS_LAST_DUE_1ST_VERSION_min`
  - `CC_AMT_DRAWINGS_CURRENT_std`
  - `BASEMENTAREA_MODE`
  - `DEF_30_CNT_SOCIAL_CIRCLE`
  - `PREV_NFLAG_INSURED_ON_APPROVAL_mean`
  - `BUREAU_BAL_MONTHS_BALANCE_std`
  - `PREV_DAYS_LAST_DUE_1ST_VERSION_mean`
  - `BUREAU_AMT_ANNUITY_mean`
  - `WEEKDAY_APPR_PROCESS_START`
  - `PREV_DAYS_FIRST_DRAWING_sum`
  - `BUREAU_AMT_CREDIT_SUM_DEBT_min`
  - `CC_AMT_PAYMENT_CURRENT_max`
  - `PREV_DAYS_TERMINATION_mean`
  - `PREV_NFLAG_INSURED_ON_APPROVAL_std`
  - `PREV_AMT_APPLICATION_max`
  - `LANDAREA_MODE`
  - `BUREAU_BAL_STATUS_NUM_mean`
  - `NAME_HOUSING_TYPE`
  - `BUREAU_AMT_ANNUITY_sum`
  - `YEARS_BEGINEXPLUATATION_MODE`
  - `PREV_AMT_APPLICATION_sum`
  - `INST_DAYS_INSTALMENT_min`
  - `YEARS_BUILD_MEDI`
  - `POS_SK_DPD_DEF_sum`
  - `POS_MONTHS_BALANCE_max`
  - `PREV_DAYS_DECISION_min`
  - `CC_CNT_DRAWINGS_CURRENT_max`
  - `LIVINGAREA_MEDI`
  - `PREV_AMT_DOWN_PAYMENT_min`
  - `POS_MONTHS_BALANCE_min`
  - `YEARS_BEGINEXPLUATATION_AVG`
  - `LIVINGAREA_MODE`
  - `PREV_DAYS_LAST_DUE_1ST_VERSION_max`
  - `OBS_60_CNT_SOCIAL_CIRCLE`
  - `ENTRANCES_AVG`
  - `PREV_AMT_APPLICATION_mean`
  - `LANDAREA_AVG`
  - `POS_SK_DPD_std`
  - `INST_DAYS_ENTRY_PAYMENT_min`
  - `CC_AMT_CREDIT_LIMIT_ACTUAL_std`
  - `CC_AMT_CREDIT_LIMIT_ACTUAL_mean`
  - `PREV_DAYS_LAST_DUE_max`
  - `APARTMENTS_AVG`
  - `PREV_DAYS_TERMINATION_max`
  - `APARTMENTS_MODE`
  - `BUREAU_AMT_CREDIT_SUM_LIMIT_mean`
  - `CC_AMT_CREDIT_LIMIT_ACTUAL_sum`
  - `BASEMENTAREA_AVG`
  - `BUREAU_AMT_CREDIT_SUM_LIMIT_std`
  - `CC_SK_DPD_std`
  - `CNT_FAM_MEMBERS`
  - `LIVINGAPARTMENTS_MODE`
  - `AMT_REQ_CREDIT_BUREAU_YEAR`
  - `PREV_AMT_GOODS_PRICE_max`
  - `CC_AMT_RECEIVABLE_PRINCIPAL_mean`
  - `BUREAU_BAL_MONTHS_BALANCE_sum`
  - `PREV_CNT_PAYMENT_min`
  - `BUREAU_AMT_ANNUITY_max`
  - `PREV_DAYS_FIRST_DUE_min`
  - `LIVINGAPARTMENTS_MEDI`
  - `BUREAU_BAL_STATUS_NUM_sum`
  - `CC_AMT_CREDIT_LIMIT_ACTUAL_min`
  - `CC_AMT_BALANCE_mean`
  - `OBS_30_CNT_SOCIAL_CIRCLE`
  - `CC_CNT_INSTALMENT_MATURE_CUM_min`
  - `LANDAREA_MEDI`
  - `PREV_NFLAG_LAST_APPL_IN_DAY_sum`
  - `CC_CNT_DRAWINGS_CURRENT_mean`
  - `YEARS_BUILD_MODE`
  - `LIVINGAPARTMENTS_AVG`
  - `CC_AMT_DRAWINGS_CURRENT_max`
  - `CC_AMT_DRAWINGS_ATM_CURRENT_sum`
  - `CC_AMT_PAYMENT_CURRENT_min`
  - `NAME_CONTRACT_TYPE`
  - `CC_AMT_PAYMENT_CURRENT_sum`
  - `CC_AMT_DRAWINGS_POS_CURRENT_sum`
  - `BASEMENTAREA_MEDI`
  - `CC_AMT_DRAWINGS_ATM_CURRENT_std`
  - `BUREAU_AMT_ANNUITY_min`
  - `NAME_TYPE_SUITE`
  - `BUREAU_COUNT`
  - `BUREAU_AMT_ANNUITY_std`
  - `COMMONAREA_MEDI`
  - `POS_SK_DPD_sum`
  - `DEF_60_CNT_SOCIAL_CIRCLE`
  - `BUREAU_AMT_CREDIT_SUM_LIMIT_max`
  - `CC_AMT_INST_MIN_REGULARITY_std`
  - `FLOORSMAX_MODE`
  - `CC_AMT_PAYMENT_CURRENT_std`
  - `NONLIVINGAPARTMENTS_MEDI`
  - `POS_CNT_INSTALMENT_max`
  - `BUREAU_AMT_CREDIT_SUM_OVERDUE_sum`
  - `CC_AMT_DRAWINGS_CURRENT_mean`
  - `CC_AMT_INST_MIN_REGULARITY_mean`
  - `YEARS_BUILD_AVG`
  - `CC_CNT_DRAWINGS_POS_CURRENT_sum`
  - `CNT_CHILDREN`
  - `BUREAU_AMT_CREDIT_MAX_OVERDUE_min`
  - `CC_AMT_DRAWINGS_CURRENT_sum`
  - `CC_AMT_PAYMENT_CURRENT_mean`
  - `BUREAU_CNT_CREDIT_PROLONG_mean`
  - `CC_MONTHS_BALANCE_mean`
  - `CC_CNT_DRAWINGS_ATM_CURRENT_sum`
  - `FLAG_EMAIL`
  - `CC_AMT_INST_MIN_REGULARITY_min`
  - `CC_AMT_PAYMENT_TOTAL_CURRENT_sum`
  - `ENTRANCES_MEDI`
  - `AMT_REQ_CREDIT_BUREAU_QRT`
  - `COMMONAREA_MODE`
  - `PREV_COUNT`
  - `COMMONAREA_AVG`
  - `FLOORSMIN_MEDI`
  - `CC_MONTHS_BALANCE_sum`
  - `POS_CNT_INSTALMENT_FUTURE_max`
  - `NONLIVINGAPARTMENTS_AVG`
  - `CC_CNT_DRAWINGS_ATM_CURRENT_max`
  - `CC_AMT_PAYMENT_TOTAL_CURRENT_std`
  - `CC_AMT_PAYMENT_TOTAL_CURRENT_max`
  - `POS_SK_DPD_max`
  - `WALLSMATERIAL_MODE`
  - `ENTRANCES_MODE`
  - `FLOORSMAX_MEDI`
  - `CC_AMT_CREDIT_LIMIT_ACTUAL_max`
  - `CC_CNT_INSTALMENT_MATURE_CUM_sum`
  - `CC_AMT_DRAWINGS_ATM_CURRENT_max`
  - `CC_CNT_INSTALMENT_MATURE_CUM_std`
  - `ELEVATORS_AVG`
  - `POS_CNT_INSTALMENT_FUTURE_min`
  - `INST_NUM_INSTALMENT_VERSION_max`
  - `CC_CNT_DRAWINGS_POS_CURRENT_std`
  - `CC_MONTHS_BALANCE_std`
  - `CC_CNT_INSTALMENT_MATURE_CUM_mean`
  - `CC_AMT_DRAWINGS_POS_CURRENT_max`

## Final Model Comparison
Selected original features were used in three feature sets: selected only, selected plus PCA-2 clusting features, and clusting only.

| Feature set | Model | Train AUC | Validation AUC | Test AUC | Features | Best iteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| selected_original_features | lightgbm_focal | 0.948843 | 0.761652 | 0.769211 | 300 | 227 |
| selected_original_plus_clusting | lightgbm_focal | 0.972674 | 0.759619 | 0.769049 | 319 | 324 |
| selected_original_features | xgboost | 0.920935 | 0.756014 | 0.768000 | 300 | 345 |
| selected_original_plus_clusting | xgboost | 0.940451 | 0.755677 | 0.758815 | 319 | 466 |
| selected_original_features | logistic_regression | 0.788934 | 0.747727 | 0.744506 | 300 |  |
| selected_original_features | random_forest | 0.957724 | 0.747244 | 0.749401 | 300 |  |
| selected_original_plus_clusting | logistic_regression | 0.789447 | 0.746803 | 0.742631 | 319 |  |
| selected_original_plus_clusting | random_forest | 0.957928 | 0.746574 | 0.745753 | 319 |  |
| clusting_only | xgboost | 0.623351 | 0.585407 | 0.563922 | 19 | 25 |
| clusting_only | lightgbm_focal | 0.640612 | 0.577974 | 0.565578 | 19 | 9 |
| clusting_only | logistic_regression | 0.580231 | 0.571944 | 0.544611 | 19 |  |
| clusting_only | random_forest | 0.745026 | 0.558326 | 0.557765 | 19 |  |

## Best By Feature Set
- `selected_original_features`: `lightgbm_focal` validation AUC `0.761652`, test AUC `0.769211`
- `selected_original_plus_clusting`: `lightgbm_focal` validation AUC `0.759619`, test AUC `0.769049`
- `clusting_only`: `xgboost` validation AUC `0.585407`, test AUC `0.563922`

## Imbalance And Early Stopping Logic
- LightGBM uses the same custom focal-loss objective pattern as the old file.
- LightGBM uses validation early stopping and reports best iteration.
- Random Forest and Logistic Regression use balanced class weights.
- XGBoost uses positive-class sample weights and validation early stopping.

## Clusting Features Added
The clusting feature set contains PCA-2 component values plus one-hot labels from KMeans, Gaussian Mixture, and HDBSCAN.

```json
{
  "pca_components": 2,
  "pca_explained_variance_ratio": [
    0.08670135587453842,
    0.05259951576590538
  ],
  "pca_explained_variance_ratio_sum": 0.1393008679151535,
  "kmeans_clusters": 2,
  "gmm_components": 5,
  "hdbscan_min_cluster_size": 50,
  "hdbscan_min_samples": null,
  "cluster_counts": {
    "kmeans": {
      "0": 44886,
      "1": 5114
    },
    "gaussian_mixture": {
      "0": 13545,
      "1": 2351,
      "2": 15005,
      "3": 9542,
      "4": 9557
    },
    "hdbscan": {
      "-1": 12079,
      "0": 76,
      "1": 54,
      "2": 66,
      "3": 63,
      "4": 87,
      "5": 50,
      "6": 35917,
      "7": 1498,
      "8": 110
    }
  },
  "note": "PCA-2 and clustering features are fit without TARGET. The labels are added as one-hot features plus PCA component values."
}
```

## Files
- Results CSV: `artifacts\model_with_clusting\model_with_clusting_results.csv`
- Feature importance CSV: `artifacts\model_with_clusting\lightgbm_feature_importance.csv`
- Best model artifact: `artifacts\model_with_clusting\best_model_with_clusting.joblib`
