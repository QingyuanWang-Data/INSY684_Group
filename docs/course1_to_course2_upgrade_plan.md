# Course 1 to Course 2 Upgrade Plan

## Purpose

This document explains how the Course 2 project builds on the previous Course 1 Home Credit
Default Risk project. The goal is to make the work more complete from a production machine
learning, business decision-making, and governance perspective.

Course 1 showed that the dataset can support meaningful default risk prediction. Course 2
should focus on making the project easier to track, explain, monitor, deploy, and connect to
business policy.

## Course 1 Baseline

The previous project already established a useful foundation:

| Area | Course 1 baseline |
| --- | --- |
| Use case | Predict loan payment difficulty for Home Credit applicants |
| Dataset | Home Credit Default Risk dataset from Kaggle |
| Target | `TARGET = 1` means payment difficulty; `TARGET = 0` means normal repayment |
| Model | LightGBM binary classification model |
| Main metric | ROC-AUC |
| Reported result | Test ROC-AUC around `0.7858` |
| Feature engineering | Applicant-level features plus aggregated historical credit behavior |
| Service | FastAPI prediction service |
| Explainability | Feature importance and local prediction contributors |
| Validation | Train/validation/test split, threshold notes, calibration, subgroup metrics, drift checks |
| Engineering | Dockerfile, tests, GitHub Actions, dependency files |

In business terms, Course 1 answered the question:

> Can applicant and credit history data help rank applicants by default risk?

The answer was yes. The model produced a risk score that could support underwriting and credit
risk decisions.

## Current Course 2 Progress

The team has started implementing the Course 2 production-oriented upgrade on the technical
branch. Current evidence includes:

| Area | Current Course 2 progress |
| --- | --- |
| Experiment tracking | MLflow hooks and a sampled run summary are documented |
| Hyperparameter tuning | Optuna tuning produced a best sampled validation ROC-AUC of `0.7679` |
| Monitoring | PSI-based score and feature drift reports are generated |
| Fairness | Subgroup metrics are generated for gender, income type, education, family status, and contract type |
| Decision policy | Threshold `0.549` is used in current validation policy evidence |
| Deployment | FastAPI readiness, Docker runtime, and cloud-native deployment notes are documented |
| CI/CD | Quality and Docker build workflows are included |

The business work should now shift from only planning the upgrade to interpreting these outputs
for final reporting and presentation.

The latest technical branch also improves model generalization by adding categorical aggregate
features. The current integrated evidence reports validation ROC-AUC `0.7666`, test ROC-AUC
`0.7647`, train ROC-AUC `0.9160`, and about `700` model columns. This is a better production
story than focusing only on the older score, because the train/test gap is smaller and the
project now has stronger tracking, monitoring, and governance evidence.

An additional `Extend` branch explores PCA/clustering features and model comparisons. Its best
reported test ROC-AUC is `0.7692`, but the clustering-only feature set is weak. Treat this as
supplementary experimentation unless the team decides to integrate it into the final `main`
branch.

## Course 2 Improvement Goals

Course 2 should not only aim for a better model score. It should improve how the model is built,
tracked, governed, and used.

| Improvement area | Why it matters | Possible Course 2 work |
| --- | --- | --- |
| GitHub collaboration | The project should show clear teamwork and contribution history | Use branches, issues, pull requests, and meaningful commits |
| CI/CD | The team needs automated checks before merging changes | Add or maintain linting, tests, and workflow checks |
| Docker | The application should run consistently across machines | Confirm Docker build and run instructions |
| MLflow / tracking | Model runs should be comparable and reproducible | Log parameters, metrics, artifacts, and model versions |
| Hyperparameter tuning | Model improvement should be systematic | Tune LightGBM or compare candidate models |
| Explainability | Business users need to understand important risk drivers | Expand global and local explanation documentation |
| Fairness and ethical AI | Credit decisions can affect applicants unevenly | Compare model behavior across available subgroups |
| Drift monitoring | Future data may differ from training data | Track data drift, score drift, and performance drift |
| Deployment / serving | The model should be usable outside a notebook | Clarify API or app serving workflow |
| Business policy | Risk scores must translate into business actions | Define low-, medium-, and high-risk decision bands |

## Expected Final Story

The final project should tell this story:

1. We started from a working Course 1 credit risk model.
2. We reused the same Home Credit use case because it is business-relevant and technically rich.
3. We upgraded the project toward a more production-ready machine learning workflow.
4. We connected model outputs to lending policy through business metrics and decision bands.
5. We considered explainability, fairness, monitoring, and deployment risks.
6. We identified what would still need validation before real business use.
