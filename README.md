# 🏦 Loan Risk Assessment System

A machine-learning based loan **non-payment risk assessment** application that estimates the probability that a loan may not be fully repaid and converts that probability into three interpretable risk categories:

- **Low Risk**
- **Medium Risk**
- **High Risk**

The project combines:

- Exploratory Data Analysis
- Financial and credit-feature analysis
- Logistic Regression
- Class-imbalance handling
- Hyperparameter tuning
- Probability calibration
- Risk-band segmentation
- Model explainability
- FastAPI
- Streamlit
- Automated testing

> **Important:** This is an educational/portfolio prototype. It is **not a production banking credit-decision engine** and does not directly approve or reject loans.

---

## 1. Project Overview

### Business Problem

Banks and financial institutions receive a large number of loan applications.

Each applicant can have multiple financial and credit-history characteristics such as:

- Annual income
- FICO score
- Debt-to-income ratio
- Credit utilization
- Recent credit inquiries
- Delinquencies
- Public records
- Loan purpose
- Interest rate
- Installment amount

The objective of this project is to use historical loan data to estimate an applicant's **risk of non-payment**.

Instead of making a direct:

> "Approve the loan" / "Reject the loan"

decision, the system provides:

1. An estimated probability of non-payment.
2. A risk category:
   - Low
   - Medium
   - High
3. Applicant-specific factors that contributed toward higher or lower modeled risk.

Therefore, the application is designed as a **risk-assessment support tool**, rather than an autonomous lending decision-maker.

---

## 2. What Does the Model Predict?

The original target variable is:

```text
not.fully.paid
```

where:

```text
0 → Loan was fully paid
1 → Loan was not fully paid
```

Therefore, the underlying machine-learning problem is a **binary classification problem**.

The final application converts the predicted probability into three risk bands.

### Risk bands used in this prototype

| Estimated non-payment probability | Risk category |
|---|---|
| `< 10%` | Low |
| `10% – < 20%` | Medium |
| `>= 20%` | High |

These thresholds are **prototype risk-segmentation thresholds**, not official banking policy or regulatory credit-risk thresholds.

---

## 3. Dataset

The dataset contains:

```text
9,578 records
14 original columns
```

The target distribution is:

```text
Fully paid      : 8,045 (83.99%)
Not fully paid  : 1,533 (16.01%)
```

Therefore, the non-payment class represents only about **16%** of the observations.

This class imbalance became an important consideration during model development and evaluation.

---

## 4. Dataset Features

| Feature | Meaning |
|---|---|
| `credit.policy` | Indicates whether the applicant meets the dataset's credit-policy condition |
| `purpose` | Purpose of the loan |
| `int.rate` | Interest rate |
| `installment` | Monthly loan installment |
| `log.annual.inc` | Logarithm of annual income |
| `dti` | Debt-to-income ratio |
| `fico` | FICO credit score |
| `days.with.cr.line` | Length of credit history in days |
| `revol.bal` | Revolving credit balance |
| `revol.util` | Revolving credit utilization |
| `inq.last.6mths` | Credit inquiries during the previous six months |
| `delinq.2yrs` | Delinquencies during the previous two years |
| `pub.rec` | Public credit records |
| `not.fully.paid` | Target: whether the loan was not fully paid |

---

## 5. Data Quality Checks

The initial data analysis included:

- Dataset shape
- Column names
- Data types
- Missing-value analysis
- Descriptive statistics
- Target distribution
- Duplicate-row check
- Categorical-value analysis
- Feature/target relationship analysis

### Missing Values

There were:

```text
0 missing values
```

across the dataset.

### Duplicate Rows

There were:

```text
0 duplicate rows
```

---

## 6. Exploratory Data Analysis

EDA was performed to understand whether the available financial and credit-history variables showed useful relationships with non-payment.

The purpose was not simply to generate charts.

The EDA was used to answer questions such as:

- Which financial characteristics are associated with higher historical non-payment?
- Which features appear useful for prediction?
- Are relationships approximately monotonic?
- Are there small groups where percentages could be misleading?
- Which features deserve further investigation?

---

## 7. Loan Purpose Analysis

Observed non-payment rates differed across loan purposes.

The dataset showed the following distribution of loan purposes:

| Loan purpose | Borrowers | Percentage |
|---|---:|---:|
| Debt consolidation | 3,957 | 41.31% |
| All other | 2,331 | 24.34% |
| Credit card | 1,262 | 13.18% |
| Home improvement | 629 | 6.57% |
| Small business | 619 | 6.46% |
| Major purchase | 437 | 4.56% |
| Educational | 343 | 3.58% |

Observed non-payment rates differed across these categories.

For example, the Logistic Regression later learned a relatively positive association for:

```text
purpose = small_business
```

and negative associations for some other categories relative to the reference category.

This suggested that `purpose` contains predictive information.

However, these values represent **historical associations in this dataset**.

They should not be interpreted as evidence that a particular loan purpose causes non-payment.

---

## 8. FICO Score Analysis

FICO showed one of the clearest relationships with historical non-payment.

The EDA showed:

| FICO band | Observed non-payment |
|---|---:|
| 600–659 | 30.88% |
| 660–699 | 19.37% |
| 700–739 | 15.29% |
| 740–779 | 8.78% |
| 780+ | 5.95% |

The observed non-payment rate decreased substantially as FICO increased.

This made FICO an important candidate feature for the model.

---

## 9. Debt-to-Income Ratio Analysis

Observed non-payment rates generally increased toward higher DTI ranges.

| DTI band | Observed non-payment |
|---|---:|
| 0–4.9 | 14.37% |
| 5–9.9 | 15.20% |
| 10–14.9 | 15.50% |
| 15–19.9 | 16.77% |
| 20–24.9 | 17.61% |
| 25–30 | 25.16% |

This suggested that higher debt burden contains useful predictive information.

The model nevertheless uses the original continuous DTI variable rather than replacing it with the EDA bins.

---

## 10. Recent Credit Inquiry Analysis

Recent credit inquiries showed useful predictive signal.

Examples from the data:

```text
0 inquiries → 11.74%
3 inquiries → 20.83%
5 inquiries → 27.34%
8 inquiries → 40.28%
```

However, some higher inquiry counts had very small sample sizes.

For example, some very high inquiry counts were represented by only a handful of borrowers.

Therefore, extreme percentages from tiny groups were **not treated as reliable standalone conclusions**.

The more important conclusion was that recent credit inquiries contain meaningful information about historical non-payment risk.

---

## 11. Delinquency Analysis

Observed non-payment rates generally increased across the common delinquency groups.

| Delinquencies | Borrowers | Observed non-payment |
|---|---:|---:|
| 0 | 8,458 | 15.81% |
| 1 | 832 | 17.43% |
| 2 | 192 | 17.71% |
| 3–4 | 84 | 20.24% |
| 5+ | 12 | 0.00% |

The `5+` group was extremely small.

Its observed 0% rate was therefore **not** interpreted as evidence that many delinquencies reduce risk.

This is an important EDA principle:

> Percentages calculated from very small groups should not automatically be treated as reliable population-level patterns.

---

## 12. Public Record Analysis

Applicants with at least one public record had a higher observed non-payment rate:

```text
No public record       → 15.47%
At least one record    → 24.69%
```

This suggested potentially useful predictive signal.

The model therefore retained `pub.rec` as a feature.

---

## 13. Credit Policy Analysis

The original `credit.policy` variable showed:

| Credit policy | Observed non-payment |
|---|---:|
| 0 | 27.78% |
| 1 | 13.15% |

This indicated that the variable contained substantial information about the target.

However, it was intentionally **excluded from the final modeling feature set**.

The reason is that the final application is intended to evaluate the applicant using the financial and credit characteristics supplied by the user, rather than relying on an existing policy decision from the historical dataset.

This also avoids making the model partially reproduce an earlier credit-policy decision.

---

## 14. Feature Engineering and Preprocessing

Temporary EDA grouping variables were created to understand the data, including:

```text
fico_band_eda
dti_band_eda
inq_group_eda
delinq_group_eda
has_public_record_eda
```

These variables were removed before model training.

This is intentional.

EDA-derived grouping variables were used for **understanding the data**, not automatically inserted into the final model.

The final model therefore uses the original feature representation.

---

## 15. Final Modeling Features

The final model uses 12 features:

```text
purpose
int.rate
installment
log.annual.inc
dti
fico
days.with.cr.line
revol.bal
revol.util
inq.last.6mths
delinq.2yrs
pub.rec
```

The target is:

```text
not.fully.paid
```

The original `credit.policy` field is not included in the final feature list.

---

## 16. Numerical Preprocessing

The numerical features are standardized using:

```python
StandardScaler()
```

This is particularly important for Logistic Regression because the numerical features have very different scales.

For example:

```text
FICO                  → hundreds
Revolving balance     → thousands
Credit history        → thousands of days
Interest rate         → decimal values
```

Standardization places numerical variables on comparable scales for model optimization.

---

## 17. Categorical Preprocessing

The `purpose` feature is categorical.

It is encoded using:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

### Why `drop="first"`?

For Logistic Regression, dropping one category avoids redundant dummy variables.

The dropped category becomes the reference category.

In this dataset:

```text
all_other
```

is the reference category.

### Why `handle_unknown="ignore"`?

It makes the preprocessing pipeline safer if an unseen category appears during inference.

---

## 18. User-Friendly Feature Transformations

The original dataset contains some modeling-specific variables that are not appropriate to expose directly to normal users.

For example:

```text
log.annual.inc
days.with.cr.line
```

The application therefore separates:

```text
User-facing representation
```

from:

```text
Model representation
```

This is an important application-engineering decision.

---

## Annual Income

The dataset contains:

```text
log.annual.inc
```

Therefore, the Streamlit application asks the user for:

```text
Annual income
```

and FastAPI internally calculates:

```text
log(actual annual income)
```

For example, the user can enter:

```text
₹5,00,000
```

instead of entering a logarithmic value.

The user should **never manually enter `log.annual.inc`**.

---

## Interest Rate

The UI accepts:

```text
15.51%
```

while the model expects:

```text
0.1551
```

The API performs this conversion internally.

---

## Credit History

The UI accepts:

```text
5 years
```

while the model expects:

```text
days.with.cr.line
```

The API converts the number of years into approximate days.

---

## 19. Train/Test Split

The dataset was split into:

```text
80% training
20% testing
```

using:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The resulting sizes were:

```text
Training set → 7,662
Test set     → 1,916
```

Stratification was important because the target is imbalanced.

It preserved approximately the same:

```text
84% / 16%
```

class distribution in both training and testing data.

---

## 20. Why Accuracy Is Not the Main Metric

Because the non-payment class represents only about 16% of the dataset, accuracy alone can be misleading.

For example, a model that predicts:

```text
Fully Paid
```

for nearly everyone could achieve high accuracy while doing a poor job identifying non-payment cases.

Therefore, the project focuses more strongly on:

- Recall
- Precision
- F1-score
- ROC-AUC
- PR-AUC
- Probability calibration
- Brier score
- Risk-band separation

---

## 21. Baseline Logistic Regression

A baseline Logistic Regression model was first trained without class weighting.

Its test-set confusion matrix was:

```text
[[1595, 14],
 [ 295, 12]]
```

This shows an important issue.

The model predicted very few positive/non-payment cases.

It correctly identified only:

```text
12
```

of the:

```text
307
```

actual non-payment cases in the test set.

Therefore, although the baseline could appear strong on the majority class, it was not suitable as the final risk model.

This motivated explicit handling of class imbalance.

---

## 22. Class-Weighted Logistic Regression

The next model used:

```python
class_weight="balanced"
```

Results on the held-out test set were:

```text
Accuracy  : 0.6336
Precision : 0.2384
Recall    : 0.5863
F1        : 0.3390
ROC-AUC   : 0.6835
PR-AUC    : 0.3030
```

The confusion matrix became:

```text
[[1034, 575],
 [ 127, 180]]
```

The important change was minority-class recall:

```text
Baseline recall       → very low
Balanced model recall → 58.63%
```

The decrease in accuracy was therefore accepted because the goal was not simply to maximize majority-class accuracy.

---

## 23. Model Comparison

Three models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

They were compared using stratified cross-validation.

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| ROC-AUC | 0.6752 | 0.6554 | 0.6501 |
| PR-AUC | 0.2888 | 0.2574 | 0.2741 |
| Accuracy | 0.6347 | 0.8401 | 0.7018 |
| Precision | 0.2451 | 0.3667 | 0.2555 |
| Recall | 0.6166 | 0.0049 | 0.4511 |
| F1 | 0.3507 | 0.0097 | 0.3262 |

---

## 24. Why Logistic Regression Was Selected

Logistic Regression was selected for several reasons.

### 1. Predictive performance

It achieved the strongest cross-validation:

```text
ROC-AUC
PR-AUC
```

among the tested models.

### 2. Minority-class recall

It provided much stronger recall than Random Forest.

Random Forest's:

```text
0.0049 recall
```

indicated that it was almost completely failing to identify the minority class under the evaluated setup.

Therefore, its high accuracy:

```text
0.8401
```

was misleading for this problem.

### 3. Interpretability

Logistic Regression provides coefficients that can be interpreted through:

- direction
- odds ratios
- standardized feature effects
- applicant-specific contributions

### 4. Probability output

The model naturally produces probability estimates, which are useful for risk segmentation.

### 5. Practical communication

A relatively interpretable model is useful when building a financial-risk prototype where model reasoning needs to be communicated to technical and non-technical stakeholders.

This does **not** mean Logistic Regression is universally the best algorithm for production credit risk.

---

## 25. Cross-Validation

A 5-fold stratified cross-validation strategy was used.

The purpose was to evaluate whether model performance was reasonably stable across different training/validation splits.

For the selected Logistic Regression configuration:

```text
Mean ROC-AUC ≈ 0.6752
Mean PR-AUC  ≈ 0.2888
```

The use of cross-validation helped avoid selecting a model based only on one particular train/test split.

---

## 26. Hyperparameter Tuning

The Logistic Regression regularization parameter:

```text
C
```

was tuned using cross-validation.

PR-AUC was used as the optimization metric.

The tested values included:

```text
0.01
0.10
0.50
1
2
5
10
```

The best value was:

```text
C = 2
```

with:

```text
Best CV PR-AUC ≈ 0.2889
```

The top values were relatively close to one another, suggesting that performance was not extremely sensitive to the exact value within the tested range.

The final underlying Logistic Regression therefore uses:

```python
LogisticRegression(
    C=2,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)
```

---

## 27. Out-of-Fold Predictions

Out-of-fold probability predictions were generated on the training set.

This was important because calibration and risk-band analysis should not rely on predictions generated from a model that was trained on the same rows.

The resulting out-of-fold performance was approximately:

```text
OOF ROC-AUC → 0.6752
OOF PR-AUC  → 0.2840
```

These predictions were then used for probability analysis and calibration-related evaluation.

---

## 28. Probability Calibration

A major requirement of this project is probability-based risk segmentation.

There is an important difference between:

> ranking applicants from lower risk to higher risk

and:

> producing probability estimates that can be meaningfully interpreted.

Therefore, the Logistic Regression model was calibrated using:

```text
CalibratedClassifierCV
```

with the tuned class-weighted Logistic Regression as the underlying estimator.

---

## 29. Calibration Results

The project obtained approximately:

```text
Uncalibrated Brier score → 0.2251
Calibrated Brier score   → 0.1271
```

while ranking performance remained approximately:

```text
Calibrated ROC-AUC → 0.6753
Calibrated PR-AUC  → 0.2837
```

The important observation is:

> Calibration substantially improved the Brier score while leaving ranking performance broadly similar.

That is consistent with the purpose of calibration: improving probability quality rather than artificially improving ranking metrics.

---

## 30. Probability Distribution

The calibrated probability distribution on the training out-of-fold predictions was:

```text
Mean   → 0.1603
Median → 0.1433
75%    → 0.1982
90%    → 0.2643
95%    → 0.3224
99%    → 0.4614
Maximum→ 0.9854
```

The distribution showed that most applicants were concentrated in relatively low probability ranges, while a smaller group received substantially higher predicted risk.

This distribution was useful when considering practical risk-band thresholds.

---

## 31. Probability Decile Analysis

The calibrated probabilities were divided into ten groups.

The observed non-payment rate increased as predicted probability increased.

Examples:

```text
Lowest probability group → 4.04% observed non-payment

Highest probability group → 35.33% observed non-payment
```

The decile analysis therefore demonstrated useful risk ordering.

This was an important step before defining the final Low / Medium / High bands.

---

## 32. Choosing Low / Medium / High Risk

Several threshold combinations were evaluated.

### Candidate: 10% / 20%

| Risk | Borrowers | Population | Mean predicted probability | Observed non-payment |
|---|---:|---:|---:|---:|
| Low | 1,855 | 24.21% | 7.37% | 6.25% |
| Medium | 3,946 | 51.50% | 14.55% | 14.95% |
| High | 1,861 | 24.29% | 27.77% | 27.94% |

### Candidate: 15% / 25%

| Risk | Borrowers | Population | Mean predicted probability | Observed non-payment |
|---|---:|---:|---:|---:|
| Low | 4,125 | 53.84% | 10.21% | 9.82% |
| Medium | 2,621 | 34.21% | 19.08% | 19.76% |
| High | 916 | 11.96% | 33.49% | 33.08% |

### Candidate: 10% / 25%

| Risk | Borrowers | Population | Mean predicted probability | Observed non-payment |
|---|---:|---:|---:|---:|
| Low | 1,855 | 24.21% | 7.37% | 6.25% |
| Medium | 4,891 | 63.83% | 16.04% | 16.50% |
| High | 916 | 11.96% | 33.49% | 33.08% |

The final prototype uses:

```text
<10%       → Low
10–<20%    → Medium
>=20%      → High
```

The 10% / 20% split was selected because it creates three reasonably populated groups while maintaining clear separation in observed non-payment rates.

These thresholds are **prototype segmentation choices**, not bank lending policy.

---

## 33. Final Test-Set Risk Segmentation

The final calibrated model was evaluated on the held-out test set.

The resulting risk bands were:

| Risk | Borrowers | Population | Mean predicted probability | Observed non-payment |
|---|---:|---:|---:|---:|
| Low | 491 | 25.63% | 7.40% | 4.89% |
| Medium | 990 | 51.67% | 14.58% | 15.56% |
| High | 435 | 22.70% | 28.67% | 29.66% |

The observed non-payment rate increases from:

```text
Low      → 4.89%
Medium   → 15.56%
High     → 29.66%
```

This is one of the most important business-facing results of the project.

It demonstrates that the risk bands provide meaningful separation on previously unseen test data.

---

## 34. High-Risk Threshold Evaluation

Using:

```text
probability >= 20%
```

as a high-risk indicator produced:

```text
Accuracy  : 0.7474
Precision : 0.2966
Recall    : 0.4202
F1        : 0.3477
```

Confusion matrix:

```text
[[1303, 306],
 [ 178, 129]]
```

Therefore:

- 129 actual non-payment cases were identified as high-risk.
- 178 actual non-payment cases were missed.
- 306 fully-paid applicants were classified as high-risk.

This demonstrates the trade-off involved in choosing a threshold.

In a real financial institution, thresholds would be selected using:

- cost of false negatives
- cost of false positives
- risk appetite
- business policy
- regulatory requirements
- portfolio characteristics

rather than simply selecting a convenient numerical threshold.

---

## 35. Feature Importance

Two complementary approaches were used.

### 35.1 Logistic Regression Coefficients

The final model coefficients were analyzed using:

- coefficient
- odds ratio
- absolute coefficient magnitude

Some stronger positive associations included:

- Small-business loan purpose
- Recent credit inquiries
- Monthly installment
- Revolving balance
- Revolving utilization
- Interest rate

Some stronger negative associations included:

- FICO score
- Log annual income
- Certain loan-purpose categories

These describe **model associations**, not causal relationships.

---

## 36. Logistic Regression Coefficients and Odds Ratios

Selected coefficients from the final underlying model included:

| Feature | Coefficient | Odds ratio |
|---|---:|---:|
| `purpose_small_business` | 0.5258 | 1.6918 |
| `purpose_credit_card` | -0.5202 | 0.5944 |
| `purpose_major_purchase` | -0.4823 | 0.6174 |
| `fico` | -0.3529 | 0.7027 |
| `purpose_debt_consolidation` | -0.3097 | 0.7337 |
| `inq.last.6mths` | 0.2910 | 1.3378 |
| `log.annual.inc` | -0.2746 | 0.7599 |
| `installment` | 0.2271 | 1.2550 |
| `revol.bal` | 0.1706 | 1.1860 |
| `revol.util` | 0.0816 | 1.0850 |

Because numerical features were standardized, numerical coefficients correspond to changes in standardized feature values.

Because categorical features are one-hot encoded, categorical coefficients are relative to the reference category.

---

## 37. Why Standardization Matters for Coefficients

The numerical features were standardized before Logistic Regression.

Therefore, a numerical coefficient represents the model effect associated with approximately a:

```text
one-standard-deviation increase
```

in that feature, holding the other model inputs constant.

Categorical variables are represented using one-hot encoded indicators relative to the reference category.

For example:

```text
purpose = small_business
```

is interpreted relative to:

```text
purpose = all_other
```

Therefore, it would be incorrect to make simplistic statements such as:

> "Small-business loans are 69% more likely to default."

The odds ratio describes the relationship learned by this specific model representation, conditional on the other variables.

---

## 38. Permutation Importance

Permutation importance was calculated on a separate interpretation validation set using:

```text
average_precision
```

(PR-AUC) as the scoring metric.

The strongest features were:

| Feature | Mean permutation importance |
|---|---:|
| `inq.last.6mths` | 0.04599 |
| `fico` | 0.04036 |
| `installment` | 0.01466 |
| `log.annual.inc` | 0.01406 |
| `purpose` | 0.01295 |
| `revol.bal` | 0.00761 |
| `revol.util` | 0.00599 |
| `pub.rec` | 0.00583 |
| `int.rate` | 0.00292 |
| `days.with.cr.line` | 0.00186 |
| `delinq.2yrs` | 0.00064 |
| `dti` | -0.00084 |

Recent inquiries and FICO were particularly important according to permutation importance.

Using both coefficient analysis and permutation importance provides a more complete view than relying on only one importance technique.

---

## 39. Individual Applicant Explainability

The application provides applicant-specific explanations.

For the sample applicant used during development:

```text
Purpose              → debt_consolidation
Interest rate        → 15.51%
Installment          → 418.99
Log annual income    → 11.03489
DTI                  → 15.38
FICO                 → 647
Credit history       → 1860 days
Revolving balance    → 11214
Revolving utilization→ 70.1%
Recent inquiries     → 0
Delinquencies        → 0
Public records       → 0
```

The calibrated model estimated:

```text
Non-payment probability ≈ 19.16%

Risk category = Medium
```

The strongest individual contributions included:

- FICO
- Loan purpose
- Recent credit inquiries
- Monthly installment
- Interest rate

---

## 40. Example Applicant Explanation

The underlying Logistic Regression contributions for the sample applicant were approximately:

| Feature | Contribution |
|---|---:|
| FICO | -0.5943 |
| Debt-consolidation purpose | -0.3097 |
| Recent credit inquiries | -0.2113 |
| Monthly installment | +0.1091 |
| Interest rate | +0.1090 |
| Revolving utilization | +0.0647 |
| Credit history length | -0.0578 |
| Annual income | -0.0451 |
| Revolving balance | -0.0294 |
| Public records | -0.0126 |
| Delinquencies | +0.0113 |
| DTI | -0.0093 |

Negative contributions push the underlying Logistic Regression score toward lower modeled risk.

Positive contributions push the score toward higher modeled risk.

These contributions are calculated in the model's underlying score/log-odds space.

They should **not** be interpreted as direct percentage-point changes in the final calibrated probability.

---

## 41. Raw vs Calibrated Probability

The system contains:

```text
Underlying Logistic Regression
            ↓
      Calibration layer
            ↓
      Final probability
```

Therefore, raw Logistic Regression probability and final calibrated probability can differ.

For the sample applicant:

```text
Raw Logistic Regression log-odds → 0.2373

Raw Logistic Regression probability → 55.9%

Calibrated probability → 19.16%
```

This is expected because the calibration layer transforms the raw probability estimates.

The application uses the **calibrated probability** for risk-band assignment.

The explanations are based on the underlying Logistic Regression model and therefore explain the underlying model score rather than claiming that each feature directly changed the calibrated probability by a particular percentage.

---

## 42. Model Persistence

The final model was saved as:

```text
models/final_model.joblib
```

The saved artifact contains the trained calibrated model together with its preprocessing pipeline.

This is important because the application must reproduce the exact preprocessing used during training.

The saved model was loaded again and tested.

The original and loaded-model probabilities matched:

```text
Original probability      → 0.1916
Loaded-model probability  → 0.1916
```

This verified that the persisted model can be loaded successfully for inference.

---

## 43. Application Architecture

The project was converted from a notebook-based ML workflow into a small application architecture.

```text
                         User
                           |
                           v
                  +----------------+
                  |   Streamlit    |
                  |    Web UI      |
                  +-------+--------+
                          |
                    HTTP POST /predict
                          |
                          v
                  +----------------+
                  |    FastAPI     |
                  | Inference API  |
                  +-------+--------+
                          |
             +------------+-------------+
             |            |             |
             v            v             v
        Validation   Transformation  Explanation
             |            |             |
             +------------+-------------+
                          |
                          v
                 Calibrated ML Model
                          |
                          v
              Probability + Risk Band
                          |
                          v
                     Streamlit
```

---

## 44. Streamlit Responsibilities

The Streamlit frontend is responsible for:

- collecting applicant information
- presenting understandable financial terminology
- validating basic UI inputs
- sending requests to FastAPI
- displaying probability
- displaying risk category
- displaying explanations
- displaying applicant summary
- displaying disclaimers

The Streamlit application does **not** directly perform the model inference.

---

## 45. FastAPI Responsibilities

FastAPI acts as the inference layer.

It is responsible for:

- request validation
- feature transformation
- model inference
- probability generation
- risk classification
- explanation generation
- structured API responses
- health checks
- model metadata

This separation makes the ML model reusable independently of the frontend.

---

## 46. Centralized Risk Logic

Risk thresholds are stored centrally in:

```text
core/risk.py
```

For example:

```python
LOW_RISK_THRESHOLD = 0.10
HIGH_RISK_THRESHOLD = 0.20
```

Both the API and tests use the same risk-classification function.

This avoids duplicating:

```text
Low / Medium / High
```

logic in multiple files.

If the thresholds change later, there is one source of truth.

---

## 47. API Endpoints

### Health

```http
GET /health
```

Checks whether the API and model are available.

---

### Model Information

```http
GET /model-info
```

Returns:

- model type
- target
- risk-band definitions
- prototype notes

---

### Prediction

```http
POST /predict
```

Accepts applicant information and returns:

- estimated probability
- probability percentage
- risk category
- description
- risk-increasing factors
- risk-reducing factors
- disclaimer

---

## 48. Example API Input

```json
{
  "purpose": "debt_consolidation",
  "annual_income": 62000,
  "interest_rate": 15.51,
  "installment": 418.99,
  "dti": 15.38,
  "fico": 647,
  "credit_history_years": 5.09,
  "revolving_balance": 11214,
  "credit_utilization": 70.1,
  "recent_inquiries": 0,
  "delinquencies": 0,
  "public_records": 0
}
```

The API converts these user-friendly values into the exact feature representation required by the trained model.

---

## 49. Project Structure

```text
loan-risk-assessment/
│
├── app.py
├── loan_risk_model.py
├── requirements.txt
├── README.md
│
├── models/
│   └── final_model.joblib
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── core/
│   ├── __init__.py
│   └── risk.py
│
└── tests/
    ├── test_api.py
    ├── test_model.py
    └── test_risk.py
```

---

## 50. Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If testing dependencies are not already included:

```bash
pip install pytest httpx
```

---

### 2. Start FastAPI

From the project root:

```bash
uvicorn api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger/OpenAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

### 3. Start Streamlit

Open another terminal:

```bash
streamlit run app.py
```

---

## 51. Running Tests

Run:

```bash
pytest -v
```

Tests cover:

### Risk logic

- Low-risk threshold
- Medium-risk threshold
- High-risk threshold
- Risk descriptions

### Model

- Model loading
- Probability range
- Deterministic inference
- Required prediction output

### API

- Health endpoint
- Model information endpoint
- Valid prediction
- Invalid FICO
- Negative income
- Missing required fields

---

## 52. Engineering Decisions

### Decision 1 — Separate EDA from modeling

EDA-specific helper variables were removed before model training.

This keeps exploratory analysis separate from the final feature pipeline.

---

### Decision 2 — Use a preprocessing pipeline

The preprocessing and model are packaged together so the same transformations are applied consistently during training and inference.

---

### Decision 3 — Use class weighting

The target is imbalanced, so:

```python
class_weight="balanced"
```

was used to give greater importance to the minority class.

---

### Decision 4 — Use PR-AUC

Because the positive class is relatively uncommon, PR-AUC provides more useful information about minority-class performance than accuracy alone.

---

### Decision 5 — Calibrate probabilities

The application uses probabilities to create risk categories.

Therefore, probability quality matters in addition to ranking quality.

The Brier score improved substantially after calibration.

---

### Decision 6 — Centralize risk thresholds

Risk thresholds are maintained in:

```text
core/risk.py
```

rather than duplicated across the application.

---

### Decision 7 — Separate frontend and inference

FastAPI owns the ML inference workflow.

Streamlit owns presentation.

This allows another application to consume the same prediction API without needing to know how the model works internally.

---

### Decision 8 — Keep user inputs human-readable

The original model expects transformations such as:

```text
log.annual.inc
days.with.cr.line
```

A normal loan applicant should not be expected to understand these model-specific representations.

Therefore:

```text
Human-friendly input
        ↓
FastAPI transformation
        ↓
Model-ready feature
```

This makes the application much more suitable for non-technical users.

---

## 53. Limitations

This project should **not** be interpreted as a production-ready banking credit-decision system.

### 1. Dataset limitations

The model is trained using one historical dataset.

Real banking environments typically contain richer information such as:

- credit bureau data
- transaction history
- repayment history
- employment information
- account behavior
- macroeconomic indicators
- institution-specific policy variables

---

### 2. Historical bias

A model learns patterns present in historical data.

Historical data may contain biases or institutional effects that should not automatically be reproduced in production.

---

### 3. No causal interpretation

The model identifies statistical associations.

It does not prove that changing one feature will cause the applicant's risk to change.

---

### 4. Limited feature set

The current dataset may not contain many variables that would be relevant to real-world underwriting.

---

### 5. No temporal validation

The project currently uses a stratified train/test split.

A production credit-risk system should also consider time-based validation to evaluate how the model performs under changing economic conditions.

---

### 6. No drift monitoring

The system does not currently monitor:

- feature drift
- prediction drift
- calibration drift
- population changes
- model performance degradation

---

### 7. No fairness assessment

A complete production system would require fairness and bias assessment across relevant applicant groups.

---

### 8. No automated retraining

The current application uses a fixed model artifact.

There is no automated model retraining pipeline.

---

### 9. No production security layer

The prototype does not yet implement a complete production security architecture such as:

- authentication
- authorization
- HTTPS configuration
- secret management
- rate limiting
- audit logging

---

### 10. Not a complete credit-decision system

The project does not implement:

- loan approval policy
- fraud detection
- identity verification
- affordability policy
- regulatory checks
- collateral evaluation
- employment verification
- manual underwriting
- institution-specific risk appetite

Therefore:

> **The output should be treated as a risk signal, not a final lending decision.**

---

## 54. Future Improvements

### Machine Learning

Potential improvements include:

- More extensive hyperparameter tuning
- Gradient boosting models
- Ensemble approaches
- Temporal validation
- Cost-sensitive optimization
- Calibration monitoring
- Model stability analysis

---

### Data

Potential additions include:

- Credit bureau variables
- Transaction behavior
- Repayment history
- Employment information
- Macroeconomic indicators
- Additional affordability measures

---

### Explainability

Future versions could add:

- SHAP
- Global explanation dashboards
- Local explanation visualizations
- Explanation stability testing

---

### MLOps

Potential improvements:

- MLflow experiment tracking
- Model registry
- Data validation
- Automated retraining
- Model monitoring
- Drift detection
- CI/CD

---

### Responsible AI

A real financial deployment would require:

- Fairness testing
- Bias assessment
- Model governance
- Auditability
- Explainability review
- Human-in-the-loop decision processes

---

### Production Deployment

Potential improvements:

- Docker
- Cloud deployment
- HTTPS
- Authentication
- Rate limiting
- Secret management
- Centralized logging
- Monitoring
- Production database integration

---

## 55. What This Project Demonstrates

This project demonstrates the complete transition from:

```text
ML Notebook
```

to:

```text
ML Application
```

The overall workflow is:

```text
Business problem
       ↓
Data understanding
       ↓
Data quality checks
       ↓
EDA
       ↓
Financial feature analysis
       ↓
Preprocessing
       ↓
Class imbalance handling
       ↓
Model comparison
       ↓
Hyperparameter tuning
       ↓
Probability calibration
       ↓
Risk segmentation
       ↓
Explainability
       ↓
Model persistence
       ↓
FastAPI inference service
       ↓
Streamlit application
       ↓
Automated testing
```

The important part is that the project does not stop at:

```text
train model → print accuracy
```

It continues into:

```text
model
  ↓
calibrated probability
  ↓
risk category
  ↓
individual explanation
  ↓
API
  ↓
user interface
  ↓
tests
```

---

## 56. Interview Summary

A concise explanation of the project:

> I built a loan non-payment risk assessment system using historical loan and credit-profile data. The target was whether a loan was not fully paid. Since the non-payment class represented only about 16% of the dataset, I focused on recall, PR-AUC and probability calibration rather than accuracy alone.
>
> I compared Logistic Regression, Random Forest and XGBoost using stratified cross-validation. Logistic Regression achieved the strongest ROC-AUC and PR-AUC among the tested models and also provided better minority-class recall and interpretability, so I selected it with balanced class weights.
>
> I then tuned the regularization parameter using PR-AUC and calibrated the resulting probabilities. The calibrated probabilities were analyzed using deciles to understand whether higher predicted probabilities corresponded to higher observed non-payment rates.
>
> Based on this analysis, I created Low, Medium and High risk bands using 10% and 20% probability thresholds. On the held-out test set, the observed non-payment rate increased from approximately 4.9% in the Low-risk group to 29.7% in the High-risk group.
>
> I also implemented applicant-level explanations using Logistic Regression contributions. Finally, I separated the ML inference layer into FastAPI and built a Streamlit frontend, with centralized risk logic and automated tests.

---

## 57. Would This Be Deployed Directly in a Bank?

No.

This project is a **portfolio/educational prototype**, not a production credit-decision system.

Before production deployment, I would require:

- larger and more representative data
- temporal validation
- fairness assessment
- model governance
- business-approved thresholds
- regulatory review
- drift monitoring
- security controls
- audit logging
- human/policy decision layers
- production-grade monitoring
- formal model validation

The current system provides a **risk signal** that could support a larger decision-making process.

---

## 58. Key Takeaway

The main goal of this project is not simply:

> "Train a model that predicts loan default."

The project demonstrates the reasoning required to turn a financial ML model into an application:

```text
Understand the business problem
          ↓
Understand the financial features
          ↓
Identify meaningful patterns
          ↓
Handle class imbalance
          ↓
Compare appropriate models
          ↓
Choose metrics based on the problem
          ↓
Calibrate probabilities
          ↓
Convert probabilities into meaningful risk bands
          ↓
Explain individual predictions
          ↓
Expose the model through an API
          ↓
Build a user-facing application
          ↓
Test the system
          ↓
Prepare it for deployment
```

The separation between:

```text
Prediction
Probability
Risk segmentation
Explanation
Final business decision
```

is intentional.

The model provides the first four.

The final lending decision would require additional business, regulatory, fairness, fraud, affordability, and human-review processes.

---

## 59. License / Usage

This project is intended for:

- Educational purposes
- Machine-learning portfolio demonstration
- Software engineering practice
- ML deployment practice
- Interview/project discussion

It should not be used as a real-world automated lending decision system without substantial additional validation, governance, security, fairness assessment, and regulatory review.
