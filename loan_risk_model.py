import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIRECTORY = "models"

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "final_model.joblib"
)

EXPLANATION_MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "explanation_model.joblib"
)


# ============================================================
# FEATURE DISPLAY NAMES
# ============================================================

FEATURE_DISPLAY_NAMES = {

    "fico":
        "FICO Score",

    "int.rate":
        "Interest Rate",

    "installment":
        "Monthly Installment",

    "log.annual.inc":
        "Annual Income",

    "dti":
        "Debt-to-Income Ratio",

    "days.with.cr.line":
        "Credit History Length",

    "revol.bal":
        "Revolving Credit Balance",

    "revol.util":
        "Credit Utilization",

    "inq.last.6mths":
        "Recent Credit Inquiries",

    "delinq.2yrs":
        "Delinquencies in Last 2 Years",

    "pub.rec":
        "Public Records",

    "purpose":
        "Loan Purpose"
}


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    """
    Load the final calibrated model and the fitted
    Logistic Regression model used for explanations.
    """

    if not os.path.exists(FINAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Final model not found: {FINAL_MODEL_PATH}"
        )

    if not os.path.exists(EXPLANATION_MODEL_PATH):
        raise FileNotFoundError(
            f"Explanation model not found: "
            f"{EXPLANATION_MODEL_PATH}\n\n"
            "Run the notebook cell that saves "
            "best_logistic as explanation_model.joblib."
        )

    final_model = joblib.load(
        FINAL_MODEL_PATH
    )

    explanation_model = joblib.load(
        EXPLANATION_MODEL_PATH
    )

    return final_model, explanation_model


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(probability):
    """
    Convert predicted non-payment probability into
    the project's three risk categories.

    Thresholds:
        < 10%       -> Low
        10%-20%     -> Medium
        >= 20%      -> High
    """

    if probability < 0.10:

        return "Low"

    elif probability < 0.20:

        return "Medium"

    else:

        return "High"


# ============================================================
# RISK DESCRIPTION
# ============================================================

def get_risk_description(risk_category, probability):
    """
    Generate a user-facing explanation of the risk category.
    """

    probability_percent = probability * 100

    if risk_category == "Low":

        return (
            f"The model estimates a {probability_percent:.2f}% "
            "probability of non-payment, placing this applicant "
            "in the Low risk band under the configured thresholds."
        )

    elif risk_category == "Medium":

        return (
            f"The model estimates a {probability_percent:.2f}% "
            "probability of non-payment, placing this applicant "
            "in the Medium risk band under the configured thresholds."
        )

    else:

        return (
            f"The model estimates a {probability_percent:.2f}% "
            "probability of non-payment, placing this applicant "
            "in the High risk band under the configured thresholds."
        )


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(applicant, final_model):
    """
    Generate calibrated non-payment probability
    and corresponding risk category.
    """

    probability = final_model.predict_proba(
        applicant
    )[:, 1][0]

    risk_category = get_risk_category(
        probability
    )

    description = get_risk_description(
        risk_category,
        probability
    )

    return {
        "probability": float(probability),
        "risk_category": risk_category,
        "description": description
    }


# ============================================================
# EXTRACT LOGISTIC REGRESSION COMPONENTS
# ============================================================

def _get_preprocessor_and_model(explanation_model):
    """
    Extract the fitted preprocessing pipeline and
    Logistic Regression model from the saved pipeline.
    """

    preprocessor = (
        explanation_model
        .named_steps["preprocessor"]
    )

    logistic_model = (
        explanation_model
        .named_steps["model"]
    )

    return preprocessor, logistic_model


# ============================================================
# GET FEATURE NAMES AFTER PREPROCESSING
# ============================================================

def _get_processed_feature_names(preprocessor):
    """
    Get feature names after StandardScaler and
    OneHotEncoder preprocessing.
    """

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    return feature_names


# ============================================================
# INDIVIDUAL EXPLANATION
# ============================================================

def explain_prediction(
    applicant,
    explanation_model
):
    """
    Calculate applicant-specific Logistic Regression
    feature contributions.

    Contribution =
        processed feature value × model coefficient

    Positive contribution:
        increases modeled non-payment risk.

    Negative contribution:
        decreases modeled non-payment risk.
    """

    preprocessor, logistic_model = (
        _get_preprocessor_and_model(
            explanation_model
        )
    )

    # --------------------------------------------------------
    # Transform applicant using the same preprocessing
    # used during model training.
    # --------------------------------------------------------

    processed_applicant = (
        preprocessor.transform(
            applicant
        )
    )

    # --------------------------------------------------------
    # Convert sparse matrix if necessary.
    # --------------------------------------------------------

    if hasattr(
        processed_applicant,
        "toarray"
    ):

        processed_applicant = (
            processed_applicant.toarray()
        )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        _get_processed_feature_names(
            preprocessor
        )
    )

    # --------------------------------------------------------
    # Model coefficients
    # --------------------------------------------------------

    coefficients = (
        logistic_model.coef_[0]
    )

    # --------------------------------------------------------
    # Applicant-specific contributions
    # --------------------------------------------------------

    processed_values = (
        processed_applicant[0]
    )

    contributions = (
        processed_values * coefficients
    )

    # --------------------------------------------------------
    # Build explanation dataframe
    # --------------------------------------------------------

    explanation_df = pd.DataFrame({

        "feature":
            feature_names,

        "processed_value":
            processed_values,

        "coefficient":
            coefficients,

        "contribution":
            contributions,

        "abs_contribution":
            np.abs(contributions)
    })

    # --------------------------------------------------------
    # Sort by contribution magnitude
    # --------------------------------------------------------

    explanation_df = (
        explanation_df
        .sort_values(
            "abs_contribution",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return explanation_df


# ============================================================
# EXPLANATION TEXT
# ============================================================

def build_explanation(
    explanation_df,
    applicant,
    top_n=5
):
    """
    Convert model contributions into user-facing
    explanations.
    """

    explanation = (
        explanation_df
        .copy()
    )

    # --------------------------------------------------------
    # Remove zero-contribution features
    # --------------------------------------------------------

    explanation = explanation[
        explanation["abs_contribution"] > 0
    ].copy()

    # --------------------------------------------------------
    # Keep top contributors
    # --------------------------------------------------------

    explanation = (
        explanation
        .head(top_n)
        .copy()
    )

    # --------------------------------------------------------
    # Determine direction
    # --------------------------------------------------------

    explanation["direction"] = np.where(
        explanation["contribution"] > 0,
        "Risk-increasing",
        "Risk-reducing"
    )

    # --------------------------------------------------------
    # Generate display names
    # --------------------------------------------------------

    display_names = []

    for feature in explanation["feature"]:

        # Numeric feature
        if feature.startswith(
            "numeric__"
        ):

            original_feature = (
                feature.replace(
                    "numeric__",
                    ""
                )
            )

            display_name = (
                FEATURE_DISPLAY_NAMES
                .get(
                    original_feature,
                    original_feature
                )
            )

        # Categorical feature
        elif feature.startswith(
            "categorical__"
        ):

            original_feature = (
                feature.replace(
                    "categorical__",
                    ""
                )
            )

            if original_feature.startswith(
                "purpose_"
            ):

                purpose_value = (
                    original_feature
                    .replace(
                        "purpose_",
                        ""
                    )
                )

                display_name = (
                    "Loan Purpose: "
                    + purpose_value.replace(
                        "_",
                        " "
                    ).title()
                )

            else:

                display_name = (
                    original_feature
                )

        else:

            display_name = feature

        display_names.append(
            display_name
        )

    explanation[
        "display_name"
    ] = display_names

    # --------------------------------------------------------
    # Generate natural-language explanation
    # --------------------------------------------------------

    explanation_texts = []

    for _, row in explanation.iterrows():

        feature = row["feature"]
        contribution = row["contribution"]

        # ----------------------------------------------------
        # Categorical purpose
        # ----------------------------------------------------

        if feature.startswith(
            "categorical__purpose_"
        ):

            purpose_value = (
                feature
                .replace(
                    "categorical__purpose_",
                    ""
                )
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            if contribution > 0:

                text = (
                    f"The {purpose_value} loan purpose "
                    "contributed toward higher modeled "
                    "non-payment risk."
                )

            else:

                text = (
                    f"The {purpose_value} loan purpose "
                    "contributed toward lower modeled "
                    "non-payment risk."
                )

        # ----------------------------------------------------
        # Numeric features
        # ----------------------------------------------------

        else:

            original_feature = (
                feature
                .replace(
                    "numeric__",
                    ""
                )
            )

            display_name = (
                FEATURE_DISPLAY_NAMES
                .get(
                    original_feature,
                    original_feature
                )
            )

            if original_feature == "fico":

                value = applicant[
                    "fico"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"The FICO score of {value:.0f} "
                        "contributed toward higher modeled "
                        "non-payment risk."
                    )

                else:

                    text = (
                        f"The FICO score of {value:.0f} "
                        "contributed toward lower modeled "
                        "non-payment risk."
                    )

            elif original_feature == "int.rate":

                value = (
                    applicant[
                        "int.rate"
                    ].iloc[0] * 100
                )

                if contribution > 0:

                    text = (
                        f"The interest rate of {value:.2f}% "
                        "contributed toward higher modeled "
                        "non-payment risk."
                    )

                else:

                    text = (
                        f"The interest rate of {value:.2f}% "
                        "contributed toward lower modeled "
                        "non-payment risk."
                    )

            elif original_feature == "installment":

                value = applicant[
                    "installment"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"The monthly installment of "
                        f"${value:,.2f} contributed toward "
                        "higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"The monthly installment of "
                        f"${value:,.2f} contributed toward "
                        "lower modeled non-payment risk."
                    )

            elif original_feature == "log.annual.inc":

                value = applicant[
                    "annual_income"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"The annual income of "
                        f"${value:,.0f} contributed toward "
                        "higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"The annual income of "
                        f"${value:,.0f} contributed toward "
                        "lower modeled non-payment risk."
                    )

            elif original_feature == "dti":

                value = applicant[
                    "dti"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"The debt-to-income ratio of "
                        f"{value:.2f}% contributed toward "
                        "higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"The debt-to-income ratio of "
                        f"{value:.2f}% contributed toward "
                        "lower modeled non-payment risk."
                    )

            elif original_feature == "days.with.cr.line":

                value = applicant[
                    "days.with.cr.line"
                ].iloc[0]

                years = (
                    value / 365.25
                )

                if contribution > 0:

                    text = (
                        f"A credit history length of "
                        f"{years:.1f} years contributed toward "
                        "higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"A credit history length of "
                        f"{years:.1f} years contributed toward "
                        "lower modeled non-payment risk."
                    )

            elif original_feature == "revol.bal":

                value = applicant[
                    "revol.bal"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"A revolving credit balance of "
                        f"${value:,.0f} contributed toward "
                        "higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"A revolving credit balance of "
                        f"${value:,.0f} contributed toward "
                        "lower modeled non-payment risk."
                    )

            elif original_feature == "revol.util":

                value = applicant[
                    "revol.util"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"Credit utilization of "
                        f"{value:.1f}% contributed toward "
                        "higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"Credit utilization of "
                        f"{value:.1f}% contributed toward "
                        "lower modeled non-payment risk."
                    )

            elif original_feature == "inq.last.6mths":

                value = applicant[
                    "inq.last.6mths"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"{value:.0f} recent credit inquiries "
                        "contributed toward higher modeled "
                        "non-payment risk."
                    )

                else:

                    text = (
                        f"{value:.0f} recent credit inquiries "
                        "contributed toward lower modeled "
                        "non-payment risk."
                    )

            elif original_feature == "delinq.2yrs":

                value = applicant[
                    "delinq.2yrs"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"{value:.0f} delinquencies in the last "
                        "two years contributed toward higher "
                        "modeled non-payment risk."
                    )

                else:

                    text = (
                        f"{value:.0f} delinquencies in the last "
                        "two years contributed toward lower "
                        "modeled non-payment risk."
                    )

            elif original_feature == "pub.rec":

                value = applicant[
                    "pub.rec"
                ].iloc[0]

                if contribution > 0:

                    text = (
                        f"{value:.0f} public records contributed "
                        "toward higher modeled non-payment risk."
                    )

                else:

                    text = (
                        f"{value:.0f} public records contributed "
                        "toward lower modeled non-payment risk."
                    )

            else:

                text = (
                    f"{display_name} contributed to the "
                    "model's risk estimate."
                )

        explanation_texts.append(
            text
        )

    explanation["text"] = (
        explanation_texts
    )

    return explanation