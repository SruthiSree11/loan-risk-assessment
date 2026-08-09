import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

FASTAPI_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🏦 Loan Risk Assessment")

st.write(
    """
    Estimate the modeled non-payment risk of a loan applicant
    using financial and credit-history information.
    """
)

st.info(
    """
    This prototype estimates non-payment risk and assigns a
    Low, Medium, or High risk category. It is not a standalone
    loan approval or rejection system.
    """
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("Applicant Information")


# ------------------------------------------------------------
# Loan purpose
# ------------------------------------------------------------

purpose_display = {
    "Credit card": "credit_card",
    "Debt consolidation": "debt_consolidation",
    "Education": "educational",
    "Home improvement": "home_improvement",
    "Major purchase": "major_purchase",
    "Small business": "small_business",
    "Other": "all_other",
}

purpose_label = st.selectbox(
    "Loan purpose",
    options=list(purpose_display.keys()),
    help="The main purpose for which the loan is being requested.",
)

purpose = purpose_display[purpose_label]


# ------------------------------------------------------------
# Income and loan details
# ------------------------------------------------------------

income_col1, income_col2 = st.columns(2)


with income_col1:

    annual_income = st.number_input(
        "Annual income",
        min_value=1000.0,
        max_value=2_000_000.0,
        value=60_000.0,
        step=1_000.0,
        help="Applicant's total annual income.",
    )


with income_col2:

    interest_rate = st.number_input(
        "Interest rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1,
        help="Annual interest rate for the proposed loan.",
    )


installment_col1, installment_col2 = st.columns(2)


with installment_col1:

    installment = st.number_input(
        "Monthly loan installment",
        min_value=0.0,
        max_value=5_000.0,
        value=400.0,
        step=10.0,
        help="Expected monthly loan payment.",
    )


with installment_col2:

    dti = st.number_input(
        "Debt-to-income ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1,
        help=(
            "Percentage of income committed to existing "
            "debt obligations."
        ),
    )


# ============================================================
# CREDIT PROFILE
# ============================================================

st.header("Credit Profile")


credit_col1, credit_col2 = st.columns(2)


with credit_col1:

    fico = st.number_input(
        "FICO credit score",
        min_value=300,
        max_value=850,
        value=700,
        step=1,
        help="Applicant's FICO credit score.",
    )


with credit_col2:

    credit_history_years = st.number_input(
        "Credit history length (years)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help=(
            "Approximate number of years the applicant "
            "has had an established credit history."
        ),
    )


# ============================================================
# CREDIT UTILIZATION
# ============================================================

st.header("Current Credit Activity")


activity_col1, activity_col2 = st.columns(2)


with activity_col1:

    revolving_balance = st.number_input(
        "Revolving credit balance",
        min_value=0.0,
        max_value=2_000_000.0,
        value=10_000.0,
        step=500.0,
        help=(
            "Outstanding balance on revolving credit "
            "accounts such as credit cards."
        ),
    )


with activity_col2:

    credit_utilization = st.number_input(
        "Credit utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
        help=(
            "Percentage of available revolving credit "
            "currently being used."
        ),
    )


recent_col1, recent_col2 = st.columns(2)


with recent_col1:

    recent_inquiries = st.number_input(
        "Recent credit inquiries",
        min_value=0,
        max_value=50,
        value=1,
        step=1,
        help=(
            "Number of credit inquiries recorded during "
            "the last six months."
        ),
    )


with recent_col2:

    delinquencies = st.number_input(
        "Delinquencies in last 2 years",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help=(
            "Number of recorded credit-payment delinquencies "
            "during the last two years."
        ),
    )


public_records = st.number_input(
    "Public credit records",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
    help=(
        "Number of public credit records associated with "
        "the applicant."
    ),
)


# ============================================================
# ASSESS RISK
# ============================================================

st.divider()


assess_button = st.button(
    "Assess Loan Risk",
    type="primary",
    use_container_width=True,
)


if assess_button:

    # --------------------------------------------------------
    # Build API request
    # --------------------------------------------------------

    request_payload = {
        "purpose": purpose,
        "annual_income": annual_income,
        "interest_rate": interest_rate,
        "installment": installment,
        "dti": dti,
        "fico": fico,
        "credit_history_years": credit_history_years,
        "revolving_balance": revolving_balance,
        "credit_utilization": credit_utilization,
        "recent_inquiries": recent_inquiries,
        "delinquencies": delinquencies,
        "public_records": public_records,
    }

    # --------------------------------------------------------
    # Send request to FastAPI
    # --------------------------------------------------------

    try:

        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json=request_payload,
            timeout=10,
        )

    except requests.exceptions.ConnectionError:

        st.error(
            """
            Could not connect to the FastAPI service.

            Make sure the API is running with:

            `uvicorn api.main:app --reload`
            """
        )

        st.stop()

    except requests.exceptions.Timeout:

        st.error(
            "The risk assessment request timed out."
        )

        st.stop()

    except requests.exceptions.RequestException as exc:

        st.error(
            f"Request failed: {exc}"
        )

        st.stop()

    # --------------------------------------------------------
    # Handle API errors
    # --------------------------------------------------------

    if response.status_code != 200:

        try:

            error_detail = response.json().get(
                "detail",
                "Unknown API error.",
            )

        except Exception:

            error_detail = response.text

        st.error(
            f"Risk assessment failed: {error_detail}"
        )

        st.stop()

    # --------------------------------------------------------
    # Read API response
    # --------------------------------------------------------

    prediction = response.json()


    # ========================================================
    # RISK ASSESSMENT RESULT
    # ========================================================

    st.divider()

    st.header("Risk Assessment Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Estimated Non-Payment Risk",
            f"{prediction['probability_percent']:.2f}%",
        )


    with result_col2:

        st.metric(
            "Risk Category",
            prediction["risk_category"],
        )


    st.write(
        prediction["description"]
    )


    st.caption(
        prediction["disclaimer"]
    )


    # ========================================================
    # APPLICANT SUMMARY
    # ========================================================

    st.divider()

    st.header("Applicant Summary")


    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.write(
            f"**Loan purpose:** {purpose_label}"
        )

        st.write(
            f"**Annual income:** {annual_income:,.0f}"
        )

        st.write(
            f"**Interest rate:** {interest_rate:.2f}%"
        )

        st.write(
            f"**Monthly installment:** "
            f"{installment:,.2f}"
        )


    with summary_col2:

        st.write(
            f"**FICO score:** {fico}"
        )

        st.write(
            f"**Credit history:** "
            f"{credit_history_years:.1f} years"
        )

        st.write(
            f"**Debt-to-income ratio:** "
            f"{dti:.1f}%"
        )


    with summary_col3:

        st.write(
            f"**Revolving balance:** "
            f"{revolving_balance:,.2f}"
        )

        st.write(
            f"**Credit utilization:** "
            f"{credit_utilization:.1f}%"
        )

        st.write(
            f"**Recent inquiries:** "
            f"{recent_inquiries}"
        )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "Why did the model assign this risk?"
    )

    st.caption(
        """
        These factors describe which applicant attributes
        contributed toward a higher or lower modeled
        non-payment risk. They are model associations,
        not causal conclusions.
        """
    )


    risk_increasing = prediction.get(
        "risk_increasing_factors",
        [],
    )

    risk_reducing = prediction.get(
        "risk_reducing_factors",
        [],
    )


    explanation_col1, explanation_col2 = st.columns(2)


    # --------------------------------------------------------
    # Risk-increasing factors
    # --------------------------------------------------------

    with explanation_col1:

        st.subheader(
            "Factors increasing modeled risk"
        )


        if len(risk_increasing) == 0:

            st.write(
                "No major risk-increasing factors "
                "were identified among the top contributors."
            )

        else:

            for factor in risk_increasing:

                st.markdown(
                    f"**{factor['display_name']}**"
                )

                st.write(
                    factor["text"]
                )

                st.write("")


    # --------------------------------------------------------
    # Risk-reducing factors
    # --------------------------------------------------------

    with explanation_col2:

        st.subheader(
            "Factors reducing modeled risk"
        )


        if len(risk_reducing) == 0:

            st.write(
                "No major risk-reducing factors "
                "were identified among the top contributors."
            )

        else:

            for factor in risk_reducing:

                st.markdown(
                    f"**{factor['display_name']}**"
                )

                st.write(
                    factor["text"]
                )

                st.write("")


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    st.divider()


    with st.expander(
        "Technical details"
    ):

        st.write(
            """
            The prediction is generated by a calibrated
            Logistic Regression model.

            The API converts user-friendly inputs into the
            feature representation used during model training.

            For example:

            • Annual income is logarithmically transformed.

            • Credit-history years are converted to days.

            • Interest rate percentage is converted to
              decimal representation.

            The risk category is then assigned using the
            project's prototype probability thresholds.

            The individual explanation is derived from the
            underlying Logistic Regression model's
            applicant-specific feature contributions.

            These explanations should not be interpreted as
            causal effects or as formal credit policy rules.
            """
        )