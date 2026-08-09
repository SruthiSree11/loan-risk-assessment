from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# APPLICANT REQUEST
# ============================================================

class ApplicantRequest(BaseModel):
    """
    Human-friendly applicant information received by the API.

    These values are transformed internally into the exact
    feature representation expected by the trained model.
    """

    purpose: Literal[
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business",
        "all_other",
    ]

    annual_income: float = Field(
        ...,
        gt=0,
        description="Annual income in currency units.",
    )

    interest_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Annual interest rate as a percentage.",
    )

    installment: float = Field(
        ...,
        ge=0,
        description="Monthly loan installment.",
    )

    dti: float = Field(
        ...,
        ge=0,
        le=100,
        description="Debt-to-income ratio in percentage terms.",
    )

    fico: int = Field(
        ...,
        ge=300,
        le=850,
        description="FICO credit score.",
    )

    credit_history_years: float = Field(
        ...,
        ge=0,
        description="Length of credit history in years.",
    )

    revolving_balance: float = Field(
        ...,
        ge=0,
        description="Outstanding revolving credit balance.",
    )

    credit_utilization: float = Field(
        ...,
        ge=0,
        le=100,
        description="Revolving credit utilization percentage.",
    )

    recent_inquiries: int = Field(
        ...,
        ge=0,
        description="Credit inquiries during the last 6 months.",
    )

    delinquencies: int = Field(
        ...,
        ge=0,
        description="Delinquencies during the last 2 years.",
    )

    public_records: int = Field(
        ...,
        ge=0,
        description="Number of public credit records.",
    )


# ============================================================
# EXPLANATION FACTOR
# ============================================================

class ExplanationFactor(BaseModel):
    """
    One applicant-specific model explanation factor.
    """

    display_name: str

    direction: Literal[
        "Risk-increasing",
        "Risk-reducing",
    ]

    text: str


# ============================================================
# RISK RESPONSE
# ============================================================

class RiskResponse(BaseModel):
    """
    Complete response returned by /predict.
    """

    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Estimated probability of non-payment.",
    )

    probability_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Estimated non-payment probability as a percentage.",
    )

    risk_category: Literal[
        "Low",
        "Medium",
        "High",
    ]

    description: str

    risk_increasing_factors: list[ExplanationFactor]

    risk_reducing_factors: list[ExplanationFactor]

    disclaimer: str


# ============================================================
# HEALTH RESPONSE
# ============================================================

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool


# ============================================================
# MODEL INFORMATION RESPONSE
# ============================================================

class ModelInfoResponse(BaseModel):

    model_type: str

    target: str

    risk_bands: dict[str, str]

    note: str