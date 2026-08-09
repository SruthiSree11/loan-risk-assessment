# ============================================================
# RISK BAND CONFIGURATION
# ============================================================

LOW_RISK_THRESHOLD = 0.10
HIGH_RISK_THRESHOLD = 0.20


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability: float) -> str:
    """
    Convert estimated non-payment probability into
    the project's prototype risk category.

    Risk bands:

        Low       : probability < 10%
        Medium    : 10% <= probability < 20%
        High      : probability >= 20%
    """

    if probability < LOW_RISK_THRESHOLD:
        return "Low"

    if probability < HIGH_RISK_THRESHOLD:
        return "Medium"

    return "High"


# ============================================================
# RISK DESCRIPTION
# ============================================================

def build_risk_description(
    risk_category: str,
    probability: float,
) -> str:
    """
    Build a human-readable description of the
    model's risk-band result.
    """

    probability_percent = probability * 100

    if risk_category == "Low":

        return (
            f"The model estimates a {probability_percent:.2f}% "
            "non-payment probability, which falls within the "
            "prototype Low-risk band."
        )

    if risk_category == "Medium":

        return (
            f"The model estimates a {probability_percent:.2f}% "
            "non-payment probability, which falls within the "
            "prototype Medium-risk band."
        )

    return (
        f"The model estimates a {probability_percent:.2f}% "
        "non-payment probability, which falls within the "
        "prototype High-risk band."
    )


# ============================================================
# RISK THRESHOLD INFORMATION
# ============================================================

def get_risk_band_information() -> dict[str, str]:
    """
    Return the project's risk-band definitions.
    """

    return {
        "Low": "< 10% estimated non-payment probability",
        "Medium": "10% to < 20% estimated non-payment probability",
        "High": ">= 20% estimated non-payment probability",
    }