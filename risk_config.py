LOW_RISK_THRESHOLD = 0.10
HIGH_RISK_THRESHOLD = 0.20


RISK_BAND_DESCRIPTIONS = {
    "Low": {
        "description": "Lower modeled non-payment risk",
        "observed_non_payment_rate": 4.89
    },
    "Medium": {
        "description": "Moderate modeled non-payment risk",
        "observed_non_payment_rate": 15.56
    },
    "High": {
        "description": "Higher modeled non-payment risk",
        "observed_non_payment_rate": 29.66
    }
}


def assign_risk_category(probability):

    if probability < LOW_RISK_THRESHOLD:
        return "Low"

    elif probability < HIGH_RISK_THRESHOLD:
        return "Medium"

    else:
        return "High"  
