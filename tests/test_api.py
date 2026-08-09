from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


SAMPLE_APPLICANT = {
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
    "public_records": 0,
}


# ============================================================
# HEALTH TEST
# ============================================================

def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data

    assert data["model_loaded"] is True


# ============================================================
# MODEL INFO TEST
# ============================================================

def test_model_info_endpoint():

    response = client.get(
        "/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == (
        "Calibrated Logistic Regression"
    )

    assert data["target"] == (
        "not.fully.paid"
    )

    assert "Low" in data["risk_bands"]
    assert "Medium" in data["risk_bands"]
    assert "High" in data["risk_bands"]


# ============================================================
# PREDICTION TEST
# ============================================================

def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json=SAMPLE_APPLICANT,
    )

    assert response.status_code == 200

    data = response.json()

    assert "probability" in data
    assert "probability_percent" in data
    assert "risk_category" in data
    assert "description" in data
    assert "risk_increasing_factors" in data
    assert "risk_reducing_factors" in data

    assert 0 <= data["probability"] <= 1

    assert data["risk_category"] in {
        "Low",
        "Medium",
        "High",
    }


# ============================================================
# INVALID INPUT TEST
# ============================================================

def test_invalid_fico_is_rejected():

    invalid_applicant = SAMPLE_APPLICANT.copy()

    invalid_applicant["fico"] = 100

    response = client.post(
        "/predict",
        json=invalid_applicant,
    )

    assert response.status_code == 422


# ============================================================
# NEGATIVE INCOME TEST
# ============================================================

def test_negative_income_is_rejected():

    invalid_applicant = SAMPLE_APPLICANT.copy()

    invalid_applicant["annual_income"] = -5000

    response = client.post(
        "/predict",
        json=invalid_applicant,
    )

    assert response.status_code == 422


# ============================================================
# MISSING FIELD TEST
# ============================================================

def test_missing_required_field_is_rejected():

    invalid_applicant = SAMPLE_APPLICANT.copy()

    del invalid_applicant["fico"]

    response = client.post(
        "/predict",
        json=invalid_applicant,
    )

    assert response.status_code == 422