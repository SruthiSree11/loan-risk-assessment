import pandas as pd

from loan_risk_model import (
    load_models,
    predict_risk,
)


def build_sample_applicant():

    return pd.DataFrame(
        [
            {
                "purpose": "debt_consolidation",
                "int.rate": 0.1551,
                "installment": 418.99,
                "log.annual.inc": 11.03489,
                "dti": 15.38,
                "fico": 647,
                "days.with.cr.line": 1860.0,
                "revol.bal": 11214,
                "revol.util": 70.1,
                "inq.last.6mths": 0,
                "delinq.2yrs": 0,
                "pub.rec": 0,
            }
        ]
    )


def test_model_loads():

    final_model, explanation_model = load_models()

    assert final_model is not None
    assert explanation_model is not None


def test_prediction_probability_is_valid():

    final_model, _ = load_models()

    applicant = build_sample_applicant()

    prediction = predict_risk(
        applicant,
        final_model,
    )

    probability = prediction["probability"]

    assert 0.0 <= probability <= 1.0


def test_prediction_is_deterministic():

    final_model, _ = load_models()

    applicant = build_sample_applicant()

    prediction_1 = predict_risk(
        applicant,
        final_model,
    )

    prediction_2 = predict_risk(
        applicant,
        final_model,
    )

    assert prediction_1["probability"] == (
        prediction_2["probability"]
    )