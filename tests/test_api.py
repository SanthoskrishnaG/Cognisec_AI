import io
import json
import pytest
from src.api.server import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models" in data
    assert data["models"]["xgboost"]["loaded"] is True
    assert data["models"]["vae"]["loaded"] is True
    assert data["models"]["preprocessor"]["loaded"] is True
    assert data["models"]["preprocessor"]["feature_count"] == 70

def test_api_metadata(client):
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.get_json()
    assert data["dataset_name"].startswith("CICIDS2017")
    assert data["retained_clean_features_count"] == 70
    assert len(data["feature_columns"]) == 70
    assert "partitioning" in data
    assert data["research_integrity"]["train_test_identical_vectors_concern"] == 14682

def test_api_experiments(client):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.get_json()
    assert "experiment_a_known" in data
    assert "experiment_b_held_out" in data
    assert "experiment_c_sensitivity" in data

def test_api_comparison(client):
    response = client.get("/api/results/comparison")
    assert response.status_code == 200
    data = response.get_json()
    assert "known_attack_metrics" in data
    assert "per_family_results" in data
    assert "threshold_sensitivity" in data
    assert "distribution_shift" in data
    assert "complementarity" in data

def test_api_sample_csv(client):
    response = client.get("/api/sample-csv")
    assert response.status_code == 200
    data = response.get_json()
    assert data["rows_count"] > 0
    assert len(data["csv_text"]) > 0

def test_csv_preview_with_sample(client):
    sample_resp = client.get("/api/sample-csv")
    csv_text = sample_resp.get_json()["csv_text"]

    response = client.post(
        "/api/analyze/csv-preview",
        data={"file": (io.BytesIO(csv_text.encode("utf-8")), "sample.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["is_compatible"] is True
    assert data["total_rows"] > 0
    assert data["matched_feature_count"] >= 70

def test_csv_preview_malformed(client):
    response = client.post(
        "/api/analyze/csv-preview",
        data={"file": (io.BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 400

    response = client.post(
        "/api/analyze/csv-preview",
        data={"file": (io.BytesIO(b"random,binary,junk\x00\x01\x02"), "file.txt")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 400

def test_xgboost_inference(client):
    sample_resp = client.get("/api/sample-csv")
    csv_text = sample_resp.get_json()["csv_text"]

    response = client.post(
        "/api/analyze/xgboost",
        data={"file": (io.BytesIO(csv_text.encode("utf-8")), "sample.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_flows"] > 0
    assert "flagged_attacks" in data
    assert "benign_flows" in data
    assert len(data["records"]) > 0
    first_record = data["records"][0]
    assert "attack_probability" in first_record
    assert "prediction" in first_record

def test_vae_inference(client):
    sample_resp = client.get("/api/sample-csv")
    csv_text = sample_resp.get_json()["csv_text"]

    response = client.post(
        "/api/analyze/vae",
        data={"file": (io.BytesIO(csv_text.encode("utf-8")), "sample.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_flows"] > 0
    assert "flagged_anomalies" in data
    assert "normal_flows" in data
    assert len(data["records"]) > 0
    first_record = data["records"][0]
    assert "anomaly_score" in first_record
    assert "calibrated_threshold" in first_record

def test_dual_inference(client):
    sample_resp = client.get("/api/sample-csv")
    csv_text = sample_resp.get_json()["csv_text"]

    response = client.post(
        "/api/analyze/dual",
        data={"file": (io.BytesIO(csv_text.encode("utf-8")), "sample.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_flows"] > 0
    assert "summary" in data
    assert "both_flagged" in data["summary"]
    assert "xgboost_only" in data["summary"]
    assert "vae_only" in data["summary"]
    assert "both_normal" in data["summary"]

def test_static_frontend_routes(client):
    # Test index.html
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert b"Cognisec AI" in resp_index.data

    # Test css
    resp_css = client.get("/css/styles.css")
    assert resp_css.status_code == 200
    assert b"--bg-primary" in resp_css.data

    # Test js modules
    resp_app = client.get("/js/app.js")
    assert resp_app.status_code == 200
    assert b"AppState" in resp_app.data

    resp_api = client.get("/js/api.js")
    assert resp_api.status_code == 200
    assert b"API" in resp_api.data

    resp_charts = client.get("/js/charts.js")
    assert resp_charts.status_code == 200
    assert b"Charts" in resp_charts.data

    # Test figure route
    resp_fig = client.get("/figures/fig2_roc_pr_curves.png")
    assert resp_fig.status_code == 200
    assert resp_fig.content_type.startswith("image/")
