"""
Web API and Backend Integration Test Suite
File: tests/test_web_api.py
Project: AI-Driven Anomaly Detection in Component Burn-In & Screening (SIH 2026)
"""

import os
import sys
import json
import pytest

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from web.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test 1: Verifies /api/health returns status healthy and active modules."""
    res = client.get('/api/health')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_get_test_components(client):
    """Test 2: Verifies /api/test-components returns real components from locked test set."""
    res = client.get('/api/test-components?limit=10')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "components" in data
    assert len(data["components"]) == 10
    first = data["components"][0]
    assert first["component_id"].startswith("SYN_C")
    assert first["module_a_label"] in [0, 1]


def test_get_component_measurements(client):
    """Test 3: Verifies /api/component/<id> returns real physical channels."""
    res = client.get('/api/component/SYN_C01216')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["component_id"] == "SYN_C01216"
    assert "measurements_0h" in data
    assert "measurements_24h" in data
    assert "measurements_96h" in data
    assert "ground_truth" in data
    assert data["measurements_0h"]["iddq_uA_0h"] is not None


def test_screen_24h_endpoint(client):
    """Test 4: Verifies /api/screen/24h executes A24 and B24 models with 11 features."""
    payload = {"component_id": "SYN_C01216"}
    res = client.post('/api/screen/24h', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "defect_probability" in data
    assert "predicted_168h_iddq_drift" in data
    assert data["decision"] in ["PASS", "REVIEW", "REJECT"]
    assert data["screening_gate"] == "24h"
    assert data["num_features_used"] == 11


def test_screen_96h_endpoint(client):
    """Test 5: Verifies /api/screen/96h executes A96 and B96 models with 19 features."""
    payload = {"component_id": "SYN_C01216"}
    res = client.post('/api/screen/96h', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "defect_probability" in data
    assert "predicted_168h_iddq_drift" in data
    assert data["decision"] in ["PASS", "REVIEW", "REJECT"]
    assert data["screening_gate"] == "96h"
    assert data["num_features_used"] == 19


def test_sequential_screening_endpoint(client):
    """Test 6: Verifies /api/screen/sequential executes 2-stage workflow."""
    payload = {"component_id": "SYN_C01216"}
    res = client.post('/api/screen/sequential', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "final_decision" in data
    assert "final_screening_gate" in data
    assert "stage_1_24h" in data


def test_model_performance_endpoint(client):
    """Test 7: Verifies /api/model-performance returns documented locked test metrics."""
    res = client.get('/api/model-performance')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "module_a" in data
    assert "module_b" in data
    assert data["module_a"]["a24"]["recall"] == 0.7378
    assert data["module_a"]["a96"]["recall"] == 0.9400
    assert data["module_b"]["b24"]["r2_score"] == 0.7890
    assert data["module_b"]["b96"]["r2_score"] == 0.9740


def test_index_page_serves_successfully(client):
    """Test 8: Verifies GET / renders the HTML template with 200 OK."""
    res = client.get('/')
    assert res.status_code == 200
    html = res.data.decode('utf-8')
    assert "AI-Driven Burn-In Screening" in html
    assert "SIH 2026" in html


if __name__ == '__main__':
    pytest.main(["-v", __file__])
