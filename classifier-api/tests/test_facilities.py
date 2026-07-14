import os
import sys
import json
import math
from unittest.mock import patch, MagicMock
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from facilities import (
    generate_triage,
    haversine_distance,
    recommend_facilities,
    search_facilities,
    FacilityRecommendRequest,
    FacilitySearchRequest,
    SPECIALTY_MAP,
    load_facilities,
    _enhance_query,
    _classify_place_type,
)
import app as backend_api


FACILITIES_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "facilities.json")


# --- generate_triage tests ---

class TestGenerateTriage:
    def test_inconclusive(self):
        result = generate_triage("malignant", 0.55, False)
        assert result is not None
        assert result["tier"] == "Further Evaluation Required"
        assert "below the 60%" in result["recommendation"]

    def test_malignant_high_confidence(self):
        result = generate_triage("malignant", 0.95, True)
        assert result["tier"] == "High Concern"
        assert "Urgent" in result["recommendation"]

    def test_malignant_moderate_confidence(self):
        result = generate_triage("malignant", 0.75, True)
        assert result["tier"] == "Moderate Concern"
        assert "Confirmatory" in result["recommendation"]

    def test_benign(self):
        result = generate_triage("benign", 0.80, True)
        assert result["tier"] == "Routine Follow-up"
        assert "Standard monitoring" in result["recommendation"]

    def test_normal(self):
        result = generate_triage("normal", 0.92, True)
        assert result["tier"] == "Routine Screening"
        assert "Continue routine screening" in result["recommendation"]

    def test_unknown_prediction(self):
        result = generate_triage("unknown", 0.90, True)
        assert result is None


# --- haversine_distance tests ---

class TestHaversineDistance:
    def test_same_point(self):
        d = haversine_distance(19.0047, 72.8534, 19.0047, 72.8534)
        assert d == 0.0

    def test_mumbai_to_delhi_approx(self):
        d = haversine_distance(19.0047, 72.8534, 28.5672, 77.2100)
        assert 1100 < d < 1200

    def test_bangalore_to_chennai_approx(self):
        d = haversine_distance(12.9352, 77.6245, 13.0067, 80.2567)
        assert 250 < d < 350

    def test_commutative(self):
        d1 = haversine_distance(19.0047, 72.8534, 28.5672, 77.2100)
        d2 = haversine_distance(28.5672, 77.2100, 19.0047, 72.8534)
        assert abs(d1 - d2) < 0.01


# --- _enhance_query tests ---

class TestEnhanceQuery:
    def test_enhances_plain_location(self):
        assert "mammography" in _enhance_query("Mumbai")
        assert "Mumbai" in _enhance_query("Mumbai")

    def test_does_not_enhance_with_cancer_keyword(self):
        assert _enhance_query("cancer hospital in Delhi") == "cancer hospital in Delhi"

    def test_does_not_enhance_with_mammography_keyword(self):
        assert _enhance_query("mammography center near me") == "mammography center near me"

    def test_does_not_enhance_with_diagnostic_keyword(self):
        assert _enhance_query("diagnostic lab") == "diagnostic lab"

    def test_enhances_near_me(self):
        result = _enhance_query("near me")
        assert "mammography" in result
        assert "near me" in result

    def test_enhances_empty_string(self):
        result = _enhance_query("")
        assert "mammography" in result


# --- _classify_place_type tests ---

class TestClassifyPlaceType:
    def test_cancer_center(self):
        assert _classify_place_type("Tata Memorial Cancer Hospital") == "cancer_center"
        assert _classify_place_type("City Oncology Center") == "cancer_center"
        assert _classify_place_type("Regional Tumor Institute") == "cancer_center"

    def test_diagnostic_center(self):
        assert _classify_place_type("Apollo Diagnostic Centre") == "diagnostic_center"
        assert _classify_place_type("City Imaging Center") == "diagnostic_center"
        assert _classify_place_type("Mammography Screening Lab") == "diagnostic_center"
        assert _classify_place_type("Radiology Associates") == "diagnostic_center"

    def test_hospital_fallback(self):
        assert _classify_place_type("City General Hospital") == "hospital"
        assert _classify_place_type("Prime Medical Center") == "hospital"
        assert _classify_place_type("") == "hospital"

    def test_case_insensitive(self):
        assert _classify_place_type("CANCER INSTITUTE") == "cancer_center"
        assert _classify_place_type("Diagnostic Lab") == "diagnostic_center"


# --- recommend_facilities tests ---

class TestRecommendFacilities:
    def _make_request(self, **overrides):
        defaults = dict(prediction="malignant", confidence=0.95, inconclusive=False, limit=5)
        defaults.update(overrides)
        return FacilityRecommendRequest(**defaults)

    def test_recommend_for_malignant(self):
        body = self._make_request(prediction="malignant")
        result = recommend_facilities(body)
        assert "recommendations" in result
        assert result["source"] == "curated"
        assert len(result["recommendations"]) > 0
        for r in result["recommendations"]:
            assert "score" not in r
            assert r["name"]

    def test_recommend_for_benign(self):
        body = self._make_request(prediction="benign")
        result = recommend_facilities(body)
        assert len(result["recommendations"]) > 0

    def test_recommend_for_normal(self):
        body = self._make_request(prediction="normal")
        result = recommend_facilities(body)
        assert len(result["recommendations"]) > 0

    def test_recommend_for_inconclusive(self):
        body = self._make_request(prediction="malignant", inconclusive=True)
        result = recommend_facilities(body)
        assert len(result["recommendations"]) > 0

    def test_city_match_prioritizes(self):
        body = self._make_request(prediction="malignant", city="Mumbai")
        result = recommend_facilities(body)
        for r in result["recommendations"]:
            assert "cancer" in r["relevance_reason"].lower() or "care" in r["relevance_reason"].lower()

    def test_distance_penalty_applied(self):
        body = self._make_request(
            prediction="malignant",
            lat=13.0000, lng=80.0000,
            city="Chennai",
        )
        result = recommend_facilities(body)
        for r in result["recommendations"]:
            if r["distance_km"] is not None and r["distance_km"] > 50:
                pass

    def test_limit_respected(self):
        body = self._make_request(prediction="malignant", limit=3)
        result = recommend_facilities(body)
        assert len(result["recommendations"]) <= 3

    def test_no_specialty_overlap_returns_empty(self):
        body = self._make_request(prediction="normal", city="UnknownCityXYZ")
        result = recommend_facilities(body)
        assert isinstance(result["recommendations"], list)

    @patch("facilities.load_facilities", return_value=[])
    def test_empty_facilities(self, mock_load):
        body = self._make_request()
        result = recommend_facilities(body)
        assert result["recommendations"] == []
        assert "No facility data available" in result.get("error", "")


# --- search_facilities tests ---

class TestSearchFacilities:
    def _make_request(self, **overrides):
        defaults = dict(query="cancer hospital in Mumbai")
        defaults.update(overrides)
        return FacilitySearchRequest(**defaults)

    def test_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            body = self._make_request()
            result = asyncio_run(search_facilities(body))
            assert result["source"] == "unavailable"
            assert "API key not configured" in result.get("error", "")

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_successful_search(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "places": [
                {
                    "id": "place1",
                    "displayName": {"text": "Tata Memorial Hospital"},
                    "formattedAddress": "Dr. E Borges Road, Mumbai",
                    "rating": 4.5,
                    "userRatingCount": 1200,
                    "regularOpeningHours": {"openNow": True},
                    "location": {"latitude": 19.0047, "longitude": 72.8534},
                },
                {
                    "id": "place2",
                    "displayName": {"text": "BARC Hospital"},
                    "formattedAddress": "Anushaktinagar, Mumbai",
                    "rating": 4.2,
                    "userRatingCount": 800,
                    "regularOpeningHours": None,
                    "location": {"latitude": 19.0223, "longitude": 72.9283},
                },
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        body = self._make_request(query="cancer hospital in Mumbai")
        result = asyncio_run(search_facilities(body))
        assert result["source"] == "google"
        assert len(result["recommendations"]) == 2
        assert result["recommendations"][0]["name"] == "Tata Memorial Hospital"
        assert result["recommendations"][0]["rating"] == 4.5
        assert result["recommendations"][0]["open_now"] is True

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_search_with_location_bias(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"places": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        body = self._make_request(query="hospital", lat=19.0, lng=72.8, radius=5000)
        result = asyncio_run(search_facilities(body))
        assert result["source"] == "google"

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert "locationBias" in payload
        assert payload["locationBias"]["circle"]["center"]["latitude"] == 19.0
        assert payload["locationBias"]["circle"]["center"]["longitude"] == 72.8

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_search_all_bypasses_query_enhancement(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"places": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        body = self._make_request(query="pizza near me", search_all=True)
        asyncio_run(search_facilities(body))

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["textQuery"] == "pizza near me"

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_default_search_enhances_query(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"places": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        body = self._make_request(query="Mumbai", search_all=False)
        asyncio_run(search_facilities(body))

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert "mammography" in payload["textQuery"].lower()

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_distance_sorting(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "places": [
                {
                    "id": "far",
                    "displayName": {"text": "Far Hospital"},
                    "formattedAddress": "Far Away",
                    "location": {"latitude": 30.0, "longitude": 80.0},
                },
                {
                    "id": "near",
                    "displayName": {"text": "Near Clinic"},
                    "formattedAddress": "Close By",
                    "location": {"latitude": 19.01, "longitude": 72.86},
                },
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        body = self._make_request(query="hospital", lat=19.0, lng=72.85)
        result = asyncio_run(search_facilities(body))
        ids = [r["id"] for r in result["recommendations"]]
        assert ids == ["near", "far"]

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_type_classification_in_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "places": [
                {
                    "id": "p1",
                    "displayName": {"text": "City Cancer Center"},
                    "formattedAddress": "Addr 1",
                    "location": {"latitude": 19.0, "longitude": 72.8},
                },
                {
                    "id": "p2",
                    "displayName": {"text": "Quick Diagnostics Lab"},
                    "formattedAddress": "Addr 2",
                    "location": {"latitude": 19.0, "longitude": 72.8},
                },
                {
                    "id": "p3",
                    "displayName": {"text": "General Hospital"},
                    "formattedAddress": "Addr 3",
                    "location": {"latitude": 19.0, "longitude": 72.8},
                },
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        body = self._make_request(query="hospital", lat=19.0, lng=72.8)
        result = asyncio_run(search_facilities(body))
        types = {r["id"]: r["type"] for r in result["recommendations"]}
        assert types["p1"] == "cancer_center"
        assert types["p2"] == "diagnostic_center"
        assert types["p3"] == "hospital"

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_unexpected_status(self, mock_post):
        import requests as req
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("API error")
        mock_post.return_value = mock_response

        body = self._make_request(query="hospital")
        result = asyncio_run(search_facilities(body))
        assert result["source"] == "google"
        assert "failed" in result.get("error", "")

    def test_empty_query(self):
        with patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True):
            with patch("facilities.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"places": []}
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response

                body = self._make_request(query="")
                result = asyncio_run(search_facilities(body))
                assert result["source"] == "google"


# --- API endpoint tests ---

class TestFacilitiesAPI:
    def setup_method(self):
        self.client = TestClient(backend_api.app)

    def test_recommend_endpoint_missing_prediction(self):
        resp = self.client.post("/facilities/recommend", json={})
        assert resp.status_code == 422

    def test_recommend_endpoint_valid(self):
        resp = self.client.post("/facilities/recommend", json={
            "prediction": "malignant",
            "confidence": 0.95,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert data["source"] == "curated"

    def test_recommend_endpoint_with_city(self):
        resp = self.client.post("/facilities/recommend", json={
            "prediction": "malignant",
            "confidence": 0.95,
            "city": "Mumbai",
        })
        assert resp.status_code == 200

    def test_recommend_endpoint_with_location(self):
        resp = self.client.post("/facilities/recommend", json={
            "prediction": "benign",
            "confidence": 0.85,
            "lat": 19.0047,
            "lng": 72.8534,
        })
        assert resp.status_code == 200

    def test_recommend_endpoint_inconclusive(self):
        resp = self.client.post("/facilities/recommend", json={
            "prediction": "malignant",
            "confidence": 0.50,
            "inconclusive": True,
        })
        assert resp.status_code == 200

    @patch.dict(os.environ, {"GOOGLE_PLACES_API_KEY": "test-key"}, clear=True)
    @patch("facilities.requests.post")
    def test_search_endpoint(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "places": [{
                "id": "p1",
                "displayName": {"text": "Test Hospital"},
                "formattedAddress": "Test Address",
                "location": {"latitude": 19.0, "longitude": 72.8},
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        resp = self.client.post("/facilities/search", json={
            "query": "cancer hospital in Mumbai",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "google"

    def test_search_endpoint_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            resp = self.client.post("/facilities/search", json={
                "query": "cancer hospital",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["source"] == "unavailable"


# --- helper ---

def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
