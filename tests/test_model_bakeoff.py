import pytest
from PIL import Image

from track_game.model_bakeoff import (
    ANCHOR_FRAMES,
    MODELS,
    NEW_PAID_CALLS,
    OBSERVED_CONTROL_COST_USD,
    annotate_detection,
    build_cost_preflight,
    run_approved_bakeoff,
    verify_catalog_payload,
)
from track_game.schema import FrameDetection


def _catalog_fixture():
    catalog = []
    endpoints = {}
    for index, model in enumerate(MODELS, start=1):
        parameters = [
            "max_tokens",
            "temperature",
            "response_format",
            "structured_outputs",
            "reasoning",
            "reasoning_effort",
        ]
        catalog.append(
            {
                "id": model.slug,
                "canonical_slug": f"{model.slug}-canonical",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": parameters,
                "reasoning": None,
            }
        )
        endpoints[model.slug] = {
            "endpoints": [
                {
                    "provider_name": f"Provider {index}",
                    "tag": model.provider_slug,
                    "status": 0,
                    "uptime_last_5m": 100,
                    "supported_parameters": parameters,
                    "pricing": {
                        "prompt": "0.00000025",
                        "completion": "0.0000015",
                    },
                }
            ]
        }
    return catalog, endpoints


def test_bakeoff_is_strictly_61_anchors_and_305_new_calls():
    assert len(ANCHOR_FRAMES) == 61
    assert ANCHOR_FRAMES[:3] == (0, 15, 30)
    assert ANCHOR_FRAMES[-2:] == (885, 899)
    assert NEW_PAID_CALLS == 305


def test_catalog_verifier_requires_exact_slugs_and_compatible_provider():
    catalog, endpoints = _catalog_fixture()
    verified = verify_catalog_payload(catalog, endpoints, captured_at="fixture")
    assert verified["all_requested_slugs_exact"] is True
    assert len(verified["models"]) == 6
    assert all(item["chosen_provider_status"] == 0 for item in verified["models"])

    catalog.pop()
    with pytest.raises(RuntimeError, match="slug is unavailable"):
        verify_catalog_payload(catalog, endpoints)


def test_cost_preflight_reproduces_observed_control_cost():
    catalog, endpoints = _catalog_fixture()
    verified = verify_catalog_payload(catalog, endpoints, captured_at="fixture")
    cost = build_cost_preflight(verified)
    control = cost["models"][0]
    assert control["expected_61_frame_cost_usd"] == pytest.approx(
        OBSERVED_CONTROL_COST_USD
    )
    assert cost["new_paid_calls_requiring_approval"] == 305


def test_paid_bakeoff_requires_exact_approval_before_repository_or_network(tmp_path):
    with pytest.raises(PermissionError, match="exactly 305 new calls"):
        run_approved_bakeoff(tmp_path, 304)


def test_raw_detection_annotation_does_not_require_tracker_output():
    detection = FrameDetection.from_dict(
        {
            "frame_id": 30,
            "players": [
                {
                    "detection_id": "player_01",
                    "team": "A",
                    "box": {"left": 0.1, "top": 0.2, "right": 0.2, "bottom": 0.8},
                    "foot": {"x": 0.15, "y": 0.8},
                    "confidence": 0.9,
                    "team_confidence": 0.9,
                }
            ],
            "ball": None,
            "possession": None,
            "uncertainty_notes": [],
        }
    )
    annotated = annotate_detection(Image.new("RGB", (320, 180)), detection)
    assert annotated.size == (320, 180)
