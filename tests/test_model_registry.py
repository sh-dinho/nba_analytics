import json

from src.model.registry import load_latest_model_metadata

()


def test_registry_loads_latest_model(tmp_path, monkeypatch):
    # Create fake registry
    registry = {
        "models": [
            {"model_type": "moneyline", "version": "1", "created_at_utc": "2024-01-01"},
            {"model_type": "moneyline", "version": "2", "created_at_utc": "2024-02-01"},
        ]
    }

    # Patch registry path
    fake_index = tmp_path / "index.json"
    fake_index.write_text(json.dumps(registry))

    monkeypatch.setattr("src.model.registry.MODEL_REGISTRY_INDEX", fake_index)

    meta = load_latest_model_metadata("moneyline")
    assert meta["version"] == "2"
