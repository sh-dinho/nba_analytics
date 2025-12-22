from src.model.predict_spread import _load_model


def test_spread_model_load(monkeypatch):
    monkeypatch.setattr("src.model.predict_spread.pd.read_pickle", lambda _: "dummy")
    assert _load_model({"path": "fake.pkl"}) == "dummy"
