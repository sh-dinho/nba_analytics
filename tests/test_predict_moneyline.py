from src.model.predict import _load_model_and_meta


def test_moneyline_model_loads(monkeypatch):
    # Patch model loader to return dummy model
    class DummyModel:
        def predict(self, X):
            return [0.7] * len(X)

    monkeypatch.setattr("src.model.predict.pd.read_pickle", lambda _: DummyModel())
    model, meta = _load_model_and_meta()
    assert model is not None
