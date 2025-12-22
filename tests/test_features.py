from src.features.builder import FeatureBuilder, FeatureConfig


def test_feature_builder_runs(sample_long_df):
    fb = FeatureBuilder(FeatureConfig(version="v1"))
    df = fb.build_from_long(sample_long_df)
    assert not df.empty
    assert "rolling_points_for_5" in df.columns
    assert "rolling_win_rate_10" in df.columns
