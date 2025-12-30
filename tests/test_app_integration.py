from streamlit.testing.v1 import AppTest


def test_streamlit_app_runs_without_crash():
    at = AppTest.from_file("src/dashboard/app.py")
    at.run()

    assert not at.exception

    # App should render at least one text element
    assert len(at.text) > 0
