from src.pipeline import parse_roi_frac


def test_parse_roi_frac_ok():
    roi = parse_roi_frac("0.03,0.14,0.98,0.72")
    assert roi == (0.03, 0.14, 0.98, 0.72)


def test_parse_roi_frac_invalid_count():
    try:
        parse_roi_frac("0.1,0.2,0.3")
        assert False, "Expected ValueError for invalid roi count"
    except ValueError:
        assert True
