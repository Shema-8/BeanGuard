from ui.helpers import class_info, color_map, fmt_uptime, result_card_html, status_pill_html


def test_known_and_unknown_classes():
    assert class_info("healthy")["title"] == "Healthy leaf"
    unknown = class_info("maize_blight")
    assert unknown["title"] == "Maize Blight" and unknown["color"].startswith("#")
    assert set(color_map(["healthy", "bean_rust"])) == {"healthy", "bean_rust"}


def test_result_card_confident_and_uncertain():
    ok = result_card_html({"label": "bean_rust", "confidence": 0.93, "uncertain": False, "model_version": 2})
    assert "Bean rust" in ok and "93.0%" in ok and "model v2" in ok
    unsure = result_card_html({"label": "bean_rust", "confidence": 0.31, "uncertain": True, "model_version": 2})
    assert "Low confidence" in unsure


def test_misc():
    assert fmt_uptime(3725) == "1h 2m 5s"
    assert "offline" in status_pill_html(False) and "online" in status_pill_html(True)
