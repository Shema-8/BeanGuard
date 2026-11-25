"""Pure helper functions for the dashboard (no Streamlit import, so they are unit-testable)."""
import html

APP_NAME = "BeanGuard"
APP_TAGLINE = "Spot bean leaf diseases early - and keep the model learning from the field."

# Display info per class. Unknown classes fall back to a neutral card, so nothing breaks if you add classes.
CLASS_INFO = {
    "healthy": {
        "title": "Healthy leaf", "emoji": "🌱", "color": "#2E7D32",
        "advice": "No disease detected. Keep monitoring the crop regularly and keep fields free of weeds.",
    },
    "angular_leaf_spot": {
        "title": "Angular leaf spot", "emoji": "🍂", "color": "#B7791F",
        "advice": "Fungal disease causing angular brown spots between leaf veins. Remove infected plant "
                  "debris, rotate crops, use clean seed and avoid working in wet fields.",
    },
    "bean_rust": {
        "title": "Bean rust", "emoji": "🟠", "color": "#C2410C",
        "advice": "Fungal disease with small rust-coloured pustules on leaves. Use resistant varieties, "
                  "rotate crops, and ask a local agronomist about treatment if it is severe.",
    },
}
FALLBACK_COLORS = ["#2563EB", "#7C3AED", "#0D9488", "#DB2777", "#65A30D", "#475569"]


def class_info(label: str) -> dict:
    if label in CLASS_INFO:
        return CLASS_INFO[label]
    idx = sum(map(ord, label)) % len(FALLBACK_COLORS)
    return {"title": label.replace("_", " ").title(), "emoji": "🍃",
            "color": FALLBACK_COLORS[idx], "advice": ""}


def color_map(classes) -> dict:
    return {c: class_info(c)["color"] for c in classes}


def fmt_uptime(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def result_card_html(res: dict) -> str:
    """The big diagnosis card shown after a prediction."""
    info = class_info(res["label"])
    conf = res["confidence"]
    if res.get("uncertain"):
        return (
            '<div class="result" style="background:#64748B">'
            '<div class="r-small">Not sure</div>'
            '<div class="r-big">🤔 Low confidence</div>'
            f'<div class="r-small">Best guess: {html.escape(info["title"])} ({conf:.0%}). '
            'This may not be a bean leaf, or the photo is unclear - try a closer, well-lit picture.</div>'
            '</div>')
    return (
        f'<div class="result" style="background:{info["color"]}">'
        '<div class="r-small">Diagnosis</div>'
        f'<div class="r-big">{info["emoji"]} {html.escape(info["title"])}</div>'
        f'<div class="r-small">{conf:.1%} confidence &middot; model v{res.get("model_version")}</div>'
        '</div>')


def status_pill_html(online: bool) -> str:
    color, text = ("#2E7D32", "● API online") if online else ("#B91C1C", "● API offline")
    return f'<span class="pill" style="background:{color}1A;color:{color}">{text}</span>'
