"""Compatibility entry point. Render uses ``app:app`` from render.yaml."""

from app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, debug=False)
