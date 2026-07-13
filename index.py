"""Vercel entrypoint: exposes the Dash app's underlying Flask (WSGI) server."""
from app import server as app  # noqa: F401
