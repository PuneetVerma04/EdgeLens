"""Settings coverage for the values the app reads at startup.

CORS_ORIGINS used to be read straight from os.getenv in main.py, bypassing pydantic-settings.
These tests pin it to the config layer so it cannot drift back out.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings


def test_cors_origins_is_a_settings_field() -> None:
    assert "cors_origins" in Settings.model_fields


def test_cors_origins_splits_on_commas() -> None:
    settings = Settings(cors_origins="http://a.test,http://b.test")

    assert settings.cors_origins_list == ["http://a.test", "http://b.test"]


def test_cors_origins_tolerates_whitespace_and_blanks() -> None:
    settings = Settings(cors_origins=" http://a.test , , http://b.test ,")

    assert settings.cors_origins_list == ["http://a.test", "http://b.test"]


def test_cors_origins_single_value() -> None:
    settings = Settings(cors_origins="http://only.test")

    assert settings.cors_origins_list == ["http://only.test"]


def test_cors_origins_reads_from_environment(monkeypatch) -> None:
    """A CORS_ORIGINS env var reaches Settings the same way the other variables do."""
    monkeypatch.setenv("CORS_ORIGINS", "http://from-env.test")

    assert Settings().cors_origins_list == ["http://from-env.test"]


def test_app_middleware_uses_configured_origins(client) -> None:
    """The running app answers a CORS preflight for a configured origin."""
    origin = get_settings().cors_origins_list[0]

    response = client.options(
        "/api/edgelens/",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_settings_are_cached() -> None:
    """get_settings() is lru_cached, so callers share one instance."""
    assert get_settings() is get_settings()
