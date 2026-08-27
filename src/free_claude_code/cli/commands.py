"""Implementations for installed Free Claude Code commands."""

import threading
import time
import webbrowser
from enum import StrEnum

import uvicorn
from loguru import logger

from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.cli.process_registry import kill_all_best_effort
from free_claude_code.cli.tls_proxy import (
    CaddyTlsProxy,
    desktop_gateway_base_url,
    probe_fcc_front,
    tls_root_url,
)
from free_claude_code.config.loader import (
    clear_settings_cache,
    get_settings,
    repair_invalid_managed_provider_proxies,
)
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.server_urls import local_admin_url, local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.bootstrap import build_asgi_app

SERVER_GRACEFUL_SHUTDOWN_SECONDS = 5
GATEWAY_HEALTH_UPGRADE_SECONDS = 10.0


def serve() -> None:
    """Start and supervise the FastAPI server."""
    ServerSupervisor().run()


class ServerStatus(StrEnum):
    """Observable state of the server owned by a supervisor."""

    STARTING = "Starting"
    RUNNING = "Running"
    STOPPING = "Stopping"
    STOPPED = "Stopped"


class ServerSupervisor:
    """Own one FCC server lifecycle, including config-driven restarts."""

    def __init__(self, *, console_logging: bool = True) -> None:
        self._console_logging = console_logging
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._desktop_gateway_url: str | None = None
        self._run_scheduled = False
        self._running = False
        self._stop_requested = False
        self._restart_generation = 0

    @property
    def status(self) -> ServerStatus:
        with self._lock:
            if self._run_scheduled:
                return ServerStatus.STARTING
            if not self._running:
                return ServerStatus.STOPPED
            if self._server is None:
                return ServerStatus.STARTING
            if self._server.should_exit:
                return ServerStatus.STOPPING
            if self._server.started:
                return ServerStatus.RUNNING
            return ServerStatus.STARTING

    def desktop_gateway_url(self) -> str | None:
        """The desktop-scoped gateway URL the active generation is serving.

        Resolved once the generation's HTTPS front has been started (or the
        plain-HTTP fallback chosen), so Claude Desktop integrations read the
        live endpoint rather than a stale or not-yet-ready value. ``None``
        before the first generation publishes one.
        """

        with self._lock:
            return self._desktop_gateway_url

    def schedule_run(self) -> bool:
        """Reserve a worker run before its thread starts."""

        with self._lock:
            if self._stop_requested or self._run_scheduled or self._running:
                return False
            self._run_scheduled = True
            return True

    def run(self, *, open_admin_browser: bool | None = None) -> None:
        """Block until stopped, applying only fully closed Admin restarts."""

        with self._lock:
            self._run_scheduled = False
            if self._running:
                raise RuntimeError("The FCC server supervisor is already running.")
            if self._stop_requested:
                return
            self._running = True

        opened_admin_browser = False
        try:
            try:
                while not self._is_stop_requested():
                    with self._lock:
                        restart_generation = self._restart_generation
                    settings = load_server_settings()
                    should_open_admin = (
                        settings.open_admin_browser
                        if open_admin_browser is None
                        else open_admin_browser
                    ) and not opened_admin_browser
                    if not self._run_once(
                        settings,
                        open_admin_browser=should_open_admin,
                        restart_generation=restart_generation,
                    ):
                        return
                    opened_admin_browser = opened_admin_browser or should_open_admin
                    clear_settings_cache()
            except KeyboardInterrupt:
                return
        finally:
            with self._lock:
                self._server = None
                self._desktop_gateway_url = None
                self._running = False
            kill_all_best_effort()

    def request_restart(self) -> bool:
        """Reload an active generation or coalesce into a scheduled fresh run."""

        with self._lock:
            if self._stop_requested:
                return False
            if self._run_scheduled:
                self._restart_generation += 1
                return True
            if not self._running:
                return False
            self._restart_generation += 1
            if self._server is not None:
                self._server.should_exit = True
            return True

    def request_stop(self) -> None:
        """Permanently stop this supervisor after graceful runtime cleanup."""

        with self._lock:
            self._stop_requested = True
            self._run_scheduled = False
            if self._server is not None:
                self._server.should_exit = True

    def _is_stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _run_once(
        self,
        settings: Settings,
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        asgi_app = build_asgi_app(
            settings,
            restart_callback=self._request_runtime_restart,
        )
        config = uvicorn.Config(
            asgi_app,
            host=settings.host,
            port=settings.port,
            log_level="debug",
            log_config=(
                uvicorn.config.LOGGING_CONFIG if self._console_logging else None
            ),
            timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_SECONDS,
        )
        server = uvicorn.Server(config)
        with self._lock:
            self._server = server
            if self._stop_requested or self._restart_generation != restart_generation:
                server.should_exit = True

        # Own the managed HTTPS front for exactly this generation: started
        # before the server accepts traffic so Claude Desktop's config-merge
        # probe finds a live TLS endpoint, stopped when the generation ends.
        # ``start()`` never raises and falls back to plain HTTP on failure.
        tls_front = CaddyTlsProxy(settings)
        try:
            front_started = tls_front.start()
            # Publish the desktop-scoped gateway URL this generation serves.
            # Resolution needs FCC's /health marker THROUGH the front, which
            # only answers once Uvicorn accepts the proxied request — so
            # before ``server.run()`` an adopted or freshly started front
            # cannot verify yet and resolution would fall back to plain HTTP.
            # Publish the plain-HTTP fallback first (correct whenever no
            # front is coming), then upgrade to the TLS-prefixed URL once
            # the server serves and the front verifies. Desktop integrations
            # read the live value instead of guessing.
            self._publish_desktop_gateway_url(settings)
            if open_admin_browser:
                schedule_open_admin_browser(settings)
            server.run()
            self._upgrade_desktop_gateway_to_https(settings, front_started)
        finally:
            tls_front.stop()

        with self._lock:
            if self._server is server:
                self._server = None
            restart_requested = self._restart_generation != restart_generation
            stop_requested = self._stop_requested
        return restart_requested and not stop_requested and asgi_app.runtime.is_closed

    def _publish_desktop_gateway_url(self, settings: Settings) -> None:
        """Publish the resolved desktop gateway URL for this generation."""

        gateway_url = desktop_gateway_base_url(settings)
        with self._lock:
            self._desktop_gateway_url = gateway_url
        logger.info("Claude Desktop gateway: {}", gateway_url)

    def _upgrade_desktop_gateway_to_https(
        self,
        settings: Settings,
        front_started: bool,
    ) -> None:
        """Swap the published gateway URL to HTTPS once the front verifies.

        Runs after ``server.run()`` returns: Uvicorn has served and shut
        down, so the front can now pass FCC's ``/health`` marker probe.
        Skipped when no front is up (disabled, failed to start, or an
        adopted front stopped verifying) — the plain-HTTP fallback already
        published then stays, matching the only conditions where HTTP is
        legitimate.
        """

        if not front_started:
            return
        root = tls_root_url(settings)
        deadline = time.monotonic() + GATEWAY_HEALTH_UPGRADE_SECONDS
        while time.monotonic() < deadline:
            if probe_fcc_front(root):
                # The front verified, so publish the TLS-prefixed URL
                # directly — re-resolving would probe the front a second
                # time for the same answer.
                gateway_url = desktop_gateway_base_url(settings, base_url=root)
                with self._lock:
                    self._desktop_gateway_url = gateway_url
                logger.info("Claude Desktop gateway: {}", gateway_url)
                return
            time.sleep(0.25)

    def _request_runtime_restart(self) -> None:
        self.request_restart()


def load_server_settings() -> Settings:
    """Return canonical settings after repairing invalid managed proxies."""

    settings = get_settings()
    removed = repair_invalid_managed_provider_proxies()
    if not removed:
        return settings

    logger.warning(
        "Removed invalid managed provider proxy settings from {}: {}. "
        "Configure valid proxy URLs in Admin if needed.",
        managed_env_path(),
        ", ".join(removed),
    )
    clear_settings_cache()
    return get_settings()


def open_admin_when_ready(settings: Settings) -> bool:
    """Wait briefly for /health, then open the current Admin UI."""

    admin_url = local_admin_url(settings)
    proxy_root_url = local_proxy_root_url(settings)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if preflight_proxy(proxy_root_url) is None:
            return webbrowser.open(admin_url)
        time.sleep(0.15)
    return False


def schedule_open_admin_browser(settings: Settings) -> None:
    """Open Admin after health succeeds without blocking the caller."""

    threading.Thread(
        target=open_admin_when_ready,
        args=(settings,),
        name="fcc-open-admin-browser",
        daemon=True,
    ).start()
