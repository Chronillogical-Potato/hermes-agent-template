"""Regression coverage for the template's Hermes v2026.9.21 integration.

Uses only stdlib unittest so the same file can run on the host and inside the
release image without installing the upstream development/test extras.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse


ROOT = Path(__file__).resolve().parents[1]


def load_server(home: Path):
    """Import a fresh server.py instance bound to ``home``."""
    name = f"server_v2026_9_21_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "server.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UpgradeServerMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hermes-template-test-")
        self.home = Path(self.tmp.name) / ".hermes"
        self.home.mkdir(parents=True)
        self.env = patch.dict(
            os.environ,
            {
                "HERMES_HOME": str(self.home),
                "ADMIN_PASSWORD": "test-password",
                "HERMES_REF": "v2026.9.21",
            },
            clear=True,
        )
        self.env.start()
        self.server = load_server(self.home)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class BackupExclusionTests(UpgradeServerMixin, unittest.TestCase):
    def _touch(self, rel: str) -> None:
        path = self.home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def test_root_and_profile_backup_exclusions_match_upstream(self):
        excluded = {
            "models/model.db",
            "profiles/work/runtimes/runtime.db",
            "browser_profiles/Cookies.db",
            "profiles/work/browser_profiles/History.db",
            "cache/scratch/temp.db",
            "cache/terminal/job.db",
            "profiles/work/cache/web/results.db",
        }
        included = {
            "state.db",
            "cache/images/media.db",
            "profiles/work/cache/citations/evidence.db",
            "skills/example/models/skill.db",
            "profiles/work/skills/example/cache/skill-cache.db",
        }
        for rel in excluded | included:
            self._touch(rel)

        self.assertEqual(
            self.server._live_db_names(),
            {Path(rel).name for rel in included},
        )

    def test_kept_cache_subdirs_are_positive_allowlist(self):
        kept = {"images", "audio", "videos", "documents", "screenshots", "citations"}
        for subdir in kept:
            self.assertFalse(
                self.server._in_excluded_root_dir(Path("cache") / subdir / "data.db"),
                subdir,
            )
            self.assertFalse(
                self.server._in_excluded_root_dir(
                    Path("profiles") / "work" / "cache" / subdir / "data.db"
                ),
                subdir,
            )
        self.assertTrue(self.server._in_excluded_root_dir(Path("cache/scratch/data.db")))


class EnvironmentTests(UpgradeServerMixin, unittest.TestCase):
    def test_generic_temp_defaults_to_ephemeral_container_dir(self):
        env = self.server.build_hermes_env()
        self.assertEqual(env["TERMINAL_TEMP_DIR"], "/tmp")
        self.assertEqual(env["TMPDIR"], "/tmp")

    def test_explicit_persistent_env_override_wins(self):
        self.server.ENV_FILE.write_text("TMPDIR=/operator/temp\n", encoding="utf-8")
        env = self.server.build_hermes_env()
        self.assertEqual(env["TMPDIR"], "/operator/temp")

    def test_dashboard_session_token_survives_in_container_respawns(self):
        first = self.server.build_dashboard_env()
        second = self.server.build_dashboard_env()
        self.assertTrue(first["HERMES_DASHBOARD_SESSION_TOKEN"])
        self.assertEqual(
            first["HERMES_DASHBOARD_SESSION_TOKEN"],
            second["HERMES_DASHBOARD_SESSION_TOKEN"],
        )
        self.assertNotIn("HERMES_DASHBOARD_SESSION_TOKEN", self.server.build_hermes_env())

    def test_untrusted_persisted_dashboard_session_token_is_sanitized(self):
        self.server.ENV_FILE.write_text(
            "HERMES_DASHBOARD_SESSION_TOKEN=stale-or-injected\nKEEP_ME=yes\n",
            encoding="utf-8",
        )
        self.server._sanitize_env_file()
        persisted = self.server.read_env(self.server.ENV_FILE)
        self.assertNotIn("HERMES_DASHBOARD_SESSION_TOKEN", persisted)
        self.assertEqual(persisted["KEEP_ME"], "yes")


class DashboardLifecycleTests(UpgradeServerMixin, unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_stop_uses_v2026_9_21_safe_grace(self):
        seen: dict[str, float] = {}

        class FakeProcess:
            returncode = None

            def __init__(self):
                self.terminated = False
                self.killed = False

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

            async def wait(self):
                self.returncode = 0
                return 0

        async def wait_for(awaitable, timeout):
            seen["timeout"] = timeout
            return await awaitable

        dashboard = self.server.Dashboard()
        process = FakeProcess()
        dashboard.proc = process
        with patch.object(self.server.asyncio, "wait_for", wait_for):
            await dashboard.stop()

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertGreaterEqual(seen["timeout"], 10)


class WebSocketLifecycleTests(UpgradeServerMixin, unittest.TestCase):
    def test_abrupt_upstream_close_uses_spa_reconnect_code(self):
        for code in (None, 1005, 1006, 1012, 1015, 999, 5000):
            self.assertEqual(self.server._browser_ws_close_code(code), 1001)

    def test_valid_upstream_close_codes_are_preserved(self):
        for code in (1000, 1001, 1011, 1013, 4410):
            self.assertEqual(self.server._browser_ws_close_code(code), code)


class BackupEndpointTests(UpgradeServerMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.server.guard = lambda _request: None
        self.server.BACKUP_DIR = self.home / "backups"
        self.work = Path(self.tmp.name) / "download"
        self.work.mkdir()

    async def test_exit_one_returns_readable_archive_with_warning(self):
        async def run_hermes(*args, **_kwargs):
            self.assertEqual(args[:2], ("backup", "-o"))
            with zipfile.ZipFile(Path(args[2]), "w") as zf:
                zf.writestr("config.yaml", "model: {}\n")
            return self.server.BACKUP_INCOMPLETE_RC, "1 file could not be added"

        async def version():
            return "Hermes 0.21.4"

        self.server._run_hermes_cli = run_hermes
        self.server._hermes_version = version
        self.server._live_db_names = lambda: set()

        with patch.object(self.server.tempfile, "mkdtemp", return_value=str(self.work)):
            response = await self.server.api_backup_download(object())
        self.assertIsInstance(response, FileResponse)
        self.assertIn("incomplete", response.headers["x-backup-warning"].lower())
        with zipfile.ZipFile(response.path) as zf:
            self.assertIn("config.yaml", zf.namelist())
            self.assertIn("template_manifest.json", zf.namelist())

    async def test_exit_one_without_archive_is_hard_failure(self):
        async def run_hermes(*_args, **_kwargs):
            return self.server.BACKUP_INCOMPLETE_RC, "failed before archive creation"

        self.server._run_hermes_cli = run_hermes
        with patch.object(self.server.tempfile, "mkdtemp", return_value=str(self.work)):
            response = await self.server.api_backup_download(object())
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 500)

    async def test_exit_two_lock_collision_remains_retryable_conflict(self):
        async def run_hermes(*_args, **_kwargs):
            return self.server.BACKUP_BUSY_RC, "another Hermes backup is already running"

        self.server._run_hermes_cli = run_hermes
        with patch.object(self.server.tempfile, "mkdtemp", return_value=str(self.work)):
            response = await self.server.api_backup_download(object())
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 409)
        self.assertIn("try again", json.loads(response.body)["error"].lower())

    async def test_exit_zero_missing_live_database_keeps_existing_warning(self):
        async def run_hermes(*args, **_kwargs):
            with zipfile.ZipFile(Path(args[2]), "w") as zf:
                zf.writestr("config.yaml", "model: {}\n")
            return 0, "Backup complete"

        async def version():
            return "Hermes 0.21.4"

        self.server._run_hermes_cli = run_hermes
        self.server._hermes_version = version
        self.server._live_db_names = lambda: {"state.db"}
        with patch.object(self.server.tempfile, "mkdtemp", return_value=str(self.work)):
            response = await self.server.api_backup_download(object())
        self.assertIsInstance(response, FileResponse)
        self.assertIn("state.db", response.headers["x-backup-warning"])

    async def test_exit_zero_complete_archive_has_no_warning(self):
        async def run_hermes(*args, **_kwargs):
            with zipfile.ZipFile(Path(args[2]), "w") as zf:
                zf.writestr("config.yaml", "model: {}\n")
                zf.writestr("state.db", b"sqlite-placeholder")
            return 0, "Backup complete"

        async def version():
            return "Hermes 0.21.4"

        self.server._run_hermes_cli = run_hermes
        self.server._hermes_version = version
        self.server._live_db_names = lambda: {"state.db"}
        with patch.object(self.server.tempfile, "mkdtemp", return_value=str(self.work)):
            response = await self.server.api_backup_download(object())
        self.assertIsInstance(response, FileResponse)
        self.assertNotIn("x-backup-warning", response.headers)

    async def test_unreadable_exit_one_archive_is_hard_failure(self):
        async def run_hermes(*args, **_kwargs):
            Path(args[2]).write_bytes(b"not a zip")
            return self.server.BACKUP_INCOMPLETE_RC, "partial"

        self.server._run_hermes_cli = run_hermes
        with patch.object(self.server.tempfile, "mkdtemp", return_value=str(self.work)):
            response = await self.server.api_backup_download(object())
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 500)
        self.assertIn("could not be read", json.loads(response.body)["error"].lower())

    async def test_pre_restore_snapshot_still_fails_closed_on_exit_one(self):
        incoming = io.BytesIO()
        with zipfile.ZipFile(incoming, "w") as zf:
            zf.writestr("config.yaml", "model: {}\n")
        upload = UploadFile(filename="backup.zip", file=io.BytesIO(incoming.getvalue()))

        class Request:
            async def form(self):
                return {"file": upload}

        async def run_hermes(*args, **_kwargs):
            self.assertEqual(args[0], "backup")
            with zipfile.ZipFile(Path(args[2]), "w") as zf:
                zf.writestr("config.yaml", "model: {}\n")
            return self.server.BACKUP_INCOMPLETE_RC, "partial safety snapshot"

        self.server._run_hermes_cli = run_hermes
        self.server.gw.stop = AsyncMock(side_effect=AssertionError("gateway must stay up"))
        self.server.dash.stop = AsyncMock(side_effect=AssertionError("dashboard must stay up"))

        response = await self.server.api_backup_restore(Request())
        self.assertEqual(response.status_code, 500)
        self.assertIn("restore aborted", json.loads(response.body)["error"].lower())
        self.server.gw.stop.assert_not_awaited()
        self.server.dash.stop.assert_not_awaited()


class RouterProviderConfigTests(UpgradeServerMixin, unittest.TestCase):
    def _config(self):
        return yaml.safe_load((self.home / "config.yaml").read_text(encoding="utf-8"))

    def test_both_routers_are_named_providers_and_preserve_existing_entries(self):
        (self.home / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {},
                "providers": {
                    "corp-gateway": {
                        "api": "https://corp.example/v1",
                        "key_env": "CORP_GATEWAY_KEY",
                    },
                    "ninerouter": {
                        "api": "https://old.example/v1",
                        "key_env": "NINEROUTER_API_KEY",
                        "extra_headers": {"X-Team": "agents"},
                    },
                },
                "mcp_servers": {"keep-me": {"command": "example"}},
            }),
            encoding="utf-8",
        )
        data = {
            "LLM_MODEL": "cc/claude-sonnet-4-6",
            "NINEROUTER_API_KEY": "nine-secret",
            "NINEROUTER_BASE_URL": "http://9router.railway.internal:20128/v1/",
            "OMNIROUTE_API_KEY": "omni-secret",
            "OMNIROUTE_BASE_URL": "http://omniroute.railway.internal:20128/v1",
        }

        self.server.write_config_yaml(data)
        cfg = self._config()

        self.assertEqual(cfg["model"]["provider"], "ninerouter")
        self.assertEqual(cfg["providers"]["ninerouter"]["api"], "http://9router.railway.internal:20128/v1")
        self.assertEqual(cfg["providers"]["ninerouter"]["key_env"], "NINEROUTER_API_KEY")
        self.assertEqual(cfg["providers"]["ninerouter"]["transport"], "chat_completions")
        self.assertTrue(cfg["providers"]["ninerouter"]["discover_models"])
        self.assertEqual(cfg["providers"]["ninerouter"]["extra_headers"], {"X-Team": "agents"})
        self.assertEqual(cfg["providers"]["omniroute"]["key_env"], "OMNIROUTE_API_KEY")
        self.assertIn("corp-gateway", cfg["providers"])
        self.assertIn("keep-me", cfg["mcp_servers"])
        serialized = (self.home / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("nine-secret", serialized)
        self.assertNotIn("omni-secret", serialized)

    def test_removing_managed_router_does_not_remove_other_providers(self):
        self.server.write_config_yaml({
            "LLM_MODEL": "auto",
            "NINEROUTER_API_KEY": "nine-secret",
            "NINEROUTER_BASE_URL": "https://nine.example/v1",
            "OMNIROUTE_API_KEY": "omni-secret",
            "OMNIROUTE_BASE_URL": "https://omni.example/v1",
        })
        cfg = self._config()
        cfg["providers"]["manual"] = {
            "api": "https://manual.example/v1",
            "key_env": "MANUAL_KEY",
        }
        (self.home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

        self.server.write_config_yaml({
            "LLM_MODEL": "auto",
            "OMNIROUTE_API_KEY": "omni-secret",
            "OMNIROUTE_BASE_URL": "https://omni.example/v1",
        })
        updated = self._config()
        self.assertNotIn("ninerouter", updated["providers"])
        self.assertIn("omniroute", updated["providers"])
        self.assertIn("manual", updated["providers"])

    def test_blank_template_fields_do_not_delete_same_named_manual_provider(self):
        (self.home / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {"default": "manual-model", "provider": "ninerouter"},
                "providers": {
                    "ninerouter": {
                        "name": "Manually managed",
                        "api": "https://manual-nine.example/v1",
                        "key_env": "MANUAL_NINE_KEY",
                    }
                },
            }),
            encoding="utf-8",
        )
        self.server.write_config_yaml({"LLM_MODEL": "manual-model"})
        self.assertEqual(
            self._config()["providers"]["ninerouter"]["key_env"],
            "MANUAL_NINE_KEY",
        )

    def test_router_requires_both_key_and_base_url_for_setup_completion(self):
        self.assertFalse(self.server.is_config_complete({
            "LLM_MODEL": "auto",
            "OMNIROUTE_API_KEY": "secret",
        }))
        self.assertTrue(self.server.is_config_complete({
            "LLM_MODEL": "auto",
            "OMNIROUTE_API_KEY": "secret",
            "OMNIROUTE_BASE_URL": "https://omni.example/v1",
        }))


class RouterProviderEndpointTests(UpgradeServerMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.server.guard = lambda _request: None

    async def test_save_pins_named_router_without_putting_secret_in_request(self):
        class Request:
            async def json(_self):
                return {
                    "vars": {
                        "LLM_MODEL": "cc/claude-sonnet-4-6",
                        "NINEROUTER_API_KEY": "nine-secret",
                        "NINEROUTER_BASE_URL": "https://nine.example/v1",
                    },
                    "_active_provider_key": "NINEROUTER_API_KEY",
                    "_restart": False,
                }

        self.server.set_active_model_via_hermes = AsyncMock(return_value=None)
        response = await self.server.api_config_put(Request())

        self.assertEqual(response.status_code, 200)
        self.server.set_active_model_via_hermes.assert_awaited_once_with(
            "ninerouter",
            "cc/claude-sonnet-4-6",
            base_url="",
            api_key="",
        )
        cfg = yaml.safe_load((self.home / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["providers"]["ninerouter"]["key_env"], "NINEROUTER_API_KEY")

    async def test_invalid_router_url_is_rejected_before_persistence(self):
        class Request:
            async def json(_self):
                return {
                    "vars": {
                        "LLM_MODEL": "auto",
                        "OMNIROUTE_API_KEY": "omni-secret",
                        "OMNIROUTE_BASE_URL": "omniroute:20128/v1",
                    },
                    "_active_provider_key": "OMNIROUTE_API_KEY",
                    "_restart": False,
                }

        response = await self.server.api_config_put(Request())
        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.server.ENV_FILE.exists())
        self.assertIn("complete http:// or https:// URL", json.loads(response.body)["error"])

    async def test_removing_active_router_clears_dangling_active_model(self):
        existing = {
            "LLM_MODEL": "minimax/MiniMax-M3",
            "OMNIROUTE_API_KEY": "omni-secret",
            "OMNIROUTE_BASE_URL": "https://omni.example/v1",
            "_MODEL_OMNIROUTE_API_KEY": "minimax/MiniMax-M3",
        }
        self.server.write_env(self.server.ENV_FILE, existing)
        self.server.write_config_yaml(existing)
        cfg = yaml.safe_load((self.home / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["model"]["provider"], "omniroute")

        class Request:
            async def json(_self):
                return {
                    "vars": {
                        "OMNIROUTE_API_KEY": "",
                        "OMNIROUTE_BASE_URL": "",
                        "_MODEL_OMNIROUTE_API_KEY": "",
                    },
                    "_active_provider_key": "",
                    "_restart": False,
                }

        response = await self.server.api_config_put(Request())
        self.assertEqual(response.status_code, 200)
        env = self.server.read_env(self.server.ENV_FILE)
        self.assertNotIn("LLM_MODEL", env)
        self.assertNotIn("OMNIROUTE_API_KEY", env)
        updated = yaml.safe_load((self.home / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(updated["model"], {"default": ""})
        self.assertNotIn("omniroute", updated.get("providers", {}))


class RouterProviderUiTests(unittest.TestCase):
    def test_setup_page_has_separate_router_fields_and_help(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        for expected in (
            "'9Router'",
            "'OmniRoute'",
            "NINEROUTER_API_KEY",
            "NINEROUTER_BASE_URL",
            "OMNIROUTE_API_KEY",
            "OMNIROUTE_BASE_URL",
            "Dashboard → Endpoints",
            "defaultModel: 'auto'",
        ):
            self.assertIn(expected, template)


class ReleasePinTests(unittest.TestCase):
    def test_dockerfile_pins_target_release(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG HERMES_REF=v2026.9.21", dockerfile)

    def test_dockerfile_pins_fixed_sqlite_runtime(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG SQLITE_AUTOCONF_VERSION=3530400", dockerfile)
        self.assertIn("libsqlite3.so.3.53.4", dockerfile)
        self.assertIn("v < (3, 51, 3)", dockerfile)
        self.assertIn("tokenize='trigram'", dockerfile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
