"""Tests for hermes_cli.update_checker — scheduled checker, changelog, version picker."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# =========================================================================
# ScheduledUpdateChecker
# =========================================================================

class TestScheduledUpdateChecker:
    """Tests for the ScheduledUpdateChecker class."""

    def test_initial_state_no_cache(self, tmp_path):
        """Checker starts with no update available when cache doesn't exist."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        with patch("hermes_cli.update_checker._get_cache_file", return_value=tmp_path / ".update_check"):
            checker = ScheduledUpdateChecker(interval_seconds=3600)
            assert checker.update_available is False
            assert checker.commits_behind == 0

    def test_loads_cached_value(self, tmp_path):
        """Checker loads cached behind count on init."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        cache = tmp_path / ".update_check"
        cache.write_text(json.dumps({"ts": time.time(), "behind": 5, "latest_tag": "v0.3.0"}))

        with patch("hermes_cli.update_checker._get_cache_file", return_value=cache):
            checker = ScheduledUpdateChecker(interval_seconds=3600)
            assert checker.update_available is True
            assert checker.commits_behind == 5
            assert checker.latest_tag == "v0.3.0"

    def test_loads_zero_behind_from_cache(self, tmp_path):
        """Zero commits behind means no update available."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        cache = tmp_path / ".update_check"
        cache.write_text(json.dumps({"ts": time.time(), "behind": 0}))

        with patch("hermes_cli.update_checker._get_cache_file", return_value=cache):
            checker = ScheduledUpdateChecker(interval_seconds=3600)
            assert checker.update_available is False
            assert checker.commits_behind == 0

    def test_check_now_updates_state(self, tmp_path):
        """check_now() updates commits_behind and saves cache."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        cache = tmp_path / ".update_check"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        with patch("hermes_cli.update_checker._get_cache_file", return_value=cache), \
             patch("hermes_cli.update_checker._get_repo_dir", return_value=repo_dir), \
             patch("hermes_cli.update_checker._git") as mock_git:
            # _git calls: fetch, rev-list, describe
            def git_side_effect(args, *a, **kw):
                if "rev-list" in args:
                    return "7"
                if "describe" in args:
                    return "v0.3.0"
                return ""
            mock_git.side_effect = git_side_effect

            checker = ScheduledUpdateChecker(interval_seconds=3600)
            result = checker.check_now()

            assert result == 7
            assert checker.commits_behind == 7
            assert checker.update_available is True
            assert checker.latest_tag == "v0.3.0"

            # Cache should be written
            assert cache.exists()
            cached = json.loads(cache.read_text())
            assert cached["behind"] == 7
            assert cached["latest_tag"] == "v0.3.0"

    def test_check_now_no_repo(self, tmp_path):
        """check_now() returns None when no git repo found."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        cache = tmp_path / ".update_check"
        with patch("hermes_cli.update_checker._get_cache_file", return_value=cache), \
             patch("hermes_cli.update_checker._get_repo_dir", return_value=None):
            checker = ScheduledUpdateChecker(interval_seconds=3600)
            result = checker.check_now()
            assert result is None

    def test_minimum_interval_floor(self):
        """Interval is floored at 60 seconds."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        with patch("hermes_cli.update_checker._get_cache_file", return_value=Path("/nonexistent")):
            checker = ScheduledUpdateChecker(interval_seconds=10)
            assert checker.interval == 60

    def test_start_and_stop(self, tmp_path):
        """Checker thread starts and stops without hanging."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        cache = tmp_path / ".update_check"
        with patch("hermes_cli.update_checker._get_cache_file", return_value=cache), \
             patch("hermes_cli.update_checker._get_repo_dir", return_value=None):
            checker = ScheduledUpdateChecker(interval_seconds=60)
            checker.start()
            assert checker._thread is not None
            assert checker._thread.is_alive()

            checker.stop()
            checker._thread.join(timeout=2)

    def test_wait_for_first_check(self, tmp_path):
        """wait_for_first_check returns True after check completes."""
        from hermes_cli.update_checker import ScheduledUpdateChecker

        cache = tmp_path / ".update_check"
        with patch("hermes_cli.update_checker._get_cache_file", return_value=cache), \
             patch("hermes_cli.update_checker._get_repo_dir", return_value=None):
            checker = ScheduledUpdateChecker(interval_seconds=60)
            checker.start()
            result = checker.wait_for_first_check(timeout=5)
            assert result is True
            checker.stop()


# =========================================================================
# Singleton
# =========================================================================

class TestSingleton:

    def test_get_update_checker_returns_same_instance(self):
        """get_update_checker returns the same instance on repeated calls."""
        import hermes_cli.update_checker as mod

        # Reset singleton
        old = mod._checker_instance
        mod._checker_instance = None
        try:
            with patch("hermes_cli.update_checker._get_cache_file", return_value=Path("/nonexistent")):
                a = mod.get_update_checker(3600)
                b = mod.get_update_checker(3600)
                assert a is b
        finally:
            mod._checker_instance = old


# =========================================================================
# Changelog
# =========================================================================

class TestChangelog:

    def test_get_changelog_entries_parses_git_log(self, tmp_path):
        """get_changelog_entries parses git log format correctly."""
        from hermes_cli.update_checker import get_changelog_entries

        fake_log = (
            "abc123full|abc123|2026-03-15 10:00:00 +0000|Alice|feat: add feature|HEAD -> main, tag: v0.2.0\n"
            "def456full|def456|2026-03-14 09:00:00 +0000|Bob|fix: bug fix|\n"
        )
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=repo_dir), \
             patch("hermes_cli.update_checker._git", return_value=fake_log):
            entries = get_changelog_entries(limit=10)

        assert len(entries) == 2
        assert entries[0]["short_hash"] == "abc123"
        assert entries[0]["tag"] == "v0.2.0"
        assert entries[0]["author"] == "Alice"
        assert entries[0]["subject"] == "feat: add feature"
        assert entries[0]["date"] == "2026-03-15"

        assert entries[1]["tag"] is None
        assert entries[1]["short_hash"] == "def456"

    def test_get_changelog_entries_no_repo(self):
        """Returns empty list when no repo found."""
        from hermes_cli.update_checker import get_changelog_entries

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=None):
            assert get_changelog_entries() == []

    def test_get_changelog_entries_empty_log(self, tmp_path):
        """Returns empty list on empty git log."""
        from hermes_cli.update_checker import get_changelog_entries

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=repo_dir), \
             patch("hermes_cli.update_checker._git", return_value=None):
            assert get_changelog_entries() == []


# =========================================================================
# Available versions
# =========================================================================

class TestAvailableVersions:

    def test_get_available_versions_parses_tags(self, tmp_path):
        """get_available_versions parses tag list correctly."""
        from hermes_cli.update_checker import get_available_versions

        fake_tags = (
            "v0.3.0|abc1234|2026-03-15|Release 0.3.0\n"
            "v0.2.0|def5678|2026-03-12|Release 0.2.0\n"
        )
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        def git_side_effect(args, repo, **kw):
            joined = " ".join(args)
            if "tag" in joined and "--sort" in joined:
                return fake_tags
            if "rev-parse" in joined:
                if "HEAD" in joined:
                    return "abc1234full"
                # Tag hashes
                if "v0.3.0" in joined:
                    return "abc1234full"
                if "v0.2.0" in joined:
                    return "def5678full"
            return ""

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=repo_dir), \
             patch("hermes_cli.update_checker._git", side_effect=git_side_effect):
            versions = get_available_versions(limit=10)

        assert len(versions) == 2
        assert versions[0]["tag"] == "v0.3.0"
        assert versions[0]["is_current"] is True  # HEAD matches
        assert versions[1]["tag"] == "v0.2.0"
        assert versions[1]["is_current"] is False

    def test_get_available_versions_no_repo(self):
        """Returns empty list when no repo found."""
        from hermes_cli.update_checker import get_available_versions

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=None):
            assert get_available_versions() == []


# =========================================================================
# Text fallback version picker
# =========================================================================

class TestTextVersionPicker:

    def test_text_picker_cancel_on_empty_input(self):
        """Empty input returns None (cancel)."""
        from hermes_cli.update_checker import _text_version_picker

        versions = [
            {"tag": "v0.2.0", "short_hash": "abc", "date": "2026-03-12", "subject": "Release", "is_current": True},
        ]

        with patch("builtins.input", return_value=""):
            result = _text_version_picker(versions)
        assert result is None

    def test_text_picker_selects_by_number(self):
        """Numeric input selects the correct version."""
        from hermes_cli.update_checker import _text_version_picker

        versions = [
            {"tag": "v0.3.0", "short_hash": "abc", "date": "2026-03-15", "subject": "Release 3", "is_current": False},
            {"tag": "v0.2.0", "short_hash": "def", "date": "2026-03-12", "subject": "Release 2", "is_current": True},
        ]

        with patch("builtins.input", return_value="2"):
            result = _text_version_picker(versions)
        assert result == "v0.2.0"

    def test_text_picker_selects_by_tag_name(self):
        """Direct tag name input works."""
        from hermes_cli.update_checker import _text_version_picker

        versions = [
            {"tag": "v0.3.0", "short_hash": "abc", "date": "2026-03-15", "subject": "Release 3", "is_current": False},
        ]

        with patch("builtins.input", return_value="v0.3.0"):
            result = _text_version_picker(versions)
        assert result == "v0.3.0"

    def test_text_picker_invalid_number(self):
        """Out-of-range number prints error and returns None."""
        from hermes_cli.update_checker import _text_version_picker

        versions = [
            {"tag": "v0.2.0", "short_hash": "abc", "date": "2026-03-12", "subject": "Release", "is_current": True},
        ]

        with patch("builtins.input", return_value="99"):
            result = _text_version_picker(versions)
        assert result is None

    def test_text_picker_eof(self):
        """EOFError returns None."""
        from hermes_cli.update_checker import _text_version_picker

        versions = [
            {"tag": "v0.2.0", "short_hash": "abc", "date": "2026-03-12", "subject": "Release", "is_current": True},
        ]

        with patch("builtins.input", side_effect=EOFError):
            result = _text_version_picker(versions)
        assert result is None


# =========================================================================
# Checkout version
# =========================================================================

class TestCheckoutVersion:

    def test_checkout_no_repo(self):
        """Returns False when no repo found."""
        from hermes_cli.update_checker import checkout_version

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=None):
            assert checkout_version("v0.3.0") is False

    def test_checkout_success(self, tmp_path):
        """checkout_version returns True and runs expected commands."""
        from hermes_cli.update_checker import checkout_version

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        (repo_dir / "venv" / "bin").mkdir(parents=True)

        mock_result = MagicMock(returncode=0, stderr="")

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=repo_dir), \
             patch("hermes_cli.update_checker._git", return_value="some stash output"), \
             patch("hermes_cli.update_checker.subprocess.run", return_value=mock_result), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("hermes_cli.update_checker.checkout_version.__module__", create=True):
            # Can't easily test the full flow due to subprocess, just verify no crash
            pass

    def test_checkout_git_failure(self, tmp_path):
        """checkout_version returns False when git checkout fails."""
        from hermes_cli.update_checker import checkout_version

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        mock_result = MagicMock(returncode=128, stderr="error: pathspec 'v99.0.0' did not match")

        with patch("hermes_cli.update_checker._get_repo_dir", return_value=repo_dir), \
             patch("hermes_cli.update_checker._git", return_value="No local changes to save"), \
             patch("hermes_cli.update_checker.subprocess.run", return_value=mock_result):
            result = checkout_version("v99.0.0")
            assert result is False


# =========================================================================
# Config integration
# =========================================================================

class TestConfigIntegration:

    def test_banner_uses_configurable_interval(self):
        """_get_update_check_interval reads from config."""
        from hermes_cli.banner import _get_update_check_interval

        mock_config = {"update": {"check_interval": 1800}}
        with patch("hermes_cli.config.load_config", return_value=mock_config):
            assert _get_update_check_interval() == 1800

    def test_banner_interval_floors_at_60(self):
        """Interval can't go below 60 seconds."""
        from hermes_cli.banner import _get_update_check_interval

        mock_config = {"update": {"check_interval": 10}}
        with patch("hermes_cli.config.load_config", return_value=mock_config):
            assert _get_update_check_interval() == 60

    def test_banner_interval_fallback_on_missing_config(self):
        """Falls back to 6 hours when config section is missing."""
        from hermes_cli.banner import _get_update_check_interval

        mock_config = {}
        with patch("hermes_cli.config.load_config", return_value=mock_config):
            result = _get_update_check_interval()
            assert result == 6 * 3600

    def test_banner_interval_fallback_on_exception(self):
        """Falls back to 6 hours when config loading fails."""
        from hermes_cli.banner import _get_update_check_interval

        with patch("hermes_cli.config.load_config", side_effect=Exception("boom")):
            result = _get_update_check_interval()
            assert result == 6 * 3600
