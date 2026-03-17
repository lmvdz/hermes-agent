"""Scheduled update checker, changelog viewer, and interactive version picker.

Provides:
- ``ScheduledUpdateChecker``: background daemon that periodically checks for
  new commits and caches the result; exposes ``update_available`` / ``commits_behind``.
- ``show_changelog()``: prints a formatted git log between HEAD and origin/main.
- ``interactive_version_picker()``: curses-based UI to browse tags/versions and
  check out a specific one with arrow keys + Enter.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# Helpers
# =========================================================================

def _get_repo_dir() -> Optional[Path]:
    """Locate the hermes-agent git repo."""
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    repo_dir = hermes_home / "hermes-agent"
    if (repo_dir / ".git").exists():
        return repo_dir
    # Fall back to project root for dev installs
    project_root = Path(__file__).parent.parent.resolve()
    if (project_root / ".git").exists():
        return project_root
    return None


def _get_cache_file() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return hermes_home / ".update_check"


def _git(args: list, repo_dir: Path, timeout: int = 10) -> Optional[str]:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(repo_dir),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# =========================================================================
# Scheduled update checker
# =========================================================================

class ScheduledUpdateChecker:
    """Background daemon that checks for updates at a configurable interval.

    The result is cached to ``~/.hermes/.update_check`` and also held in
    memory so the status bar can read it without I/O.

    Usage::

        checker = ScheduledUpdateChecker(interval_seconds=3600)
        checker.start()
        ...
        if checker.update_available:
            print(f"{checker.commits_behind} commits behind")
        ...
        checker.stop()
    """

    def __init__(self, interval_seconds: int = 3600):
        self.interval = max(60, interval_seconds)  # Floor at 1 minute
        self._commits_behind: Optional[int] = None
        self._latest_tag: Optional[str] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._checked_once = threading.Event()

        # Load cached value immediately so status bar has data before first check
        self._load_cache()

    # -- public API --------------------------------------------------------

    @property
    def update_available(self) -> bool:
        with self._lock:
            return (self._commits_behind or 0) > 0

    @property
    def commits_behind(self) -> int:
        with self._lock:
            return self._commits_behind or 0

    @property
    def latest_tag(self) -> Optional[str]:
        with self._lock:
            return self._latest_tag

    def start(self):
        """Start the background checker daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="update-checker")
        self._thread.start()

    def stop(self):
        """Signal the checker to stop."""
        self._stop_event.set()

    def check_now(self) -> Optional[int]:
        """Run an immediate (blocking) check. Returns commits behind or None."""
        return self._do_check()

    def wait_for_first_check(self, timeout: float = 5.0) -> bool:
        """Block until the first check completes. Returns True if completed."""
        return self._checked_once.wait(timeout=timeout)

    # -- internal ----------------------------------------------------------

    def _load_cache(self):
        """Load cached check result from disk."""
        try:
            cache_file = _get_cache_file()
            if cache_file.exists():
                data = json.loads(cache_file.read_text())
                with self._lock:
                    self._commits_behind = data.get("behind")
                    self._latest_tag = data.get("latest_tag")
        except Exception:
            pass

    def _save_cache(self, behind: Optional[int], latest_tag: Optional[str] = None):
        """Persist check result to disk."""
        try:
            cache_file = _get_cache_file()
            cache_file.write_text(json.dumps({
                "ts": time.time(),
                "behind": behind,
                "latest_tag": latest_tag,
            }))
        except Exception:
            pass

    def _do_check(self) -> Optional[int]:
        """Perform the actual git check."""
        repo_dir = _get_repo_dir()
        if repo_dir is None:
            self._checked_once.set()
            return None

        # Fetch latest refs
        _git(["fetch", "origin", "--quiet", "--tags"], repo_dir)

        # Count commits behind origin/main
        behind_str = _git(["rev-list", "--count", "HEAD..origin/main"], repo_dir, timeout=5)
        behind = int(behind_str) if behind_str and behind_str.isdigit() else None

        # Find latest tag
        latest_tag = _git(["describe", "--tags", "--abbrev=0", "origin/main"], repo_dir, timeout=5)

        with self._lock:
            self._commits_behind = behind
            self._latest_tag = latest_tag

        self._save_cache(behind, latest_tag)
        self._checked_once.set()
        return behind

    def _loop(self):
        """Background loop: check immediately, then every interval seconds."""
        try:
            self._do_check()
        except Exception:
            self._checked_once.set()  # Don't block waiters on error

        while not self._stop_event.wait(timeout=self.interval):
            try:
                self._do_check()
            except Exception:
                pass


# =========================================================================
# Singleton accessor
# =========================================================================

_checker_instance: Optional[ScheduledUpdateChecker] = None
_checker_lock = threading.Lock()


def get_update_checker(interval_seconds: int = 3600) -> ScheduledUpdateChecker:
    """Get or create the singleton ScheduledUpdateChecker."""
    global _checker_instance
    with _checker_lock:
        if _checker_instance is None:
            _checker_instance = ScheduledUpdateChecker(interval_seconds=interval_seconds)
        return _checker_instance


def start_update_checker(interval_seconds: int = 3600):
    """Convenience: get the singleton and start it."""
    checker = get_update_checker(interval_seconds)
    checker.start()
    return checker


# =========================================================================
# Changelog viewer
# =========================================================================

def get_changelog_entries(limit: int = 50, skip: int = 0) -> List[dict]:
    """Get git log entries from origin/main.

    Returns a list of dicts with keys: hash, short_hash, date, author, subject, tag.
    Commits are returned in reverse chronological order (newest first).

    Args:
        limit: Max entries to return per call.
        skip: Number of commits to skip (for pagination).
    """
    repo_dir = _get_repo_dir()
    if repo_dir is None:
        return []

    # Get log with tags decorated, sorted by commit date descending
    fmt = "%H|%h|%ai|%an|%s|%D"
    args = [
        "log", "--format=" + fmt,
        f"--max-count={limit}",
        "--date-order",
        "origin/main",
    ]
    if skip > 0:
        args.insert(-1, f"--skip={skip}")
    log_output = _git(args, repo_dir, timeout=15)
    if not log_output:
        return []

    entries = []
    for line in log_output.splitlines():
        parts = line.split("|", 5)
        if len(parts) < 6:
            continue
        full_hash, short_hash, date, author, subject, refs = parts

        # Extract tag from refs (e.g. "HEAD -> main, tag: v0.2.0, origin/main")
        tag = None
        if refs:
            for ref_part in refs.split(","):
                ref_part = ref_part.strip()
                if ref_part.startswith("tag:"):
                    tag = ref_part[4:].strip()
                    break

        entries.append({
            "hash": full_hash,
            "short_hash": short_hash,
            "date": date[:10],  # Just the date part
            "author": author,
            "subject": subject,
            "tag": tag,
        })

    return entries


def get_available_versions(limit: int = 30) -> List[dict]:
    """Get available tagged versions from the repo.

    Returns list of dicts: tag, hash, date, subject, is_current.
    """
    repo_dir = _get_repo_dir()
    if repo_dir is None:
        return []

    # Get current HEAD hash
    current_hash = _git(["rev-parse", "HEAD"], repo_dir) or ""

    # Get tags sorted by version (newest first)
    tag_output = _git(
        ["tag", "--sort=-version:refname", "--format=%(refname:short)|%(objectname:short)|%(creatordate:short)|%(subject)"],
        repo_dir, timeout=10,
    )
    if not tag_output:
        return []

    versions = []
    for line in tag_output.splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        tag, short_hash, date, subject = parts

        # Check if this tag is at current HEAD
        tag_hash = _git(["rev-parse", tag], repo_dir) or ""
        is_current = tag_hash == current_hash

        versions.append({
            "tag": tag,
            "short_hash": short_hash,
            "date": date,
            "subject": subject or "(no message)",
            "is_current": is_current,
        })
        if len(versions) >= limit:
            break

    return versions


def get_commit_detail(short_hash: str) -> Optional[dict]:
    """Get full commit details for a given hash.

    Returns a dict with: hash, author, date, subject, body, stats.
    """
    repo_dir = _get_repo_dir()
    if repo_dir is None:
        return None

    output = _git(
        ["show", "--stat", "--format=%H%n%an <%ae>%n%ai%n%s%n%b%n---END-BODY---", short_hash],
        repo_dir, timeout=15,
    )
    if not output:
        return None

    body_end = output.find("---END-BODY---")
    if body_end == -1:
        return None

    header_part = output[:body_end].strip()
    stats_part = output[body_end + len("---END-BODY---"):].strip()

    lines = header_part.split("\n")
    if len(lines) < 4:
        return None

    return {
        "hash": lines[0],
        "author": lines[1],
        "date": lines[2],
        "subject": lines[3],
        "body": "\n".join(lines[4:]).strip(),
        "stats": stats_part,
        "repo_dir": str(repo_dir),
    }


def show_changelog(limit: int = 30, printer=None):
    """Print a formatted changelog.

    Args:
        limit: Max entries to show.
        printer: Callable to output a line.  When running inside the
                 prompt_toolkit TUI, pass ``_cprint`` so ANSI escapes
                 render correctly through patch_stdout.  Defaults to
                 ``print``.
    """
    # Use raw ANSI codes — hermes_cli.colors.color() checks isatty()
    # which returns False inside the prompt_toolkit TUI.
    C = "\033[36m"; B = "\033[1m"; D = "\033[2m"
    Y = "\033[1;33m"; BLU = "\033[34m"; G = "\033[32m"; R = "\033[0m"

    out = printer or print

    entries = get_changelog_entries(limit=limit)
    if not entries:
        out(f"{D}  No changelog entries found.{R}")
        return

    out("")
    out(f"{C}{B}  Changelog (origin/main){R}")
    out(f"{D}  {'─' * 56}{R}")
    out("")

    for entry in entries:
        tag_label = ""
        if entry["tag"]:
            tag_label = f" {Y}[{entry['tag']}]{R}"

        subject = entry["subject"]
        if len(subject) > 60:
            subject = subject[:57] + "..."

        out(f"  {D}{entry['date']}{R}  {BLU}{entry['short_hash']}{R}  {subject}{tag_label}")

    out("")
    out(f"{D}  Showing {len(entries)} entries{R}")
    out("")


# =========================================================================
# Interactive version picker (curses TUI)
# =========================================================================

def interactive_version_picker() -> Optional[str]:
    """Show an interactive curses UI to browse and select a version/tag.

    Navigate with arrow keys, press Enter to select, Escape/q to cancel.
    Returns the selected tag string, or None if cancelled.
    """
    versions = get_available_versions(limit=50)
    if not versions:
        print("  No tagged versions found in the repository.")
        return None

    try:
        import curses
        result_holder = [None]

        def _draw(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)     # current version
                curses.init_pair(2, curses.COLOR_YELLOW, -1)    # header
                curses.init_pair(3, curses.COLOR_CYAN, -1)      # selected/highlight
                curses.init_pair(4, 8, -1)                      # dim
                curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # cursor line

            cursor = 0
            scroll_offset = 0

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()

                # Header
                try:
                    hattr = curses.A_BOLD
                    if curses.has_colors():
                        hattr |= curses.color_pair(2)
                    stdscr.addnstr(0, 0, " ⚕ Hermes Agent — Version Picker", max_x - 1, hattr)
                    stdscr.addnstr(
                        1, 0,
                        "  ↑↓ navigate   Enter select   Esc/q cancel",
                        max_x - 1, curses.A_DIM,
                    )
                    # Column headers
                    header = f"  {'Version':<20} {'Date':<12} {'Description'}"
                    stdscr.addnstr(2, 0, header[:max_x - 1], max_x - 1, curses.A_BOLD | curses.A_UNDERLINE)
                except curses.error:
                    pass

                # Scrollable version list
                visible_rows = max_y - 4
                if visible_rows <= 0:
                    visible_rows = 1
                if cursor < scroll_offset:
                    scroll_offset = cursor
                elif cursor >= scroll_offset + visible_rows:
                    scroll_offset = cursor - visible_rows + 1

                for draw_i, i in enumerate(
                    range(scroll_offset, min(len(versions), scroll_offset + visible_rows))
                ):
                    y = draw_i + 3
                    if y >= max_y - 1:
                        break

                    v = versions[i]
                    tag_str = v["tag"]
                    if v["is_current"]:
                        tag_str += " (current)"

                    subject = v["subject"]
                    # Truncate subject to fit
                    max_subject_len = max(10, max_x - 36)
                    if len(subject) > max_subject_len:
                        subject = subject[:max_subject_len - 3] + "..."

                    line = f"  {tag_str:<20} {v['date']:<12} {subject}"
                    line = line[:max_x - 1]

                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(5)
                        # Fill the line background
                        try:
                            stdscr.addnstr(y, 0, " " * (max_x - 1), max_x - 1, attr)
                        except curses.error:
                            pass
                    elif v["is_current"] and curses.has_colors():
                        attr |= curses.color_pair(1)

                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass

                # Footer
                try:
                    footer_y = max_y - 1
                    footer = f" {cursor + 1}/{len(versions)} versions"
                    if curses.has_colors():
                        stdscr.addnstr(footer_y, 0, footer, max_x - 1, curses.color_pair(4))
                    else:
                        stdscr.addnstr(footer_y, 0, footer, max_x - 1, curses.A_DIM)
                except curses.error:
                    pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ord('k')):
                    cursor = max(0, cursor - 1)
                elif key in (curses.KEY_DOWN, ord('j')):
                    cursor = min(len(versions) - 1, cursor + 1)
                elif key == curses.KEY_PPAGE:  # Page Up
                    cursor = max(0, cursor - visible_rows)
                elif key == curses.KEY_NPAGE:  # Page Down
                    cursor = min(len(versions) - 1, cursor + visible_rows)
                elif key == curses.KEY_HOME:
                    cursor = 0
                elif key == curses.KEY_END:
                    cursor = len(versions) - 1
                elif key in (curses.KEY_ENTER, 10, 13):
                    result_holder[0] = versions[cursor]["tag"]
                    return
                elif key in (27, ord('q')):  # Escape or q
                    return

        curses.wrapper(_draw)
        return result_holder[0]

    except ImportError:
        # No curses — fall back to numbered list
        return _text_version_picker(versions)
    except Exception as e:
        logger.debug("Curses version picker failed: %s", e)
        return _text_version_picker(versions)


def _text_version_picker(versions: List[dict], printer=None) -> Optional[str]:
    """Fallback text-based version picker when curses isn't available.

    Args:
        versions: List of version dicts from ``get_available_versions``.
        printer: Callable to output a line.  Pass ``_cprint`` when running
                 inside the prompt_toolkit TUI.  Defaults to ``print``.
    """
    from hermes_cli.colors import Colors, color

    out = printer or print

    out("")
    out(color("  Available versions:", Colors.CYAN, Colors.BOLD))
    out("")

    for i, v in enumerate(versions):
        current_marker = color(" (current)", Colors.GREEN) if v["is_current"] else ""
        idx = color(f"  [{i + 1:>2}]", Colors.YELLOW)
        tag_label = color(v["tag"], Colors.BOLD)
        date_label = color(v["date"], Colors.DIM)
        subject = v["subject"]
        if len(subject) > 50:
            subject = subject[:47] + "..."
        out(f"{idx}  {tag_label:<20} {date_label}  {subject}{current_marker}")

    out("")
    try:
        choice = input(color("  Enter version number (or press Enter to cancel): ", Colors.CYAN)).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice:
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(versions):
            return versions[idx]["tag"]
    except ValueError:
        # Maybe they typed a tag name directly
        for v in versions:
            if v["tag"] == choice:
                return v["tag"]

    out(color("  Invalid selection.", Colors.RED))
    return None


# =========================================================================
# Update-to-version action
# =========================================================================

def checkout_version(tag: str, printer=None) -> bool:
    """Check out a specific tag/version and reinstall dependencies.

    Args:
        tag: Git tag or ref to check out.
        printer: Callable to output a line.  Pass ``_cprint`` when inside
                 the prompt_toolkit TUI.  Defaults to ``print``.

    Returns True on success, False on failure.
    """
    import shutil

    C = "\033[36m"; G = "\033[32m"; Y = "\033[33m"
    RED = "\033[31m"; B = "\033[1m"; R = "\033[0m"
    out = printer or print

    repo_dir = _get_repo_dir()
    if repo_dir is None:
        out(f"{RED}  ✗ Cannot find hermes-agent git repository.{R}")
        return False

    out(f"{C}  → Checking out {tag}...{R}")

    # Stash any local changes
    stash_result = _git(["stash", "--include-untracked"], repo_dir)
    if stash_result and stash_result != "No local changes to save":
        out(f"{G}  ✓ Stashed local changes{R}")

    # Checkout the tag
    result = subprocess.run(
        ["git", "checkout", tag],
        cwd=str(repo_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        out(f"{RED}  ✗ Failed to checkout {tag}: {result.stderr.strip()}{R}")
        # Restore stash
        if stash_result and stash_result != "No local changes to save":
            _git(["stash", "pop"], repo_dir)
        return False
     
    out(f"{G}  ✓ Checked out {tag}{R}")
    
    # Reinstall Python dependencies
    out(f"{C}  → Updating Python dependencies...{R}")
    uv_bin = shutil.which("uv")
    try:
        if uv_bin:
            subprocess.run(
                [uv_bin, "pip", "install", "-e", ".", "--quiet"],
                cwd=str(repo_dir), check=True,
                env={**os.environ, "VIRTUAL_ENV": str(repo_dir / "venv")},
            )
        else:
            venv_pip = repo_dir / "venv" / "bin" / "pip"
            if venv_pip.exists():
                subprocess.run([str(venv_pip), "install", "-e", ".", "--quiet"],
                               cwd=str(repo_dir), check=True)
            else:
                subprocess.run(["pip", "install", "-e", ".", "--quiet"],
                               cwd=str(repo_dir), check=True)
        out(f"{G}  ✓ Dependencies updated{R}")
    except subprocess.CalledProcessError as e:
        out(f"{Y}  ⚠ Dependency install failed: {e}{R}")

    # Sync skills
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
        out(f"{G}  ✓ Skills synced{R}")
    except Exception:
        pass

    out("")
    out(f"{G}{B}  ✓ Updated to {tag}! Restart hermes to use the new version.{R}")
    out("")
    return True
