"""Cross-platform desktop notifications, no extra pip dependency.

Each OS has a notifier built into a normal install, so we shell out to it
directly instead of depending on a Python notification package -- handy
since brand-new Python versions often lack prebuilt wheels for such
packages (as happened with opencv on this project):

  - Windows: the built-in Windows.UI.Notifications WinRT API, driven via a
    small bundled PowerShell script (assets/toast.ps1).
  - macOS: `osascript` (ships with the OS), via `display notification`.
  - Linux: `notify-send` (part of libnotify, present on most desktop
    environments -- GNOME, KDE, etc). If it's missing, notify() no-ops.

In every case, title/message are passed as separate process arguments
rather than concatenated into a shell/script string, so user-typed names
can't inject commands. The macOS AppleScript source is the one place text
is embedded into a larger string we hand to a subprocess, so it's
explicitly escaped before embedding.
"""

import os
import platform
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOAST_SCRIPT = os.path.join(BASE_DIR, "assets", "toast.ps1")

_CREATE_NO_WINDOW = 0x08000000


def notify(title, message):
    """Fire-and-forget a native desktop notification. Returns False (no-op)
    if the platform's notifier isn't available."""
    system = platform.system()
    try:
        if system == "Windows":
            return _notify_windows(title, message)
        if system == "Darwin":
            return _notify_macos(title, message)
        if system == "Linux":
            return _notify_linux(title, message)
    except OSError:
        return False
    return False


def _notify_windows(title, message):
    if not os.path.exists(TOAST_SCRIPT):
        return False
    subprocess.Popen(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-File",
            TOAST_SCRIPT,
            "-Title",
            title,
            "-Message",
            message,
        ],
        creationflags=_CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def _applescript_string(text):
    """Escape text for safe embedding inside an AppleScript string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _notify_macos(title, message):
    if shutil.which("osascript") is None:
        return False
    script = (
        f"display notification {_applescript_string(message)} "
        f"with title {_applescript_string(title)}"
    )
    subprocess.Popen(
        ["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return True


def _notify_linux(title, message):
    exe = shutil.which("notify-send")
    if exe is None:
        return False
    subprocess.Popen(
        [exe, title, message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return True
