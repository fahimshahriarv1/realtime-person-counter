"""Windows desktop toast notifications, no extra pip dependency.

Uses the built-in Windows.UI.Notifications WinRT API via a small bundled
PowerShell script (assets/toast.ps1). This sidesteps needing a Python
notification package -- handy since brand-new Python versions often lack
prebuilt wheels for such packages (as happened with opencv on this
project). Title/message are passed as separate process arguments, never
concatenated into a PowerShell command string, so there's no injection
surface from user-typed names.
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOAST_SCRIPT = os.path.join(BASE_DIR, "assets", "toast.ps1")

_CREATE_NO_WINDOW = 0x08000000


def notify(title, message):
    """Fire-and-forget a Windows toast notification. No-op (returns False)
    on non-Windows platforms or if PowerShell/the script isn't available."""
    if os.name != "nt" or not os.path.exists(TOAST_SCRIPT):
        return False
    try:
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
    except OSError:
        return False
