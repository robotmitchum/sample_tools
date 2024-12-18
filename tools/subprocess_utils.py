# coding:utf-8
"""
    :module: subprocess_utils.py
    :description: Functions to suppress shell windows when using subprocess
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.12
"""

import subprocess


def suppress_shell_windows():
    if hasattr(subprocess, 'STARTUPINFO'):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        return si
    return None


class DisableShellWindows:
    """
    Temporary suppress shell windows
    """

    def __enter__(self):
        self.original_popen = subprocess.Popen

        def silent_popen(*args, **kwargs):
            if "startupinfo" not in kwargs:
                kwargs["startupinfo"] = suppress_shell_windows()
            return self.original_popen(*args, **kwargs)

        subprocess.Popen = silent_popen
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        subprocess.Popen = self.original_popen
