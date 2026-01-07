# coding:utf-8
"""
    :module: install_desktop.py
    :description: Create desktop file and copy supplied icon or install an app to Linux
    :author: Michel 'Mitch' Pecqueur
    :date: 2026.01
"""

import shutil
import sys
import traceback
from pathlib import Path


def install_desktop(exe_path: Path, icon_file: Path | None = None, app_name: str | None = None) -> Path:
    """
    Create desktop file and copy supplied icon

    :param exe_path: Full path to the executable
    :param app_name: Application Name
    :param icon_file: .png icon file

    :return: Path to created desktop file
    """
    home = Path.home()
    p = Path(exe_path)

    desktop_dir = home / '.local/share/applications'
    icons_dir = home / '.local/share/icons/hicolor'

    desktop_dir.mkdir(parents=True, exist_ok=True)

    # Copy icon
    icon_path = None
    if Path(icon_file).is_file():
        icon_path = icons_dir / f'{p.stem}.png'
        print(icon_path)
        shutil.copy(icon_file, icon_path)

    # Write desktop file

    app_name = app_name or p.stem

    desktop_file = desktop_dir / f'{p.stem}.desktop'

    desktop_str = (f'[Desktop Entry]\n'
                   f'Type=Application\n'
                   f'Comment=Impulse Response Batch Processing Tool\n'
                   f'Name={app_name}\n'
                   f'Exec={p}\n'
                   f'Path={p.parent}\n')
    if icon_path:
        desktop_str += f'Icon={icon_path.stem}\n'
    desktop_str += f'Terminal=false\nCategories=Audio;'

    with open(desktop_file, 'w') as f:
        f.write(desktop_str)

    desktop_file.chmod(0o755)

    print(f"Installed {desktop_file}")

    return desktop_file


def main():
    if len(sys.argv) >= 2:
        try:
            install_desktop(*sys.argv[1:])
        except Exception:
            traceback.print_exc()
            input('Press ENTER to continue...')
    else:
        sys.stderr.write("Wrong number or arguments\n"
                         "Please provide at least an executable\n")
        input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
