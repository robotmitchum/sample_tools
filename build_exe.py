# coding:utf-8
"""
    :module: build_exe.py
    :description: Generic cross-platform function to compile a python app to an executable using pyInstaller
    Takes a json config file for options
    :author: Michel 'Mitch' Pecqueur
    :date: 2026.01

Example:
build_exe('my_script.py', 'build_config.json')
"""

import json
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def build_exe(scriptname: Path | str = '', config_path: Path | str = 'build_config.json') -> Path | None:
    """
    Compile a python app to an executable using pyInstaller
    :param scriptname: Main script of the python app
    :param config_path: Path to a json config file
    :return: Path to created executable
    """
    os_name = platform.system()

    defaults = {'--name': Path(scriptname).stem,
                '--workpath': 'build',
                '--distpath': 'dist'}

    with open(config_path, 'r') as f:
        json_data = f.read()

    data = json.loads(json_data)
    for k, v in defaults.items():
        if k not in data:
            data[k] = v

    cmd_args = []
    for arg, data_value in data.items():

        values = data_value if isinstance(data_value, list) else [data_value]

        for value in values:
            match value:
                case str():
                    if value:
                        match arg:
                            case '--add-data' | '--add-binary':
                                # = separator is required by pyinstaller syntax
                                cmd_args.append(f'{arg}={value}')
                            case _:
                                cmd_args.extend([arg, str(value)])
                case bool():
                    # Only add a bool argument if value is True
                    if value:
                        cmd_args.append(arg)
                case dict():
                    # Add argument depending on the current OS
                    value = value.get(os_name, None)
                    if value:
                        match arg:
                            case '--add-data' | '--add-binary':
                                # = separator is required by pyinstaller syntax
                                cmd_args.append(f'{arg}={value}')
                            case _:
                                cmd_args.extend([arg, str(value)])
                case _:
                    pass

    cmdlist = ['pyinstaller', scriptname, *cmd_args]

    result = subprocess.run(cmdlist)

    if result.returncode == 0:
        # Clean build directory and spec file after process
        shutil.rmtree('build', ignore_errors=True)
        spec_file = Path(scriptname).parent / f'{data['--name']}.spec'
        spec_file.unlink(missing_ok=True)

        result = Path(config_path).parent / f'{data['--distpath']}/{data['--name']}'
        if os_name == 'Windows':
            result = result.with_suffix('.exe')

        print(result)
        return result

    return None


def main():
    if len(sys.argv) == 3:
        try:
            build_exe(sys.argv[1], sys.argv[2])
        except Exception:
            traceback.print_exc()
            input('Press ENTER to continue...')
    else:
        sys.stderr.write("Wrong number or arguments\n"
                         "Please provide a python script and a json config\n")
        input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
