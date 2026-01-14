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
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def build_exe(scriptname: Path | str = '', build_config: Path | str = 'build_config.json',
              create_bundle_dir: bool = True, bundle_config: Path | str = 'bundle_config.json',
              archive_bundle=True, dry_run: bool = False) -> Path | None:
    """
    Compile a python app to an executable using pyInstaller

    :param scriptname: Main script of the python app
    :param build_config: Path to a json config file for pyInstaller

    :param create_bundle_dir: Copy built executable and additional files to a "bundle directory" located in dist
    It will be named the same as the project
    :param bundle_config: List of additional files in .json format to copy along the executable in the bundle directory
    :param archive_bundle: Create an archive from the bundle directory

    :param dry_run: Simulate the process only, for debug

    :return: Path to the created executable
    """
    p = Path(build_config)

    if not p.is_file():
        sys.stderr.write('Missing build config file')
        return None

    os_name = platform.system()
    project_dir = get_project(p)

    if project_dir is None:
        return None

    defaults = {'--name': Path(scriptname).stem,
                '--workpath': 'build',
                '--distpath': 'dist'}

    with open(str(p), 'r') as f:
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
                                if isinstance(value, bool):
                                    cmd_args.append(arg)
                                else:
                                    cmd_args.extend([arg, str(value)])
                case _:
                    pass

    cmdlist = ['pyinstaller', scriptname, *cmd_args]

    print(cmdlist)

    if not dry_run:
        result = subprocess.run(cmdlist)
        return_code = result.returncode
    else:
        return_code = 0

    bundle_archive = None

    if return_code == 0:
        # Clean build directory and spec file after process
        if not dry_run:
            shutil.rmtree('build', ignore_errors=True)
            spec_file = Path(scriptname).parent / f'{data['--name']}.spec'
            spec_file.unlink(missing_ok=True)

        output_path = Path(build_config).parent / f'{data['--distpath']}/{data['--name']}'

        match os_name:
            case 'Darwin':
                exe_path = output_path.with_suffix('.app')
                # Clean raw compiled file outside the .app bundle
                # FIXME: permission error
                # if not dry_run:
                #     output_path.unlink(missing_ok=True)
            case 'Windows':
                exe_path = output_path.with_suffix('.exe')
            case _:
                exe_path = output_path

        if create_bundle_dir:
            # - Move executable to a subdirectory with the program's name

            bundle_dir = exe_path.resolve().parent / project_dir.resolve().name
            # Use a temp name to deal with an executable and the project_dir having the same name
            tmp_dir = exe_path.resolve().parent / 'temp_bundle_dir'

            print(f'\nBundle Directory: {bundle_dir}')
            new_exe_path = bundle_dir / exe_path.name
            tmp_exe_path = tmp_dir / exe_path.name

            if not dry_run:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(exe_path, tmp_exe_path)
            tmp_dir.rename(bundle_dir)

            exe_path = new_exe_path

            # Copy app files
            p = Path(bundle_config)
            if p.is_file():
                with open(str(p), 'r') as f:
                    json_data = f.read()
                data = json.loads(json_data)
                for item in data:
                    p = Path(item)
                    dest = bundle_dir.joinpath(item).relative_to(project_dir.resolve())
                    if p.exists() and not dest.exists():
                        print(f'{dest}')
                        if not dry_run:
                            if p.is_file():
                                shutil.copy(p, dest)
                            if p.is_dir():
                                shutil.copytree(p, dest)

            if archive_bundle:
                bundle_archive = archive_dir(bundle_dir, os_name=None)

        print(f'Executable: {exe_path.resolve().relative_to(project_dir.resolve())}')
        if bundle_archive:
            print(f'Bundle Archive: {bundle_archive.relative_to(project_dir.resolve())}')

        return exe_path

    return None


def archive_dir(dir_name: Path | str, os_name: str | None = None) -> Path | None:
    """
    Compress given directory to an archive with proper naming depending on OS, so it can be released on GitHub
    :param dir_name: Directory to archive
    :param os_name: 'mac', 'linux' or 'win', Auto-detect if None
    :return:
    """
    if not os_name:
        os_str = {'Darwin': 'mac', 'Linux': 'linux', 'Windows': 'win'}[platform.system()]
    else:
        os_str = os_name

    p = Path(dir_name)
    project_dir = get_project(p)
    if project_dir is None:
        return None
    else:
        version = get_version(project_dir / '__init__.py')
        if version is None:
            sys.stderr.write('Invalid project: no version found')
            return None

    if os_str == 'linux':
        fmt = 'gztar'
    else:
        fmt = 'zip'

    base_name = p.parent / f'{project_dir.name}-{version}-{os_str}'

    result = shutil.make_archive(base_name=str(base_name), format=fmt, root_dir=p.parent, base_dir=p.name)
    print(result)

    return Path(result)


def get_project(asset: Path | str) -> Path | None:
    """
    Retrieve project path from a given file
    :param asset:
    :return:
    """
    p = Path(asset)

    project_dir = p
    while project_dir != Path():
        project_dir = project_dir.parent
        if project_dir.joinpath('.gitignore').exists():
            return project_dir

    sys.stderr.write('.gitignore not found in upstream hierarchy')
    return None


def get_version(filepath: Path | str) -> str | None:
    """
    Get value of __version__ variable from a given file
    :param filepath:
    :return:
    """
    fp = Path(filepath)
    if fp.is_file():
        text = fp.read_text()
        value = re.findall('__version__.*\n', text)
        if value:
            version = eval(value[0].split('=')[-1].strip())
            return version

    sys.stderr.write(f'{fp} or __version__ not found')
    return None


def main():
    if len(sys.argv) == 4:
        try:
            build_exe(scriptname=sys.argv[1], build_config=sys.argv[2], bundle_config=sys.argv[3])
        except Exception:
            traceback.print_exc()
            input('Press ENTER to continue...')
    else:
        sys.stderr.write("Wrong number or arguments\n"
                         "Please provide a python script, a build config and a bundle config\n")
        input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
