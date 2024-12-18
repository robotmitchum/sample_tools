# coding:utf-8
"""
    :module: file_utils.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""

import tempfile
from pathlib import Path


def resolve_overwriting(input_path, mode='dir', dir_name='backup_', test_run=False):
    """
    Move/rename given input file path if it already exists
    :param str or Path input_path:
    :param str mode: 'file' or 'dir'
    :param str dir_name: Directory name when using 'dir' mode
    :param bool test_run: Execute move/rename operation otherwise only return new name
    :return:
    :rtype: Path
    """
    p = Path(input_path)
    parent, stem, ext = p.parent, p.stem, p.suffix

    i = 0
    new_path = p
    while new_path.is_file():
        i += 1
        if mode == 'file':
            new_path = parent.joinpath(f'{stem}_{i:03d}{ext}')
        else:
            new_path = parent.joinpath(f'{dir_name}{i:03d}/{stem}{ext}')

    if not test_run and Path(input_path) != Path(new_path):
        new_path.parent.mkdir(exist_ok=True)
        Path(input_path).rename(new_path)

    return new_path


def recursive_search(root_dir, input_ext=('wav', 'flac'), relpath=None, exclude=('backup_', 'ignore')):
    """
    Retrieve all files with given extension from a root directory
    :param root_dir: Path to a given directory
    :param list or tuple input_ext: File extensions to look for
    :param str or None relpath: Result will be relative to given directory (optional)
    :param list or tuple or None exclude: 
    :return: List of path
    :rtype: list
    """
    files = []
    for ext in input_ext:
        files.extend(Path(root_dir).rglob(f'*.{ext}'))

    if exclude:
        files = [f for f in files if not any(word in str(f.relative_to(root_dir)) for word in exclude)]

    if relpath:
        files = [f.relative_to(relpath) for f in files]

    return files


def move_to_subdir(files, sub_dir=None, test_run=False):
    """
    Move given files to a sub_dir (typically temp dir to avoid overwriting)
    :param list files:
    :param str or None sub_dir: provide explicit name (might exist) or nothing for a safe temp dir
    :param bool test_run: Only return simulated path(s)
    :return:
    :rtype: list
    """
    result = []

    parent_dirs = {Path(f).parent for f in files}

    temp_dir = None
    if not sub_dir:
        temp_dir = {k: tempfile.mkdtemp(dir=k) for k in parent_dirs}

    for f in files:
        p = Path(f)
        if temp_dir is None:
            new_path = Path.joinpath(p.parent, f'{sub_dir}/{p.name}')
        else:
            new_path = Path(temp_dir[p.parent]).joinpath(p.name)
        if not test_run:
            if temp_dir is None:
                Path(new_path.parent).mkdir(exist_ok=True)
            Path(f).rename(new_path)
        result.append(new_path)

    if test_run:
        for d in temp_dir.values():
            Path(d).rmdir()

    return result
