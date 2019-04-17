# -*- coding: utf-8 -*-
"""
    Script for weights download

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import re

import shutil
import tempfile
import hashlib
import zipfile
from tqdm import tqdm
import torch
from requests import get as urlopen


# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def load_seafile_url(url: str, model_dir: str = None, map_location: str = None) -> tuple:
    r""" Function loads .zip archive with weights and (optionally) model meta-information

    If the archive is already present in `model_dir`, it's deserialized and
    returned along with the path to weights.
    The filename part of the URL should follow the naming convention
    ``filename-<sha256>.zip`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the archive. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to
                                 remap storage locations (see torch.load)
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    u = urlopen(url, stream=True)
    headers = u.headers
    if 'Content-Disposition' in headers.keys():
        filename = headers['Content-Disposition'].split('"')[-2]
        hash_prefix = HASH_REGEX.search(filename).group(1)
    else:
        raise RuntimeError(
            f'Wrong url: {url}. It have no "Content-Disposition" in keys.'
        )

    if filename[-4:] != '.zip':
        raise RuntimeError(
            f'File should be .zip archive.'
        )
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        _download_url_to_file(url=url, hash_prefix=hash_prefix, dst=cached_file)

    zip_ref = zipfile.ZipFile(cached_file, 'r')
    zip_ref.extractall(cached_file[:-4])
    zip_ref.close()

    weights_folder = os.path.join(cached_file[:-4], 'weights')
    if os.path.exists(weights_folder):
        weights_name = os.listdir(weights_folder)
        if len(weights_name) != 1:
            raise RuntimeError(
                f'Weights folder should have only one weight. Got {len(weights_name)}.'
            )
        weights = os.path.join(weights_folder, weights_name[0])
    else:
        raise RuntimeError(
            f'File {weights_folder} does not exists.'
        )

    return weights, torch.load(weights, map_location=map_location)


def _download_url_to_file(url: str, hash_prefix: str, dst: str):
    """ Function downloads the file for given url into dst path if the given hash_prefix
    is equal to the file prefix

    :param url: Link to the file
    :param hash_prefix: Prefix of the file
    :param dst: Destination folder to save file
    """
    file_size = None
    u = urlopen(url, stream=True)
    if "Content-Length" in u.headers.keys():
        file_size = int(u.headers['Content-Length'])
    u = u.raw
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                    pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[-len(hash_prefix):] != hash_prefix:
                raise RuntimeError(
                    f'Invalid hash value '
                    f'(expected "{hash_prefix}", got "{digest[:len(hash_prefix)]}")'
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


if __name__ == '__main__':
    import torch
    from hashlib import sha256
    # print(requests_available)
    # path = r'http://213.108.129.195:8000/f/e1543cdcfa/?raw=1'
    # path = r'http://213.108.129.195:8000/f/12f7db0c38/?raw=1'
    path = r'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'
    # weights = torch.load(path)
    # print(type(weights))
    # print(str(sha256(b'se_resnext50').hexdigest()))
    # print(HASH_REGEX)
    # get_hash(path)
    from mts_cv import dirs
    root_dir = r'/home/ido-mts/Work/Projects/cs-scripts'
    weights_dir = dirs.make_dir(relative_path='weights', top_dir=root_dir)
    load_seafile_url(url=path, model_dir=weights_dir)
