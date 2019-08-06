# -*- coding: utf-8 -*-
"""
A little module for easy access to root and data folders and
quickly directories creation

"""
import os


if os.name == 'nt':
    DIR_SPLITTER = '\\'
else:
    DIR_SPLITTER = '/'


def make_dir(relative_path, top_dir='/'):
    """ Function to make complex path

    :param relative_path: Path to create on top of top_dir
    :param top_dir: Root path
    :return: top_dir
    """
    dirs = relative_path.replace(DIR_SPLITTER, "/").split('/')
    for d in dirs:
        top_dir = os.path.join(top_dir, d)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
    return top_dir


def split_path(path):
    dirs = path.replace(DIR_SPLITTER, "/").split('/')
    return dirs

