# -*- coding: utf-8 -*-
"""
    Testing function for ido_cv.main

"""

import datetime

from ido_cv.tests import test_main
from ido_cv import main_pipe
from ido_cv.src.utils.get_model_config import ConfigParser


parameters = dict(
    # task=tasks[arguments.task],
    # model_name=arguments.model_name,
    # loss_name=arguments.loss_name,
    # stages=arguments.stages,
    # valid_metrics=val_metrics,
    # checkpoint_metric=arguments.chp_metric,
    # tta_list=tta_list,
    # path_to_weights=path_to_weights,

    # task='detection',
    # mode='all',
    # model_name='RetinaNet',
    # loss_name='focal_loss',
    # valid_metrics=[],
    # checkpoint_metric='loss',
    # tta_list=None,
    # # path_to_weights=WEIGHTS_PATH,
    # path_to_weights=None,

    task='classification',
    # mode='binary',
    mode='multi',
    model_name='basic_model',
    # loss_name='bce',
    # loss_name='nll',
    loss_name='ce',
    valid_metrics=['accuracy'],
    checkpoint_metric='accuracy',
    tta_list=None,
    in_size_orig=(256, 256),
    in_size_target=256,

    # task='segmentation',
    # # mode='binary',
    # mode='multi',
    # model_name='unet',
    # # model_name='fpn',
    # # model_name='deeplabv3',
    # # loss_name='lovasz',
    # loss_name='bce_jaccard',
    # valid_metrics=['dice', 'jaccard'],
    # checkpoint_metric='jaccard',
    # tta_list=['nothing', 'h_flip'],
    # in_size_orig=(45, 256),
    # in_size_target=256,

    device_ids=[0, 1],
    # device_ids=[0],
    workers=10,
    batch_size=64,
    learning_rate=0.0001,
    epochs=200,


    # STAGES
    # stages=['f', 't', 'v', 'p'],
    # stages=['f', 't', 'v'],
    stages=['t', 'v'],
    # stages=['v'],
    # stages=['v'],
    # stages=['v', 'p'],
    # stages=['p'],

    # COMMON PARAMETERS
    optimizer='adam',
    cudnn_benchmark=True,
    shuffle_train=True,
    train_augs=True,
    lr_factor=10,
    n_best=1,
    scheduler='rop',
    patience=10,
    time="{:%Y%m%dT%H%M}".format(datetime.datetime.now()),
    seed=42,

    # DETECTION OPTIONS
    cls_thresh=0.5,
    nms_thresh=0.7,
    val_iou_thresholds=0.7,
    show_preds=True,

    # Flags for validation
    validate_images=True,
    validate_dates=False,

    # Postproc
    postproc_test=True,

    # Save outputs parameters
    default_threshold=0.5,
    save_val=False,
    save_test=False,

)

# Directories
dirs_parser = ConfigParser(
    cfg_type='dirs',
    cfg_path='/home/ilyado/Programming/pet_projects/ido_cv/tests/cfg/directories.yml'
)
DIRECTORIES = dirs_parser.parameters
DATA_DIR = DIRECTORIES['data_dir']
OUT_DIR = DIRECTORIES['output_dir']
if 'weights_path' in DIRECTORIES.keys():
    WEIGHTS_PATH = DIRECTORIES['weights_path']
else:
    WEIGHTS_PATH = None
parameters['data_path'] = DATA_DIR
parameters['output_path'] = OUT_DIR
parameters['path_to_weights'] = WEIGHTS_PATH

model_parameters_parser = ConfigParser(
    cfg_type='model',
    cfg_path=f'/home/ilyado/Programming/pet_projects/ido_cv/tests/cfg/'
    f'{parameters["task"]}/{parameters["mode"]}/{parameters["model_name"]}.yml'
)
parameters['model_parameters'] = model_parameters_parser.parameters

# Test input parameters
test_main.test_parameters(parameters)
main_pipe(args=parameters)
