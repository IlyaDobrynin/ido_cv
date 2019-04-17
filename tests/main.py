import argparse
import datetime

from mts_cv.tests import test_main
from mts_cv.main import main_pipe
from mts_cv.src.utils.get_model_config import ConfigParser


parser = argparse.ArgumentParser(description='Symbol identification pipeline')
parser.add_argument("-t", "--task",
                    type=str,
                    default=str("s"),
                    help='Pipeline stages. Should be one of:\n'
                         '"s" - segmentation\n'
                         '"d" - detection\n'
                         '"c" - classification\n')
parser.add_argument("-m", "--mode",
                    type=str,
                    default=str("b"),
                    help='Mode. Should be:\n'
                         '- for segmentation and classification: b or m\n'
                         '- for detection: a')
parser.add_argument("-mn", "--model-name",
                    type=str,
                    default=str("unet"),
                    help='Model name. Should be one of allowed for current task:\n'
                         'segmentation: unet or deeplabv3\n'
                         'detection: RetinaNet'
                         'classification: basic_model')
parser.add_argument("-l", "--loss-name",
                    type=str,
                    default=str("bce_jaccard"),
                    help='Loss name. Should be one of allowed for current task:\n'
                         'segmentation: bce_dice, bce_jaccard\n'
                         'detection: focal_loss\n'
                         'classification: bce or ce')
parser.add_argument("-s", "--stages",
                    type=list,
                    default=["f", "t", "v"],
                    help='Pipeline stages. Should be a list, contained follow parameters:\n'
                         '"f" - for find learning rate line\n'
                         '"t" - for training line\n'
                         '"v" - for validation line\n'
                         '"p" - for prediction line')
parser.add_argument("-cm", "--chp-metric",
                    type=str,
                    default="jaccard",
                    help='Metrics to monitoring sheduler policy:\n'
                         'segmentation: one of dice, jaccard, loss\n'
                         'detection: loss'
                         'classification: loss, accuracy')
parser.add_argument("-di", "--device-ids",
                    type=list,
                    # default=[0, 1],
                    default=[0],
                    # default=[-1],
                    help='List of the GPU ids to train on (default [0])')
parser.add_argument("-w", "--workers",
                    type=int,
                    default=10,
                    help='Amount of parallel workers (default 1)')
parser.add_argument("-o", "--optimizer",
                    type=str,
                    default='adam',
                    help='Name of the optimizer to train with (default "adam").'
                         'Wariants are: "adam", "sgd"')
parser.add_argument("-bs", "--batch-size",
                    type=int,
                    default=42,
                    help='Size of the mini batch in data loader (default 10)')
parser.add_argument("-lr", "--learning-rate",
                    type=float,
                    default=0.00003,
                    help='Default learning rate (default 0.001). To find optimal learning rate'
                         'include "f" parameter to stages (-s ["f", "t"])')

arguments = parser.parse_args()

tasks = dict(
    s='segmentation',
    d='detection',
    c='classification'
)

modes = dict(
    b='binary',
    m='multi',
    a='all'
)

if arguments.task == 's':
    tta_list = ['nothing', 'h_flip']
elif arguments.task == 'd':
    tta_list = None
else:
    tta_list = None

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

    # task='classification',
    # # mode='binary',
    # mode='multi',
    # model_name='basic_model',
    # # loss_name='bce',
    # # loss_name='nll',
    # loss_name='ce',
    # valid_metrics=['accuracy'],
    # checkpoint_metric='accuracy',
    # tta_list=None,
    # labels_name='labels.csv',
    # # path_to_weights=WEIGHTS_PATH,
    # path_to_weights=None,

    task='segmentation',
    mode='binary',
    # mode='multi',
    model_name='unet',
    # model_name='deeplabv3',
    # loss_name='lovasz',
    loss_name='bce_jaccard',
    valid_metrics=['dice', 'jaccard'],
    checkpoint_metric='jaccard',
    tta_list=['nothing', 'h_flip'],
    labels_name='masks',
    # path_to_weights=WEIGHTS_PATH,
    path_to_weights=None,
    model_parameters=None,

    optimizer=arguments.optimizer,
    # stages=['f', 't', 'v', 'p'],
    stages=['f', 't', 'v'],
    # stages=['t', 'v'],
    # stages=['v'],
    # stages=['v', 'p'],
    # stages=['p'],
    device_ids=arguments.device_ids,
    workers=arguments.workers,
    batch_size=arguments.batch_size,
    cudnn_benchmark=True,
    shuffle_train=False,
    train_augs=True,
    learning_rate=arguments.learning_rate,
    lr_factor=10,
    epochs=200,
    n_best=1,
    scheduler='rop',
    patience=10,
    in_size_orig=(45, 256),
    # in_size_orig=(720, 720),
    in_size_target=256,
    time="{:%Y%m%dT%H%M}".format(datetime.datetime.now()),
    seed=42,

    # Detection options
    cls_thresh=0.5,
    nms_thresh=0.7,
    val_iou_thresholds=0.7,
    show_preds=True,

    # Flags for validation
    validate_images=True,
    validate_dates=False,

    # Postproc
    postproc_test=False,

    # Save outputs parameters
    save_val=False,
    save_test=False,

)

# Directories
config_parser = ConfigParser(cfg_type='dirs', cfg_path='/home/ido-mts/Work/Projects/setup_test/cfg/directories.yml')
DIRECTORIES = config_parser.parameters
DATA_DIR = DIRECTORIES['data_dir']
OUT_DIR = DIRECTORIES['output_dir']
data_path = DATA_DIR
output_path = OUT_DIR

parameters['data_path'] = data_path
parameters['output_path'] = output_path


# Test input parameters
test_main.test_parameters(parameters)
main_pipe(args=parameters)