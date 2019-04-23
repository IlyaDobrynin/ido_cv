# -*- coding: utf-8 -*-
"""
Main line

"""
import os
import yaml

from . import dirs
from . import allowed_parameters
from . import Pipeline
from . import find_lr
from . import train
from . import validation
from . import prediction

# Get parameters
METRICS = allowed_parameters.METRICS
OPTIMIZERS = allowed_parameters.OPTIMIZERS
CHECKPOINT_METRICS = allowed_parameters.CHECKPOINT_METRICS
TTA = allowed_parameters.TTA


def main_pipe(args):
    """ Main pipeline

    :param args: Initial parameters
    :return:
    """
    if args['device_ids'][0] == -1:
        allocate_on = 'cpu'
    else:
        allocate_on = 'gpu'

    if args['tta_list'] is not None:
        tta_list = [TTA[args['task']][args['mode']][tta_name] for tta_name in args['tta_list']]
    else:
        tta_list = None

    # Get main pipeline class
    pipe_class = Pipeline(
        task=args['task'],
        mode=args['mode'],
        loss_name=args['loss_name'],
        optim_name=args['optimizer'],
        allocate_on=allocate_on,
        tta_list=tta_list,
        img_size_orig=args['in_size_orig'],
        img_size_target=args['in_size_target'],
        random_seed=args['seed'],
        time=args['time']
    )

    # Get initial model, initial parameters (first epoch, first step, best measure)
    # and model_parameters
    model, initial_parameters, model_parameters = pipe_class.get_model(
        model_name=args['model_name'],
        device_ids=args['device_ids'],
        cudnn_bench=args['cudnn_benchmark'],
        path_to_weights=args['path_to_weights'],
        model_parameters=args['model_parameters']
    )
    args['first_epoch'] = initial_parameters['epoch']
    args['first_step'] = initial_parameters['step']
    args['best_measure'] = initial_parameters['best_measure']
    args['model_parameters'] = model_parameters

    # Paths
    path_to_train = os.path.join(args['data_path'], 'train')
    path_to_valid = os.path.join(args['data_path'], 'val')
    path_to_holdout = os.path.join(args['data_path'], 'holdout')
    path_to_holdout_labels = os.path.join(args['data_path'], 'holdout/masks')
    path_to_test = os.path.join(args['data_path'], 'test')

    scores = None
    stages = ['f', 't', 'v']
    if set(args['stages']).intersection(set(stages)):
        # Find learning rate
        if 'f' in args['stages']:
            print('-' * 30, ' FINDING LEARNING RATE ', '-' * 30)
            args['learning_rate'] = find_lr(
                pipeline=pipe_class, model_name=args['model_name'], path_to_dataset=path_to_train,
                batch_size=5, workers=args['workers'], shuffle_dataset=args['shuffle_train'],
                use_augs=args['train_augs'], device_ids=args['device_ids'],
                cudnn_benchmark=args['cudnn_benchmark'], lr_factor=args['lr_factor'],
                path_to_weights=args['path_to_weights']
            )
        
        # Training line
        if 't' in args['stages']:

            task = pipe_class.task
            mode = pipe_class.mode
            time = pipe_class.time

            model_save_dir = dirs.make_dir(
                relative_path=f'models/{task}/{mode}/{args["model_name"]}/{time}',
                top_dir=args['output_path']
            )

            # Save hyperparameters dictionary
            with open(os.path.join(model_save_dir, 'hyperparameters.yml'), 'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)

            print('-' * 30, ' TRAINING ', '-' * 30)
            model = train(
                model=model, pipeline=pipe_class, train_data_path=path_to_train,
                val_data_path=path_to_valid, model_save_dir=model_save_dir,
                val_metrics=args['valid_metrics'], checkpoint_metric=args['checkpoint_metric'],
                batch_size=args['batch_size'], first_step=args['first_step'],
                best_measure=args['best_measure'], first_epoch=args['first_epoch'],
                epochs=args['epochs'], n_best=args['n_best'], scheduler=args['scheduler'],
                workers=args['workers'], shuffle_train=args['shuffle_train'],
                augs=args['train_augs'], patience=args['patience'],
                learning_rate=args['learning_rate']

            )

        if 'v' in args['stages']:
            # Validation (metrics evaluation) line
            print('-' * 30, ' VALIDATION ', '-' * 30)
            scores = validation(
                model=model, pipeline=pipe_class, data_path=path_to_holdout,
                labels_path=path_to_holdout_labels, val_metrics=args['valid_metrics'],
                batch_size=args['batch_size'], workers=args['workers'], save_preds=args['save_val'],
                output_path=args['output_path']
            )

    # Prediction line
    if 'p' in args['stages']:
        print('-' * 30, ' PREDICTION ', '-' * 30)
        threshold = scores[args['checkpoint_metric']]['threshold'] if scores is not None \
            else args['default_threshold']
        print(f'Prediction threshold: {threshold}')
        prediction(model=model, pipeline=pipe_class, data_path=path_to_test,
                   batch_size=args['batch_size'], workers=args['workers'], threshold=threshold,
                   postprocess=args['postproc_test'], output_path=args['output_path'],
                   show_preds=args['show_preds'], save_preds=args['save_test'])

