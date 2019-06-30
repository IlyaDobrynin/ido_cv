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

    # Get main pipeline class
    pipe_class = Pipeline(
        task=args['task'],
        mode=args['mode'],
        allocate_on=allocate_on,
        random_seed=args['seed'],
        time=args['time'],
        label_colors=args['label_colors'],
        alphabet_characters=args['alphabet_characters'],
    )

    task = pipe_class.task
    mode = pipe_class.mode
    time = pipe_class.time

    if 't' in args['stages']:
        model_save_dir = dirs.make_dir(
            relative_path=f'models/{task}/{mode}/{args["model_name"]}/{time}',
            top_dir=args['output_path']
        )
    else:
        model_save_dir = None

    # Get initial model, initial parameters (first epoch, first step, best measure)
    # and model_parameters
    model, initial_parameters, model_parameters = pipe_class.get_model(
        model_name=args['model_name'],
        save_path=model_save_dir,
        device_ids=args['device_ids'],
        cudnn_bench=args['cudnn_benchmark'],
        path_to_weights=args['path_to_weights'],
        model_parameters=args['model_parameters'],
        verbose=True,
        show_model=args['show_model_info']
    )
    args['first_epoch'] = initial_parameters['epoch']
    args['first_step'] = initial_parameters['step']
    args['best_measure'] = 0
    args['model_parameters'] = model_parameters

    # Paths
    path_to_train = os.path.join(args['data_path'], 'train')
    path_to_valid = os.path.join(args['data_path'], 'val')
    path_to_holdout = os.path.join(args['data_path'], 'holdout')
    path_to_test = os.path.join(args['data_path'], 'test')

    # Get dataloaders
    common_dataloader_parameters = dict(
        workers=args['workers'],
        common_augs=args['common_augs']
    )
    find_lr_dataloader = pipe_class.get_dataloaders(
        path_to_dataset=path_to_train,
        batch_size=5,
        is_train=True,
        shuffle=args['shuffle_train'],
        train_time_augs=args['train_time_augs'],
        **common_dataloader_parameters
    )
    train_loader = pipe_class.get_dataloaders(
        path_to_dataset=path_to_train,
        batch_size=args['batch_size'],
        is_train=True,
        shuffle=args['shuffle_train'],
        train_time_augs=args['train_time_augs'],
        **common_dataloader_parameters
    )
    val_loader = pipe_class.get_dataloaders(
        path_to_dataset=path_to_valid,
        batch_size=args['batch_size'],
        is_train=True,
        shuffle=False,
        **common_dataloader_parameters
    )
    holdout_loader = pipe_class.get_dataloaders(
        path_to_dataset=path_to_holdout,
        batch_size=args['batch_size'],
        is_train=True,
        shuffle=False,
        **common_dataloader_parameters
    )
    test_loader = pipe_class.get_dataloaders(
        path_to_dataset=path_to_test,
        batch_size=args['batch_size'],
        is_train=False,
        shuffle=False,
        **common_dataloader_parameters
    )
    dataloaders = dict(
        find_lr_dataloader=find_lr_dataloader,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        holdout_dataloader=holdout_loader,
        test_dataloader=test_loader,
    )

    scores = None
    stages = ['f', 't', 'v']
    if set(args['stages']).intersection(set(stages)):
        # Find learning rate
        if 'f' in args['stages']:
            print('-' * 30, ' FINDING LEARNING RATE ', '-' * 30)
            args['learning_rate'] = find_lr(
                pipeline=pipe_class, model_name=args['model_name'],
                model_parameters=args['model_parameters'], dataloaders=dataloaders,
                loss_name=args['loss_name'], optim_name=args['optimizer'],
                device_ids=args['device_ids'], cudnn_benchmark=args['cudnn_benchmark'],
                lr_factor=args['lr_factor'], path_to_weights=args['path_to_weights']
            )

        # Training line
        if 't' in args['stages']:
            # Save hyperparameters dictionary
            with open(os.path.join(model_save_dir, 'hyperparameters.yml'), 'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)

            print('-' * 30, ' TRAINING ', '-' * 30)
            model = train(
                model=model, pipeline=pipe_class, dataloaders=dataloaders,
                loss_name=args['loss_name'], optim_name=args['optimizer'],
                model_save_dir=model_save_dir, val_metrics=args['valid_metrics'],
                checkpoint_metric=args['checkpoint_metric'], first_step=args['first_step'],
                best_measure=args['best_measure'], first_epoch=args['first_epoch'],
                epochs=args['epochs'], n_best=args['n_best'], scheduler=args['scheduler'],
                patience=args['patience'], learning_rate=args['learning_rate'])

        # Validation (metrics evaluation) line
        if 'v' in args['stages']:
            # Validation (metrics evaluation) line
            print('-' * 30, ' VALIDATION ', '-' * 30)
            scores = validation(
                model=model, pipeline=pipe_class, dataloaders=dataloaders,
                val_metrics=args['valid_metrics'], tta_list=args['tta_list'],
                save_preds=args['save_val'], output_path=args['output_path']
            )
            print(scores)
            if 't' in args['stages']:
                # Save thresholds dict
                with open(os.path.join(model_save_dir, 'thresholds.yml'), 'w') as outfile:
                    yaml.dump(scores, outfile, default_flow_style=False)

    # Prediction line
    if 'p' in args['stages']:
        print('-' * 30, ' PREDICTION ', '-' * 30)
        threshold = None
        if args['task'] == 'segmentation':
            if args['mode'] == 'binary':
                threshold = scores[args['checkpoint_metric']]['threshold'] if scores is not None \
                    else args['default_threshold']
            elif args['mode'] == 'multi':
                threshold = [scores[cls][args['checkpoint_metric']]['threshold']
                             for cls in scores.keys()] if scores is not None \
                    else args['default_threshold']

        print(f'Prediction threshold: {threshold}')
        test_preds = prediction(
            model=model, pipeline=pipe_class, data_path=path_to_test, dataloaders=dataloaders,
            tta_list=args['tta_list'], threshold=threshold, postprocess=args['postproc_test'],
            with_labels=args['predict_with_labels'], output_path=args['output_path'],
            show_preds=args['show_preds'], save_preds=args['save_test']
        )
        return test_preds


