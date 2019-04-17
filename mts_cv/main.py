# -*- coding: utf-8 -*-
"""
Main line

"""
import os
import gc
import yaml
import numpy as np

# from . import dirs
from mts_cv import dirs
from .src import allowed_parameters
from .src.pipeline_class import Pipeline

# Get parameters
METRICS = allowed_parameters.METRICS
OPTIMIZERS = allowed_parameters.OPTIMIZERS
CHECKPOINT_METRICS = allowed_parameters.CHECKPOINT_METRICS
TTA = allowed_parameters.TTA


def find_lr(pipeline, args):
    print('-' * 30, ' FINDING LEARNING RATE ', '-' * 30)
    # Get dataloader for learning rate finding process
    dataloader = pipeline.get_dataloaders(
        path_to_dataset=os.path.join(args['data_path'], 'train'),
        path_to_labels=os.path.join(args['data_path'], f"train/{args['labels_name']}"),
        batch_size=5,
        is_train=True,
        workers=args['workers'],
        shuffle=args['shuffle_train'],
        augs=args['train_augs']
    )
    # Get model
    find_lr_model, _, _ = pipeline.get_model(
        model_name=args['model_name'],
        device_ids=args['device_ids'],
        cudnn_bench=args['cudnn_benchmark'],
        path_to_weights=args['path_to_weights']
    )
    # Find learning rate process
    optimum_lr = pipeline.find_lr(
        model=find_lr_model,
        dataloader=dataloader,
        lr_reduce_factor=args['lr_factor'],
        verbose=1,
        show_graph=True
    )
    del dataloader
    gc.collect()
    return optimum_lr


def train(pipeline, model, args):
    print('-' * 30, ' TRAINING ', '-' * 30)
    train_loader = pipeline.get_dataloaders(
        path_to_dataset=os.path.join(args['data_path'], 'train'),
        path_to_labels=os.path.join(args['data_path'], f"train/{args['labels_name']}"),
        batch_size=args['batch_size'],
        is_train=True,
        workers=args['workers'],
        shuffle=args['shuffle_train'],
        augs=args['train_augs']
    )
    val_loader = pipeline.get_dataloaders(
        path_to_dataset=os.path.join(args['data_path'], 'val'),
        path_to_labels=os.path.join(args['data_path'], f"val/{args['labels_name']}"),
        batch_size=args['batch_size'],
        is_train=True,
        workers=args['workers'],
        shuffle=False,
        augs=False
    )
    model_save_dir = dirs.make_dir(
        relative_path=f'models/{args["task"]}/{args["mode"]}/{args["model_name"]}/{args["time"]}',
        top_dir=args['output_path']
    )

    # Save hyperparameters dictionary
    with open(os.path.join(model_save_dir, 'hyperparameters.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
    
    model = pipeline.train(
        model=model,
        lr=args['learning_rate'],
        train_loader=train_loader,
        val_loader=val_loader,
        metric_names=args['valid_metrics'],
        best_measure=args['best_measure'],
        first_step=args['first_step'],
        first_epoch=args['first_epoch'],
        chp_metric=args['checkpoint_metric'],
        n_epochs=args['epochs'],
        n_best=args['n_best'],
        scheduler=args['scheduler'],
        patience=args['patience'],
        save_dir=model_save_dir
    )
    print('\nTraining process done. Weights are here: {}/weights\n'.format(model_save_dir))
    del train_loader, val_loader
    gc.collect()
    return model


def validation(pipeline, model, args):
    print('-' * 30, ' VALIDATION ', '-' * 30)
    labels_path = os.path.join(args['data_path'], f'holdout')
    holdout_loader = pipeline.get_dataloaders(
        path_to_dataset=os.path.join(args['data_path'], 'holdout'),
        path_to_labels=labels_path,
        batch_size=args['batch_size'],
        is_train=False,
        workers=args['workers'],
        shuffle=False,
        augs=False
    )
    pred_df = pipeline.predict(
        model=model,
        dataloader=holdout_loader,
        cls_thresh=args['cls_thresh'],
        nms_thresh=args['nms_thresh'],
        save=args['save_val'],
        save_dir=args['output_path']
    )
    del holdout_loader
    gc.collect()

    true_df = pipeline.get_true_labels(labels_path=labels_path)

    print(pred_df, true_df)
    if args['validate_images']:
        # Get score for detection
        if args['task'] == 'detection':
            iou_thresholds = args['val_iou_thresholds']
            args['scores'] = pipeline.evaluate_metrics(
                true_df=true_df,
                pred_df=pred_df,
                iou_thresholds=iou_thresholds
            )
            print('Average Precisions for all classes: {}'.format(args['scores']))
            print('Mean Average Precision (mAP) for all classes: {:.4f}'.format(np.mean(args['scores'])))
        
        # Get score for segmentation
        elif args['task'] =='segmentation':
            args['valid_metrics_dict'] = pipeline.evaluate_metrics(
                true_df=true_df,
                pred_df=pred_df,
                metric_names=args['valid_metrics']
            )
            for k, v in args['valid_metrics_dict'].items():
                print(f'{k} metric: ')
                for m_k, m_v in v.items():
                    print(f'- best {m_k}: {m_v:.5f}')

        # Get score for classification
        elif args['task'] == 'classification':
            args['valid_metrics_dict'] = pipeline.evaluate_metrics(
                true_df=true_df,
                pred_df=pred_df,
                metric_names=args['valid_metrics']
            )
            for k, v in args['valid_metrics_dict'].items():
                print(f'{k} metric: {v}')
    

def prediction(pipeline, model, args):
    print('-' * 30, ' PREDICTION ', '-' * 30)
    test_loader = pipeline.get_dataloaders(
        path_to_dataset=os.path.join(args['data_path'], 'test'),
        path_to_labels=os.path.join(args['data_path'], f"test/{args['labels_name']}"),
        batch_size=args['batch_size'],
        is_train=False,
        workers=args['workers'],
        shuffle=False,
        augs=False
    )
    test_pred_df = pipeline.predict(
        model=model,
        dataloader=test_loader,
        cls_thresh=args['cls_thresh'],
        nms_thresh=args['nms_thresh']
    )
    
    del test_loader
    gc.collect()
    
    # For segmentation
    if args['task'] in ['segmentation_b', 'segmentation_m']:
        threshold = args['valid_metrics_dict'][args['checkpoint_metric']]['threshold']\
            if 'valid_metrics_dict' in args.keys() else 0.99
        print(f'Prediction threshold: {threshold}')

        if args['postproc_test']:
            test_pred_df = pipeline.postprocessing(
                pred_df=test_pred_df,
                threshold=threshold,
                save=args['save_test'],
                save_dir=args['output_path']
            )

        if args['show_preds']:
            pipeline.visualize_preds(preds_df=test_pred_df,
                                     images_path=os.path.join(args['data_path'], 'test/images'))
    if args['task'] == 'classification':
        print(f'Classification preds: {test_pred_df}')
        if args['save_test']:
            out_path = dirs.make_dir(f"preds/{args['task']}/{args['time']}", top_dir=args['output_path'])
            test_pred_df.to_csv(path_or_buf=os.path.join(out_path, f"preds.csv"), index=False)
    

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
    
    # Get initial model and initial parameters (first epoch, first step, best measure)
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
    
    stages = ['f', 't', 'v']
    if set(args['stages']).intersection(set(stages)):
        
        # Find learning rate
        if 'f' in args['stages']:
            args['learning_rate'] = find_lr(pipeline=pipe_class, args=args)
        
        # Training line
        if 't' in args['stages']:
            model = train(pipeline=pipe_class, model=model, args=args)
            
        # Validation (metrics evaluation) line
        if 'v' in args['stages']:
            validation(pipeline=pipe_class, model=model, args=args)

    # Prediction line
    if 'p' in args['stages']:
        prediction(pipeline=pipe_class, model=model, args=args)
