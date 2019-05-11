# -*- coding: utf-8 -*-
"""
Module make tests for main.py

"""
from ..src import allowed_parameters

MODELS = allowed_parameters.MODELS
LOSSES = allowed_parameters.LOSSES
METRIC_NAMES = allowed_parameters.METRIC_NAMES
OPTIMIZERS = allowed_parameters.OPTIMIZERS
CHECKPOINT_METRICS = allowed_parameters.CHECKPOINT_METRICS
TTA = allowed_parameters.TTA


def test_parameters(parameters):
    for m_name in parameters['valid_metrics']:
        assert m_name in METRIC_NAMES[parameters['task']][parameters['mode']], f"Wrong metric: {m_name}. " \
            f"Should be one of {METRIC_NAMES[parameters['task']][parameters['mode']]}"
    
    if parameters['tta_list'] is not None:
        for tta in parameters['tta_list']:
            assert tta in TTA[parameters['task']][parameters['mode']].keys(), f"Wrong tta: {tta}. " \
                f"Should be one of {TTA[parameters['task']][parameters['mode']].keys()}"
    
    assert parameters['model_name'] in MODELS[parameters['task']][parameters['mode']].keys(), f"Wrong model_name: " \
        f"{parameters['model_name']}. Should be one of {MODELS[parameters['task']][parameters['mode']].keys()}"
    
    assert parameters['loss_name'] in LOSSES[parameters['task']][parameters['mode']].keys(), f"Wrong loss name: " \
        f"{parameters['loss_name']}. Should be one of {LOSSES[parameters['task']][parameters['mode']].keys()}"
    
    assert parameters['optimizer'] in OPTIMIZERS, f"Wrong optimizer: {parameters['optimizer']}." \
        f"Should be one of {OPTIMIZERS}"
    
    assert parameters['checkpoint_metric'] in CHECKPOINT_METRICS[parameters['task']][parameters['mode']], f"Wrong " \
        f"checkpoint_metric: {parameters['checkpoint_metric']}. Should be one of " \
        f"{CHECKPOINT_METRICS[parameters['task']][parameters['mode']]}"
