# -*- coding: utf-8 -*-
"""
Module make tests for main.py

"""
from ..src import allowed_parameters

TASKS_MODES = allowed_parameters.TASKS_MODES
LOSS_NAMES = allowed_parameters.LOSS_PARAMETERS
METRIC_NAMES = allowed_parameters.METRIC_NAMES
OPTIMIZERS = allowed_parameters.OPTIMIZERS
CHP_METRICS = allowed_parameters.CHECKPOINT_METRICS
TTA = allowed_parameters.TTA


def test_parameters(parameters):
    for m_name in parameters['valid_metrics']:
        assert m_name in METRIC_NAMES[parameters['task']][parameters['mode']], \
            f"Wrong metric: {m_name}. " \
            f"Should be one of {METRIC_NAMES[parameters['task']][parameters['mode']]}"
    
    if parameters['tta_list'] is not None:
        for tta in parameters['tta_list']:
            assert tta in TTA[parameters['task']][parameters['mode']].keys(), \
                f"Wrong tta: {tta}. " \
                f"Should be one of {TTA[parameters['task']][parameters['mode']].keys()}"
    
    assert parameters['model_name'] in TASKS_MODES[parameters['task']][parameters['mode']],\
        f"Wrong model_name: {parameters['model_name']}. " \
        f"Should be one of {TASKS_MODES[parameters['task']][parameters['mode']]}"
    
    assert parameters['loss_name'] in LOSS_NAMES[parameters['task']][parameters['mode']].keys(),\
        f"Wrong loss name: {parameters['loss_name']}." \
        f" Should be one of {LOSS_NAMES[parameters['task']][parameters['mode']].keys()}"
    
    assert parameters['optimizer'] in OPTIMIZERS,\
        f"Wrong optimizer: {parameters['optimizer']}." \
        f"Should be one of {OPTIMIZERS}"
    
    assert parameters['checkpoint_metric'] in CHP_METRICS[parameters['task']][parameters['mode']], \
        f"Wrong checkpoint_metric: {parameters['checkpoint_metric']}. " \
        f"Should be one of {CHP_METRICS[parameters['task']][parameters['mode']]}"
