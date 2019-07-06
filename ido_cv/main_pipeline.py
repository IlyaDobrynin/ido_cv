import os
import gc
import yaml
from typing import Dict, Tuple, Union, List
from torch.nn import Module

from ido_cv.src.pipeline_class import Pipeline



class MainPipeline:
    def __init__(self, parameters: Dict):

        self._hyperparameters_dict = dict(
            # Common parameters
            stages=parameters['stages'],
            task=parameters['task'],
            mode=parameters['mode'],
            time=parameters['time'],
            seed=parameters['seed'],

            # Paths
            train_path=parameters['train_path'],
            valid_path=parameters['valid_path'],
            holdout_path=parameters['holdout_path'],
            test_path=parameters['test_path'],
            output_path=parameters['output_path'],
            path_to_weights=parameters['path_to_weights'],

            # Model parameters
            model_name=parameters['model_name'],
            device_ids=parameters['device_ids'],
            cudnn_benchmark=parameters['cudnn_benchmark'],
            model_parameters=parameters['model_parameters'],
            show_model_info=parameters['show_model_info'],

            # Dataloader parameters
            workers=parameters['workers'],
            batch_size=parameters['batch_size'],
            common_augs=parameters['common_augs'],
            train_time_augs=parameters['train_time_augs'],
            show_samples=parameters['show_samples'],
            shuffle_train=parameters['shuffle_train'],
            dataloaders_schema=parameters['dataloaders_schema'],

            # Find learning rate parameters
            find_lr_factor=parameters['find_lr_factor'],
            learning_rate=parameters['learning_rate'],

            # Common training parameters
            loss_name=parameters['loss_name'],
            optim_name=parameters['optim_name'],
            validation_metrics=parameters['validation_metrics'],
            checkpoint_metric=parameters['checkpoint_metric'],
            epochs=parameters['epochs'],
            save_n_best=parameters['save_n_best'],
            scheduler_name=parameters['scheduler_name'],
            early_stop_patience=parameters['early_stop_patience'],

            # Inference parameters
            tta_list=parameters['tta_list'],
            save_preds=parameters['save_preds'],
            save_inference_path=parameters['save_inference_path'],
            inference_with_labels=parameters['inference_with_labels']
        )

        if 'label_colors' in parameters:
            self._hyperparameters_dict['label_colors'] = parameters['label_colors']
        else:
            self._hyperparameters_dict['label_colors'] = None

        if self._hyperparameters_dict['task'] == 'segmentation' \
                and self._hyperparameters_dict['mode'] == 'multi':
            if self._hyperparameters_dict['label_colors'] is None:
                raise ValueError(
                    f"You should provide label_colors parameter for multiclass segmentation!"
                )
        if 'alphabet_characters' in parameters:
            self._hyperparameters_dict['alphabet_characters'] = parameters['alphabet_characters']
        else:
            self._hyperparameters_dict['alphabet_characters'] = None

        if self._hyperparameters_dict['task'] == 'ocr':
            if self._hyperparameters_dict['alphabet_characters'] is None:
                raise ValueError(
                    f"You should provide alphabet_characters parameter "
                    f"for OCR process!"
                )

        if self._hyperparameters_dict['device_ids'][0] == -1:
            self._hyperparameters_dict['allocate_on'] = 'cpu'
        else:
            self._hyperparameters_dict['allocate_on'] = 'gpu'

        # Get main pipeline class
        self.pipeline_object = Pipeline(
            task=self._hyperparameters_dict['task'],
            mode=self._hyperparameters_dict['mode'],
            allocate_on=self._hyperparameters_dict['allocate_on'],
            random_seed=self._hyperparameters_dict['seed'],
            time=self._hyperparameters_dict['time'],
            label_colors=self._hyperparameters_dict['label_colors'],
            alphabet_characters=self._hyperparameters_dict['alphabet_characters']
        )

        if 't' in self._hyperparameters_dict['stages']:
            same_out_path = False
        else:
            same_out_path = True

        model_data = self._get_model_data(same_out_path=same_out_path, verbose=1)
        self.model = model_data[0]
        self.model_save_path = model_data[1]
        self._hyperparameters_dict['model_parameters'] = model_data[3]

        initial_parameters = model_data[2]
        self._hyperparameters_dict['first_epoch'] = initial_parameters['epoch']
        self._hyperparameters_dict['first_step'] = initial_parameters['step']
        self._hyperparameters_dict['best_measure'] = 0

        self.paths = dict(
            train_path=self._hyperparameters_dict['train_path'],
            valid_path=self._hyperparameters_dict['valid_path'],
            holdout_path=self._hyperparameters_dict['holdout_path'],
            test_path=self._hyperparameters_dict['test_path']
        )
        self.dataloaders_dict = self._get_dataloaders()

    def _get_model_data(
            self,
            same_out_path,
            verbose: int = 1
    ) -> Tuple:
        model_data = self.pipeline_object.get_model(
            model_name=self._hyperparameters_dict['model_name'],
            device_ids=self._hyperparameters_dict['device_ids'],
            path_to_weights=self._hyperparameters_dict['path_to_weights'],
            model_parameters=self._hyperparameters_dict['model_parameters'],
            cudnn_bench=self._hyperparameters_dict['cudnn_benchmark'],
            out_path=self._hyperparameters_dict['output_path'],
            same_out_path=same_out_path,
            show_model=self._hyperparameters_dict['show_model_info'],
            verbose=verbose
        )
        return model_data

    def _get_dataloaders(self) -> Dict:
        default_dataloaders_schema = {
            'find_lr_dataloader':   ['train_path', True],
            'train_dataloader':     ['train_path', True],
            'valid_dataloader':     ['valid_path', True],
            'holdout_dataloader':   ['holdout_path', True],
            'test_dataloader':      ['test_path', False]
        }

        if self._hyperparameters_dict['dataloaders_schema'] is None:
            self._hyperparameters_dict['dataloaders_schema'] = default_dataloaders_schema

        print(
            f"Paths: \n"
            f"Find_lr: {self.paths[default_dataloaders_schema['find_lr_dataloader'][0]]}\n"
            f"Train: {self.paths[default_dataloaders_schema['train_dataloader'][0]]}\n"
            f"Validation: {self.paths[default_dataloaders_schema['valid_dataloader'][0]]}\n"
            f"Test: {self.paths[default_dataloaders_schema['test_dataloader'][0]]}"
        )
        dataloaders_dict = dict()
        for loader_type, parameters in self._hyperparameters_dict['dataloaders_schema'].items():
            if loader_type == 'find_lr_dataloader':
                batch_size = 5
            else:
                batch_size = self._hyperparameters_dict['batch_size']

            dataloader = self.pipeline_object.get_dataloaders(
                path_to_dataset=self.paths[parameters[0]],
                workers=self._hyperparameters_dict['workers'],
                batch_size=batch_size,
                is_train=parameters[1],
                show_samples=self._hyperparameters_dict['show_samples'],
                shuffle=self._hyperparameters_dict['shuffle_train'],
                common_augs=self._hyperparameters_dict['common_augs'],
                train_time_augs=self._hyperparameters_dict['train_time_augs']
            )
            dataloaders_dict[loader_type] = dataloader
        return dataloaders_dict

    def _find_lr(self) -> float:
        """ Function to find optimal learning rate

        :return: Optimal learning rate
        """
        # Get model
        find_lr_model, _, _ = self.pipeline_object.get_model(
            model_name=self._hyperparameters_dict['model_name'],
            device_ids=self._hyperparameters_dict['device_ids'],
            cudnn_bench=self._hyperparameters_dict['cudnn_benchmark'],
            path_to_weights=self._hyperparameters_dict['path_to_weights'],
            model_parameters=self._hyperparameters_dict['model_parameters']
        )

        find_lr_loader = self.dataloaders_dict['find_lr_dataloader']

        # Find learning rate process
        optimum_lr = self.pipeline_object.find_lr(
            model=find_lr_model,
            dataloader=find_lr_loader,
            loss_name=self._hyperparameters_dict['loss_name'],
            optim_name=self._hyperparameters_dict['optim_name'],
            lr_reduce_factor=self._hyperparameters_dict['find_lr_factor'],
            verbose=1,
            show_graph=True
        )
        del find_lr_loader
        gc.collect()
        return optimum_lr

    def _train(self):
        """ Training function

        :return: Trained model
        """
        self.model = self.pipeline_object.train(
            model=self.model,
            lr=self._hyperparameters_dict['learning_rate'],
            train_loader=self.dataloaders_dict['train_dataloader'],
            val_loader=self.dataloaders_dict['valid_dataloader'],
            loss_name=self._hyperparameters_dict['loss_name'],
            optim_name=self._hyperparameters_dict['optim_name'],
            metric_names=self._hyperparameters_dict['validation_metrics'],
            best_measure=self._hyperparameters_dict['best_measure'],
            first_step=self._hyperparameters_dict['first_step'],
            first_epoch=self._hyperparameters_dict['first_epoch'],
            chp_metric=self._hyperparameters_dict['checkpoint_metric'],
            n_epochs=self._hyperparameters_dict['epochs'],
            n_best=self._hyperparameters_dict['save_n_best'],
            scheduler=self._hyperparameters_dict['scheduler_name'],
            patience=self._hyperparameters_dict['early_stop_patience'],
            save_dir=self.model_save_path
        )

    def _validation(self) -> Dict:
        """ Validation process

        :return:
        """

        scores = self.pipeline_object.validation(
            model=self.model,
            dataloader=self.dataloaders_dict['holdout_dataloader'],
            metric_names=self._hyperparameters_dict['validation_metrics'],
            validation_mode='test',
            tta_list=self._hyperparameters_dict['tta_list']
        )
        return scores

    def _prediction(self, threshold: Union[List, float]) -> List:
        """ Inference process

        :param threshold:       Threshold to binarize predictions
        :return:
        """

        test_preds = self.pipeline_object.prediction(
            model=self.model,
            dataloader=self.dataloaders_dict['test_dataloader'],
            tta_list=self._hyperparameters_dict['tta_list'],
            save_batch=self._hyperparameters_dict['save_preds'],
            save_dir=self._hyperparameters_dict['save_inference_path'],
            with_labels=self._hyperparameters_dict['inference_with_labels']
        )

        # For segmentation
        if self._hyperparameters_dict['task'] == 'segmentation':
            if self._hyperparameters_dict['show_preds']:
                self.pipeline_object.visualize_preds(
                    preds=test_preds,
                    threshold=threshold,
                    with_labels=self._hyperparameters_dict['inference_with_labels']
                )

        return test_preds

    def main_pipeline(self, verbose: int = 1):
        scores = None
        stages = ['f', 't', 'v']
        if set(self._hyperparameters_dict['stages']).intersection(set(stages)):
            # Find learning rate
            if 'f' in self._hyperparameters_dict['stages']:
                if verbose == 1:
                    print('-' * 30, ' FINDING LEARNING RATE ', '-' * 30)
                self._hyperparameters_dict['learning_rate'] = self._find_lr()
            if 't' in self._hyperparameters_dict['stages']:
                # Save hyperparameters dictionary
                with open(os.path.join(self.model_save_path, 'hyperparameters.yml'), 'w') as f:
                    yaml.dump(self._hyperparameters_dict, f, default_flow_style=False)

                print('-' * 30, ' TRAINING ', '-' * 30)
                self._train()

            # Validation (metrics evaluation) line
            if 'v' in self._hyperparameters_dict['stages']:
                # Validation (metrics evaluation) line
                if verbose == 1:
                    print('-' * 30, ' VALIDATION ', '-' * 30)
                scores = self._validation()

                if 'thresholds.yml' not in os.listdir(self.model_save_path):
                    with open(os.path.join(self.model_save_path, 'thresholds.yml'), 'w') as f:
                        yaml.dump(scores, f, default_flow_style=False)
                if verbose == 1:
                    print(f"Validation scores: {scores}")

        # Prediction line
        if 'p' in self._hyperparameters_dict['stages']:
            if verbose == 1:
                print('-' * 30, ' PREDICTION ', '-' * 30)

            threshold = None
            if self._hyperparameters_dict['task'] == 'segmentation':
                if scores is not None:
                    if self._hyperparameters_dict['mode'] == 'binary':
                        threshold = scores[
                            self._hyperparameters_dict['checkpoint_metric']
                        ]['threshold']
                    elif self._hyperparameters_dict['mode'] == 'multi':
                        threshold = [
                            scores[cls][
                                self._hyperparameters_dict['checkpoint_metric']
                            ]['threshold'] for cls in scores.keys()
                        ]
                else:
                    threshold = self._hyperparameters_dict['segmentation_threshold']
            if verbose == 1:
                print(f'Prediction threshold: {threshold}')
            test_preds = self._prediction(threshold=threshold)
            return test_preds


