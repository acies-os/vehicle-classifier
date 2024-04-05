import logging
from pathlib import Path

import click
import numpy as np
from acies.FoundationSense.inference import ModelForInference
from acies.node.logging import init_logger
from acies.node.net import common_options, get_zconf
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import count_elements, update_sys_argv

logger = logging.getLogger('acies.infer')


class VibroFM(Classifier):
    def load_model(self, classifier_config_file: Path):
        model = ModelForInference(classifier_config_file)
        logger.info(
            f'loaded model to cpu, '
            f'definition from {ModelForInference.__name__}, '
            f'weights from {classifier_config_file}, '
            f'#params={len(list(model.parameters()))}, '
            f'#elements={count_elements(model)}'
        )
        self.modalities = model.args.dataset_config['modality_names']
        _mapping = {'seismic': 'geo', 'acoustic': 'mic', 'audio': 'mic', 'sei': 'geo', 'aco': 'mic'}
        self.modalities = [_mapping[x] for x in self.modalities]
        return model

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        arrays = {k: self.concat(v) for k, v in samples.items()}
        # now arrays: dict[str, np.ndarray]
        # {'rs1/geo': np.array([...]), 'rs1/mic': np.array([...])}

        # result = self.model(arrays)
        result = {}

        return result


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=str)
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
def main(mode, connect, listen, topic, namespace, proc_name, weight, model_args):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    init_logger(f'{namespace}_{proc_name}.log', get_logger='acies')
    z_conf = get_zconf(mode, connect, listen)

    # initialize the class
    clf = VibroFM(
        conf=z_conf,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        classifier_config_file=weight,
    )

    # start
    clf.start()
