import logging
from pathlib import Path

import click
import numpy as np
import torch
from acies.FoundationSense.inference import ModelForInference
from acies.node.logging import init_logger
from acies.node.net import common_options, get_zconf
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import count_elements, update_sys_argv

logger = logging.getLogger('acies.infer')


class VibroFM(Classifier):
    def load_model(self, classifier_config_file: Path):
        freq_mae = True if 'mae' in self.proc_name else False
        model = ModelForInference(classifier_config_file, freq_mae)
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
        arrays = {k.split('/')[1]: v for k, v in arrays.items()}
        seismic_data = arrays['geo']
        acoustic_data = arrays['mic']

        seismic_data = seismic_data[::2].reshape(1, 1, 10, 20)
        acoustic_data = acoustic_data[::2].reshape(1, 1, 10, 1600)

        seismic_data = torch.from_numpy(seismic_data)
        acoustic_data = torch.from_numpy(acoustic_data)

        data = {'shake': {'audio': acoustic_data, 'seismic': seismic_data}}

        logit = self.model(data)  # returns logits [[x, y, z, w]],

        # result = {
        #     "gle350": logit[0][0],
        #     "miata": logit[0][1],
        #     "cx30": logit[0][2],
        #     "mustang": logit[0][3],
        # }
        result = dict(zip(np.arange(4), logit[0]))

        return result


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=click.Path(exists=True))
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
