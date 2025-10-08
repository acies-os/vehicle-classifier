import logging
from pathlib import Path

import click
import numpy as np
import torch
from acies.core import common_options, get_zconf, init_logger
from acies.SPAR.inference import ModelForInference
from acies.SPAR.input_utils.augmented_dataset import process_data
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import TimeProfiler, count_elements, update_sys_argv

logger = logging.getLogger('acies.infer')


class SPAR(Classifier):
    def __init__(self, classifier_config_file, *args, **kwargs):
        super().__init__(classifier_config_file, *args, **kwargs)

    def load_model(self, classifier_config_file: Path):
        model = ModelForInference(classifier_config_file)

        logger.info(
            f'loaded model to cpu, '
            f'definition from {ModelForInference.__name__}, '
            f'weights from {classifier_config_file}, '
            f'#params={len(list(model.parameters()))}, '
            f'#elements={count_elements(model)}'
        )

        self.modalities = ['spar_feature']
        self.required_nodes = ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6']

        return model

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        data = [samples[f'{n}/{m}'][0] for n in self.required_nodes for m in self.modalities]

        seismic_data = [torch.from_numpy(a[:256]) for a in data]
        acoustic_data = [torch.from_numpy(a[256:]) for a in data]

        data = process_data(False, False, self.required_nodes, acoustic_data, seismic_data, False, None, None)

        outputs = self.model(data)

        return outputs


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=click.Path(exists=True))
@click.option('--modality', type=str, help='Single modality: seismic, audio')
@click.option('--sync-interval', help='Sync interval in seconds', type=int, default=1)
@click.option('--feature-twin', help='Enable digital twin features', is_flag=True, default=False)
@click.option('--twin-model', help='Model used in the digital twin', type=str, default='multimodal')
@click.option('--twin-buff-len', help='Buffer length in the digital twin', type=int, default=2)
@click.option('--heartbeat-interval-s', help='Heartbeat interval in seconds', type=int, default=5)
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    deactivated,
    weight,
    modality,
    model_args,
    sync_interval,
    feature_twin,
    twin_model,
    twin_buff_len,
    heartbeat_interval_s,
):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    log_file = f'{namespace.replace("/", "_")}_{proc_name.replace("/", "_")}.log'
    init_logger(log_file, name='acies')
    z_conf = get_zconf(mode, connect, listen)

    logger.debug(f'{modality=}')

    # initialize the class
    clf = SPAR(
        modality=modality,
        conf=z_conf,
        twin_model=twin_model,
        twin_buff_len=twin_buff_len,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        deactivated=deactivated,
        classifier_config_file=weight,
        sync_interval=sync_interval,
        feature_twin=feature_twin,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # start
    clf.start()