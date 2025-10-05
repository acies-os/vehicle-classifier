import logging
from pathlib import Path

import click
import numpy as np
import torch
import librosa
from acies.core import common_options, get_zconf, init_logger
from acies.SPAR.inference import ModelForInference
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

        self.modalities = model.args.dataset_config['modalities']
        _mapping = {
            'seismic': 'geo',
            'acoustic': 'mic',
            'audio': 'mic',
            'sei': 'geo',
            'aco': 'mic',
        }
        self.modalities = [_mapping[x] for x in self.modalities]

        return model

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        arrays = {k: self.concat(v) for k, v in samples.items()}

        # split per-node per-mod data
        node_arrays = {}
        for k, v in arrays.items():
            node, mod = k.split('/')
            if node not in node_arrays:
                node_arrays[node] = {}
            node_arrays[node][mod] = v

        data = {}
        for mod in self.modalities:
            mod_video = []
            mod_valid = []
            for node in node_arrays:
                if node not in data or mod not in data[node]:
                    mod_video.append(torch.zeros(81, 1, 128, 1))
                    mod_valid.append(0)
                    continue
                mod_data = node_arrays[node][mod]
                if mod == 'geo':
                    mod_data -= torch.mean(mod_data)
                    mod_spec = librosa.feature.melspectrogram(
                        y=mod_data,
                        sr=200,
                        n_fft=80,
                        hop_length=5,
                        n_mels=128,
                        power=2.0,
                    )
                    mod_db = librosa.power_to_db(mod_spec, ref=np.max)
                    mod_video = torch.from_numpy(mod_db).T.unsqueeze(1).unsqueeze(-1)
                else:
                    mod_spec = librosa.feature.melspectrogram(
                        y=mod_data,
                        sr=16000,
                        n_fft=1600,
                        hop_length=400,
                        n_mels=128,
                        power=2.0,
                    )
                    mod_db = librosa.power_to_db(mod_spec, ref=np.max)
                    mod_video = torch.from_numpy(mod_db).T.unsqueeze(1).unsqueeze(-1)
                mod_video.append(mod_video)
                mod_valid.append(1)
            mod_video = torch.cat(mod_video, dim=-1)

            if mod == 'geo':
                data['seismic']['data'] = (mod_video + 49.52) / 26.55
                data['seismic']['valid'] = torch.tensor(mod_valid)
            else:
                data['acoustic']['data'] = (mod_video + 44.74) / 17.71
                data['acoustic']['valid'] = torch.tensor(mod_valid)

        data['vantage_ids'] = torch.tensor([0, 1, 2, 3])

        vantage_spatial_locations = self.model.args.dataset_config[
            'vantage_spatial_locations'
        ]
        vantage_spatial_locations = [loc for loc in vantage_spatial_locations.values()]
        data['vantage_spatial_locations'] = torch.tensor(vantage_spatial_locations)

        with TimeProfiler() as timer:
            logit = self.model(data)  # returns logits [[x, y, z, w]],
        elapsed_ms = timer.elapsed_time_ns / 1e6
        logger.debug(f'Time (ms) to infer: {elapsed_ms}')

        result = dict(zip(np.arange(4), logit[0]))

        return result


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