import logging
import queue
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from acies.node.logging import init_logger
from acies.node.net import common_options, get_zconf
from acies.node.service import Service, pretty
from acies.vehicle_classifier.buffer import StreamBuffer
from acies.vehicle_classifier.utils import TimeProfiler, update_sys_argv

logger = logging.getLogger('acies.infer')

LABEL_TO_STR = {
    0: 'miata',
    1: 'gle350',
    2: 'mustang',
    3: 'cx30',
}


class Classifier(Service):
    def __init__(self, classifier_config_file, *args, **kwargs):
        # pass other args to parent type
        super().__init__(*args, **kwargs)

        # buffer 10s of data for each topic
        self.buffer = StreamBuffer(size=10)

        # how many input messages the model needs to run inference once
        # each message contains 1s of data:
        #     seismic  :    200 samples
        #     acoustic : 16_000 samples
        self.input_len = 2

        # the topic we publish inference results to
        self.pub_topic = self.ns_topic_str('vehicle')
        logger.info(f'classification result published to {self.pub_topic}')

        self._counter = 0

        # your inference model
        classifier_config_file = Path(classifier_config_file)
        assert classifier_config_file.exists(), f'{classifier_config_file} does not exist'

        self.modalities = []
        self.model = self.load_model(classifier_config_file)

    def load_model(self, path_to_weight: Path):
        """Load model from give path."""
        raise NotImplementedError

    def combine_meta(self, meta_data: dict[str, dict[int, dict]]):
        result = {'label': None, 'distance': None, 'mean_geo_energy': [], 'mean_mic_energy': []}
        for topic, topic_data in meta_data.items():
            topic = topic.split('/')[1]
            for t_meta in topic_data.values():
                result['label'] = t_meta['label']
                result['distance'] = t_meta['distance']
                if topic == 'geo':
                    result['mean_geo_energy'].append(t_meta['energy'])
                else:
                    result['mean_mic_energy'].append(t_meta['energy'])
        result['mean_geo_energy'] = np.mean(result['mean_geo_energy'])
        result['mean_mic_energy'] = np.mean(result['mean_mic_energy'])
        return result

    def run_inference(self):
        keys = [self.ns_topic_str(x) for x in self.modalities]
        try:
            samples, meta_data = self.buffer.get(keys, self.input_len)
        except ValueError:
            return

        with TimeProfiler() as timer:
            result = self.infer(samples)

        result = {LABEL_TO_STR[k]: v for k, v in result.items()}

        infer_time_ms = timer.elapsed_time_ns / 1e6
        log_msg = {'inference_time_ms': infer_time_ms}
        logger.debug(f'{log_msg}')
        msg = self.make_msg(
            'classification',
            result,
            meta={
                'timestamp': datetime.now().timestamp(),
                'inference_time_ms': infer_time_ms,
                'inputs': dict(meta_data),
            },
        )
        self.send('vehicle', msg)
        log_msg = pretty(asdict(msg), max_seq_length=6, max_width=500, newline='')
        logger.debug(f'inference result: {log_msg}')

        pred, confidence = max(result.items(), key=lambda x: x[1])

        one_meta = self.combine_meta(meta_data)
        logger.info(
            f'detected {pred:<7} ({confidence:.4f}): '
            f'truth={one_meta["label"]:<7}, '
            f'D={one_meta["distance"]:<6.2f}m, '
            f'E(geo)={one_meta["mean_geo_energy"]:<8.2f}, '
            f'E(mic)={one_meta["mean_mic_energy"]:<8.2f}'
        )
        log_msg = {
            'pred_label': pred,
            'confidence': confidence,
            'true_label': one_meta['label'],
            'distance': one_meta['distance'],
            'energy_geo': one_meta['mean_geo_energy'],
            'energy_mic': one_meta['mean_mic_energy'],
        }
        logger.info(f'{log_msg}')

    def infer(self, samples):
        raise NotImplementedError()

    @staticmethod
    def concat(arrays: dict[int, np.ndarray]):
        # the samples in v are sorted by timestamp
        assert list(arrays.keys()) == sorted(arrays.keys())
        return np.concatenate(list(arrays.values()))

    def handle_message(self):
        try:
            topic, msg = self.msg_q.get(timeout=0.1)
        except queue.Empty:
            return

        if any(topic.endswith(x) for x in ['geo', 'mic']):
            timestamp = int(msg.meta['timestamp'])
            array = np.array(msg.payload['samples'])
            self.buffer.add(topic, timestamp, array, msg.meta)
        else:
            logger.info(f'unhandled msg received at topic {topic}: {msg}')

    def run(self):
        while True:
            self._counter += 1
            self.handle_message()
            self.run_inference()


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=str)
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
def main(mode, connect, listen, topic, namespace, proc_name, weight, model_args):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    init_logger(f'{namespace}_{proc_name}.log', get_logger='acies.infer')
    z_conf = get_zconf(mode, connect, listen)

    # initialize the class
    clf = Classifier(
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
