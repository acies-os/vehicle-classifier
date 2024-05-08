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
                result['label'] = t_meta.get('label')
                result['distance'] = t_meta.get('distance')
                result[f'mean_{topic}_energy'].append(t_meta.get('energy'))
        result['mean_geo_energy'] = np.mean(result['mean_geo_energy'])
        result['mean_mic_energy'] = np.mean(result['mean_mic_energy'])
        return result

    def get_keys_per_node(self, modalities):
        keys = list(self.buffer._data.keys())
        nodes = set(x.split('/')[0] for x in keys)
        ns_keys = {n: [f'{n}/{m}' for m in modalities] for n in nodes}
        return ns_keys

    def run_inference(self):
        node_keys = self.get_keys_per_node(self.modalities)
        for node, keys in node_keys.items():
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
            self.send(f'{node}/vehicle', msg)
            log_msg = pretty(asdict(msg), max_seq_length=6, max_width=500, newline='')
            logger.debug(f'inference result: {log_msg}')

            pred, confidence = max(result.items(), key=lambda x: x[1])

            one_meta = self.combine_meta(meta_data)
            log_msg = {
                'pred_label': pred,
                'confidence': confidence,
                'true_label': one_meta['label'],
                'distance': one_meta['distance'],
                'energy_geo': one_meta['mean_geo_energy'],
                'energy_mic': one_meta['mean_mic_energy'],
            }
            logger.info(f'{log_msg}')

            console_msg = f'detected {pred:<7} ({confidence:.4f}): '
            if one_meta['label'] is not None:
                console_msg += f'truth={one_meta["label"]:<7}, '
            else:
                console_msg += f'truth={"n/a":<7}, '
            if one_meta['distance'] is not None:
                console_msg += f'D={one_meta["distance"]:<6.2f}m, '
            else:
                console_msg += f'D={"n/a":<6}m, '
            console_msg += f'E(geo)={one_meta["mean_geo_energy"]:<8.2f}, E(mic)={one_meta["mean_mic_energy"]:<8.2f}'
            logger.info(console_msg)

    def infer(self, samples):
        raise NotImplementedError()

    @staticmethod
    def concat(arrays: dict[int, np.ndarray]):
        # the samples in v are sorted by timestamp
        assert list(arrays.keys()) == sorted(arrays.keys())
        return np.concatenate(list(arrays.values()))

    def handle_message(self):
        try:
            topic, msg = self.msg_q.get_nowait()
        except queue.Empty:
            return

        # if deactivated, drop the message
        if self.service_states.get('deactivated', False):
            return

        if any(topic.endswith(x) for x in ['geo', 'mic']):
            timestamp = int(msg.meta['timestamp'])
            array = np.array(msg.payload['samples'])
            self.buffer.add(topic, timestamp, array, msg.meta)
        else:
            logger.info(f'unhandled msg received at topic {topic}: {msg}')

    def log_activate_status(self):
        if self.service_states.get('deactivated', False):
            logger.debug('currently deactivated, standing by')

    def run(self):
        self.sched_periodic(2, self.log_activate_status)
        self.sched_periodic(0.1, self.handle_message)
        self.sched_periodic(1, self.run_inference)
        self._scheduler.run()


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
