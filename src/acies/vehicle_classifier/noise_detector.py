import logging
import queue
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import click
import numpy as np
from acies.buffers import EnsembleBuffer
from acies.core import AciesMsg, Service, common_options, get_zconf, init_logger, pretty

logger = logging.getLogger('acies.noise_detector')


def get_node_name(msg: AciesMsg) -> str:
    reply_to = msg.reply_to
    return reply_to.split('/')[0]


class NoiseDetector(Service):
    def __init__(self, win_size: int, *args, **kwargs):
        # pass other args to parent type
        super().__init__(*args, **kwargs)

        self.buffer = EnsembleBuffer(Path.home() / '.acies' / 'noise.db')
        self.win_size = win_size

        # the topic we publish inference results to
        self.pub_topic = self.ns_topic_str('noise')
        logger.info(f'noise info published to {self.pub_topic}')

    def handle_message(self):
        try:
            topic, msg = self.msg_q.get_nowait()
            assert isinstance(msg, AciesMsg)
        except queue.Empty:
            return

        # if deactivated, drop the message
        if self.service_states.get('deactivated', False):
            return

        if any(topic.endswith(x) for x in ['geo', 'mic']):
            # msg.timestamp is in ns
            timestamp = int(msg.timestamp / 1e9)
            array = np.array(msg.get_payload())
            mod = 'geo' if topic.endswith('geo') else 'mic'
            energy = np.std(array).item()
            node_name = get_node_name(msg)
            self.buffer.add_entry(node_name, 'noise_detector', timestamp, {mod: energy}, None)
        else:
            logger.info(f'unhandled msg received at topic {topic}: {msg}')

    def detect(self):
        now = int(datetime.now().timestamp())
        oldest = now - self.win_size
        samples = self.buffer.get_range(oldest, now)
        # dict keys:
        # id integer primary key,
        # node_name text not null,
        # model_name text not null,
        # timestamp integer not null,
        # prediction text not null,
        # metadata text not null,
        # status text default 'unprocessed'

        # a prediction:
        # {'geo': 123.123}
        # {'mic': 123.123}
        raise NotImplementedError()
        logger.debug(f'{now}: {energy=}, {thresh=}, {result=}')

        msg = self.make_msg('json', result)
        self.send(self.pub_topic, msg)

    def run(self):
        self.schedule(0.1, self.handle_message, periodic=True)
        self.schedule(1, self.detect, periodic=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--win-size', help='Window size in seconds', type=int)
@click.option('--heartbeat-interval-s', help='Heartbeat interval in seconds', type=int, default=5)
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    win_size,
    heartbeat_interval_s,
):
    init_logger(f'{namespace}_{proc_name}.log', name='acies.noise_detector')
    z_conf = get_zconf(mode, connect, listen)

    # initialize the class
    clf = NoiseDetector(
        win_size=win_size,
        conf=z_conf,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # start
    clf.start()
