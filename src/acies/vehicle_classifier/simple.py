import logging
from pathlib import Path

import click
import numpy as np
import torch
from acies.core import common_options, get_zconf, init_logger
from acies.FoundationSense.inference import ModelForInference
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import TimeProfiler, count_elements, update_sys_argv

logger = logging.getLogger('acies.infer')

class SimpleClassifier(Classifier):
    def __init__(self, modality, classifier_config_file, *args, **kwargs):
        self._single_modality = modality
        super().__init__(classifier_config_file, *args, **kwargs)
        
        # # your inference model
        # self.model = Inference(weight)

        # # buffer incoming messages
        # self.buffs = {'sei': deque(), 'aco': deque()}

        # # how many input messages the model needs to run inference once
        # # each message contains 1s of data:
        # #     seismic  :    200 samples
        # #     acoustic : 16_000 samples
        # self.input_len = 1  # intra window for now

        # # the topic we publish inference results to
        # self.model_name = 'simple'
        # self.pub_topic_vehicle = f'{self.get_hostname()}/{self.model_name}/vehicle'

        # # the topic we publish target distance results to
        # self.pub_topic_distance = f'{self.get_hostname()}/{self.model_name}/distance'

        # # distance classifier
        # self.distance_classifier = DistInference()

        # # 1. Variables for energy detector
        # self.acoustic_energy_buffer = []  # Buffer for energy level for acoustic signal
        # self.acoustic_energy_buffer_size = 2  # Maximum enegy level buffer size for acoustic signal

        # self.seismic_energy_buffer = []  # Buffer for energy level for seismic signal
        # self.seismic_energy_buffer_size = 2  # Maximum enegy level buffer size for seismic signal

    def load_model(self, classifier_config_file: Path):
        freq_mae = True if 'mae' in self.proc_name else False
        model = ModelForInference(classifier_config_file, freq_mae, modality=self._single_modality)

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
        arrays = {k.split('/')[-1]: v for k, v in arrays.items()}

        # data = {'shake': {'audio': acoustic_data, 'seismic': seismic_data}}
        data = {'shake': {}}
        for mod in self.modalities:
            mod_data = arrays[mod]
            if mod == 'geo':
                mod_data = mod_data[::2].reshape(1, 1, 10, 20)
            else:
                mod_data = mod_data[::2].reshape(1, 1, 10, 1600)
            mod_data = torch.from_numpy(mod_data)
            if mod == 'geo':
                data['shake']['seismic'] = mod_data
            else:
                data['shake']['audio'] = mod_data

        with TimeProfiler() as timer:
            logit = self.model(data)  # returns logits [[x, y, z, w]],
        elapsed_ms = timer.elapsed_time_ns / 1e6
        logger.debug(f'Time (ms) to infer: {elapsed_ms}')

        # result = {
        #     "gle350": logit[0][0],
        #     "miata": logit[0][1],
        #     "cx30": logit[0][2],
        #     "mustang": logit[0][3],
        # }
        result = dict(zip(np.arange(4), logit[0]))

        return result

    def inference(self):
        # buffer incoming messages
        for k, q in self.queue.items():
            if not q.empty():
                logger.debug(f'enqueue: {k}')
                data = q.get(False)
                data = json.loads(data)
                mod, data = normalize_key(data)
                self.buffs[mod].append(data)

        # publish distance info
        if len(self.buffs['sei']) >= 1 and len(self.buffs['aco']) >= 1:
            # access data without taking it out of the queue
            input_sei = self.buffs['sei'][-1]
            # access data without taking it out of the queue
            input_aco = self.buffs['aco'][-1]
            dist_input = {'x_sei': input_sei['samples'], 'x_aud': input_aco['samples']}
            dist: int = self.distance_classifier.predict_distance(dist_input)
            dist_msg = distance_msg(input_sei['timestamp'], self.model_name, dist)
            self.publish(self.pub_topic_distance, json.dumps(dist_msg))

            # 2. Calcualte current energy levels, update energy bufferes
            sei_energy, self.seismic_energy_buffer = calculate_mean_energy(
                input_sei['samples'],
                self.seismic_energy_buffer,
                self.seismic_energy_buffer_size,
            )
            aco_energy, self.acoustic_energy_buffer = calculate_mean_energy(
                input_aco['samples'],
                self.acoustic_energy_buffer,
                self.acoustic_energy_buffer_size,
            )
        # check if we have enough data to run inference
        if len(self.buffs['sei']) >= self.input_len and len(self.buffs['aco']) >= self.input_len:
            input_sei: List[Dict] = [self.buffs['sei'].popleft() for _ in range(self.input_len)]
            input_aco: List[Dict] = [self.buffs['aco'].popleft() for _ in range(self.input_len)]

            start_time, end_time = get_time_range(input_sei)

            # flatten
            input_sei = np.array([x['samples'] for x in input_sei]).flatten()
            input_aco = np.array([x['samples'] for x in input_aco]).flatten()
            assert len(input_sei) == 200 * self.input_len, f'input_sei={len(input_sei)}'
            assert len(input_aco) == 16000 * self.input_len, f'input_aco={len(input_aco)}'

            # down sampling
            input_sei = input_sei[::2]
            input_aco = input_aco[::16]
            assert len(input_sei) == 100 * self.input_len, f'input_sei={len(input_sei)}'
            assert len(input_aco) == 1000 * self.input_len, f'input_aco={len(input_aco)}'

            data = {
                'x_aud': input_aco,
                'x_sei': input_sei,
            }

            with TimeProfiler() as timer:
                result = self.model.predict(data)
            logger.debug(f'Inference time: {timer.elapsed_time_ns / 1e6} ms')

            # 3. Publish energy level and classification result
            msg = classification_msg(start_time, end_time, self.model_name, result, sei_energy, aco_energy)
            logger.info(f'{self.pub_topic_vehicle}: {msg}')
            self.publish(self.pub_topic_vehicle, json.dumps(msg))

    async def run(self):
        try:
            # Register your inference func as a callback.
            await self.add_callback(self.inference)

            # You can add more callbacks if needed, they will be run concurrently.
            # await self.add_callback(self.another_callback)
            # await self.add_callback(self.3rd_callback)

            # self.start() must be called
            await self.start()
        except KeyboardInterrupt:
            self.close()


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
    clf = SimpleClassifier(
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
