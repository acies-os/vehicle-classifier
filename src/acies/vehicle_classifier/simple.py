import logging
from pathlib import Path

import click
import numpy as np
import torch
import os
import sys

# # TODO: Remove, for debugging only
# # Ensure the top-level directory is in sys.path
# top_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
# if top_level_path not in sys.path:
#     sys.path.append(top_level_path)

# # Ensure the src directory is in sys.path
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
# if src_path not in sys.path:
#     sys.path.append(src_path)

# # Ensure the acies directory is in sys.path
# acies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# if acies_path not in sys.path:
#     sys.path.append(acies_path)
    
# # Ensure the acies/vehicle_classifier directory is in sys.path
# vehicle_classifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# if vehicle_classifier_path not in sys.path:
#     sys.path.append(vehicle_classifier_path)


from acies.core import common_options, get_zconf, init_logger
from acies.FoundationSense.inference import ModelForInference
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import TimeProfiler, count_elements, update_sys_argv
import pandas as pd

###
import pickle
from acies.vehicle_classifier.utils_simple import extract_features
import pickle
import time
import matplotlib.pyplot as plt
import sys
import os
import json
import logging
from concurrent.futures import ProcessPoolExecutor


TEST = True
if TEST:
    logging.info("Running in test mode")
    print("Running in test mode")
version = "v1"


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
            f'loaded simple model to cpu, '
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
    
    #### old inference code to migrate to above infer methods ####
    def load_model_simple(self, label_model_path, formation_model_path, targets, formation_targets):
        # Load the label and formation models
        self.label_classifier = pickle.load(open(label_model_path, "rb"))
        self.formation_classifier = pickle.load(open(formation_model_path, "rb"))
        self.targets = targets
        self.formation_targets = formation_targets

    def predict_label(self, data):
        all_features = self.preprocess_features(data)
        label_pred = self.label_classifier.predict_proba(all_features)
        
        # Ensure the number of predictions matches the number of targets
        num_classes = label_pred.shape[1]
        if num_classes != len(self.targets):
            logging.warning(f"Number of classes in model ({num_classes}) does not match number of targets ({len(self.targets)})")
            label_pred = np.pad(label_pred, ((0, 0), (0, len(self.targets) - num_classes)), mode='constant', constant_values=0)

        label_result = {self.targets[i]: float(label_pred[0][i]) for i in range(len(self.targets))}
        return label_result

    def predict_formation(self, data):
        all_features = self.preprocess_features(data)
        formation_pred = self.formation_classifier.predict(all_features)
        formation_result = self.formation_targets[int(formation_pred[0])]
        return formation_result

    def preprocess_features(self, data):
        # Convert data to desired format for extract_features
        x_aud = data["x_aud"]
        x_sei = data["x_sei"]
        timestamp = data["timestamp"]
        station_id = data["station_id"]

        # Create a DataFrame in the format expected by extract_features
        samples = pd.DataFrame({
            'samples_acoustic': x_aud,
            'samples_seismic': x_sei,
            'timestamp_seconds': [timestamp] * len(x_aud),
            'label': [0] * len(x_aud),  # Placeholder, replace with actual label if available
            'formation': [0] * len(x_aud),  # Placeholder, replace with actual formation if available
            'id': [(f'{station_id}_0', timestamp)] * len(x_aud)  # Use a unique identifier
        })

        # Extract features
        all_features = extract_features(samples)
        
        # Remove the label and formation and id and timestamp columns, don't do anything if they don't exist
        all_features = all_features.drop(columns=['label', 'formation', 'id', 'timestamp_seconds'], errors='ignore')
        
        return all_features


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



##### BELOW IS FOR TESTING #####


def process_run_node(run_id, node_id, label_model_path, formation_model_path, targets, formation_targets, base_path, output_dir):
    logging.info(f"Processing run {run_id}, node {node_id}")

    try:
        inference = Inference(label_model_path, formation_model_path, targets, formation_targets)
        geo_file = os.path.join(base_path, f'run{run_id}_gq-{node_id}_geo.parquet')
        mic_file = os.path.join(base_path, f'run{run_id}_gq-{node_id}_mic.parquet')

        if not (os.path.exists(geo_file) and os.path.exists(mic_file)):
            logging.info(f"Skipping run {run_id}, node {node_id} - files not found")
            return

        df_mic = pd.read_parquet(mic_file, engine='pyarrow')
        df_geo = pd.read_parquet(geo_file, engine='pyarrow')
        
        if TEST:
            df_mic = df_mic.head(16000 * 10)
            df_geo = df_geo.head(200 * 10)

        if df_mic.empty or df_geo.empty:
            logging.warning(f"Empty data in run {run_id}, node {node_id}")
            return

        num_seconds = 1
        window_size_audio = num_seconds * 16000
        window_size_geo = num_seconds * 200

        samples_audio = df_mic['samples']
        samples_geo = df_geo['samples']

        samples_audio = np.array(samples_audio)
        samples_geo = np.array(samples_geo)
        total_seconds = len(samples_audio) // 16000

        data_trace = []
        label_predictions = []
        formation_predictions = []
        timestamped_predictions = {}

        for t_0 in range(total_seconds):
            if (t_0 + 1) * window_size_audio > samples_audio.shape[0]:
                break

            if (t_0 + 1) * window_size_geo > samples_geo.shape[0]:
                break

            audio_packet = samples_audio[t_0 * window_size_audio: (t_0 + 1) * window_size_audio]
            geo_packet = samples_geo[t_0 * window_size_geo: (t_0 + 1) * window_size_geo]

            # Downsample audio to 1000 Hz
            audio_packet = audio_packet[::16]

            # Check audio shape, upsample geo to match audio shape if needed
            if len(audio_packet) > len(geo_packet):
                geo_packet = np.interp(np.linspace(0, len(geo_packet), len(audio_packet)), np.arange(len(geo_packet)), geo_packet)

            station_ID = node_id
            timestamp = time.time()
            data = {
                'x_aud': audio_packet,
                'x_sei': geo_packet,
                'timestamp': timestamp,
                'station_id': station_ID
            }

            # Predict label and formation
            label_prediction = inference.predict_label(data)
            formation_prediction = inference.predict_formation(data)

            label_predictions.append(label_prediction)
            formation_predictions.append(formation_prediction)

            data_trace.append((audio_packet, geo_packet))
            
            # Save timestamped predictions
            timestamped_predictions[t_0] = {
                'label_prediction': label_prediction,
                'formation_prediction': formation_prediction
            }

        # Plot predictions and original data
        df_labels = pd.DataFrame(label_predictions)
        df_formations = pd.Series(formation_predictions, name='formation')

        # Find the predicted label for each second
        predicted_labels = df_labels.idxmax(axis=1)

        plt.figure(figsize=(10, 8))
        plt.subplot(4, 1, 1)
        plt.plot(predicted_labels, marker='o', linestyle='None')
        plt.yticks(range(len(inference.targets)), inference.targets)
        plt.title('Predicted Labels')
        plt.xlabel('Seconds')
        plt.ylabel('Labels')

        plt.subplot(4, 1, 2)
        plt.plot(df_formations, marker='o', linestyle='None')
        plt.yticks(range(len(inference.formation_targets)), inference.formation_targets)
        plt.title('Formation Predictions')
        plt.xlabel('Seconds')
        plt.ylabel('Formation')

        plt.subplot(4, 1, 3)
        plt.plot(samples_audio, label='Original Audio')
        plt.title('Original Audio Data')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(samples_geo, label='Original Geo')
        plt.title('Original Geo Data')
        plt.legend()

        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'run{run_id}_gq-{node_id}_results.png')
        plt.savefig(plot_filename)
        plt.close()

        # Save predictions to JSON
        json_filename = os.path.join(output_dir, f'run{run_id}_gq-{node_id}_results.json')
        with open(json_filename, 'w') as json_file:
            json.dump(timestamped_predictions, json_file, indent=4)
        
        logging.info(f"Finished processing run {run_id}, node {node_id}")
    except Exception as e:
        logging.error(f"Error processing run {run_id}, node {node_id}: {e}")


def main_data_loop():
    label_model_path = "/data/kara4/2023-graces-quarters/models/v1_label/model_1.pkl"
    formation_model_path = "/data/kara4/2023-graces-quarters/models/v1_formation/model_1.pkl"
    targets = ["background", "pedestrian", "fog-machine", "husky", "sedan", "silverado", "warthog", "polaris"]
    formation_targets = ["single", "multi-target", "trailing"]

    base_path = '/data/kara4/2023-graces-quarters/raw'
    print("Running in test mode")
    return
    if TEST:
        output_dir = f'logs/predictions_test_{version}'
    else:
        output_dir = f'logs/predictions_{version}'
    os.makedirs(output_dir, exist_ok=True)

    for run_id in range(37):
        for node_id in range(1, 5):
            process_run_node(run_id, node_id, label_model_path, formation_model_path, targets, formation_targets, base_path, output_dir)

if __name__ == "__main__":
    # main_data()
    # Uncomment the line below to test with loops
    main_data_loop()
