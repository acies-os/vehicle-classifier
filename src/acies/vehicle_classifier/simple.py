import logging
from pathlib import Path

import click
import numpy as np
import torch
import os
import sys


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

PARALLEL = True
TEST = False
if TEST:
    logging.info("Running in test mode")
    print("Running in test mode")
version = "v2"

TARGETS = ["background", "pedestrian", "fog-machine", "husky", "sedan", "silverado", "warthog", "polaris"]
FORMATION_TARGETS = ["single", "multi-target"]


logger = logging.getLogger('acies.infer')

class SimpleClassifier(Classifier):
    def __init__(self, modality, classifier_config_file, *args, **kwargs):
        self._single_modality = modality
        self.modalities = ['mic', 'geo']
        
        # super().__init__(classifier_config_file, *args, **kwargs)
        
        self.model = self.load_model(classifier_config_file)
        self.formation_model = self.load_model(kwargs['formation_classifier_config_file'])
        # Load the label and formation models
    
    def infer(self, samples: dict[str, dict[int, np.ndarray]],option = 'label'):
        
        # get only the first second of data for now
        current_timestamps = list(samples[list(samples.keys())[0]].keys())
        current_timestamp = current_timestamps[0] # first timestamp
            
        samples = {k: v[current_timestamp] for k, v in samples.items()}
        
        # arrays = {k: self.concat(v) for k, v in samples.items()}
        
        # go as regular
        arrays = {k: v for k, v in samples.items()}
        arrays = {k.split('/')[-1]: v for k, v in arrays.items()}

        audio_downsampled_len_1sec = 1000

        if option == 'label':
            # data = {'shake': {'audio': acoustic_data, 'seismic': seismic_data}}
            # data = {
            #     'x_aud': audio_packet,
            #     'x_sei': geo_packet,
            #     'timestamp': timestamp,
            #     'station_id': station_ID
            # }

            data = {}
            for mod in self.modalities:
                mod_data = arrays[mod]
                if mod == 'geo':
                    mod_data = self.upsample_nearest(mod_data, audio_downsampled_len_1sec)
                    data['x_sei'] = mod_data
                    assert len(mod_data) == audio_downsampled_len_1sec
                else:
                    mod_data = mod_data[::16] # downsample to 1000 Hz
                    assert len(mod_data) == audio_downsampled_len_1sec
                    data['x_aud'] = mod_data
                    
            
            data['timestamp'] = current_timestamp
            data['station_id'] = 0
            with TimeProfiler() as timer:
                logit = self.predict_label(data)
            elapsed_ms = timer.elapsed_time_ns / 1e6
            logger.debug(f'Time (ms) to infer: {elapsed_ms}')

            
            # create a result dict from TARGET 
            logits = []
            for target in TARGETS:
                logits.append(logit[target])
                
            result = dict(zip(np.arange(len(TARGETS)), logits))

            return result
        
        
        elif option == 'formation':
            data = {}
            for mod in self.modalities:
                mod_data = arrays[mod]
                if mod == 'geo':
                    mod_data = self.upsample_nearest(mod_data, audio_downsampled_len_1sec)
                    data['x_sei'] = mod_data
                    assert len(mod_data) == audio_downsampled_len_1sec
                else:
                    mod_data = mod_data[::16]
                    assert len(mod_data) == audio_downsampled_len_1sec
                    data['x_aud'] = mod_data
                    
            data['timestamp'] = current_timestamp
            data['station_id'] = 0
            with TimeProfiler() as timer:
                logit = self.predict_formation(data)
            elapsed_ms = timer.elapsed_time_ns / 1e6
            logger.debug(f'Time (ms) to infer: {elapsed_ms}')
            
            
            # create a result dict from FORMATION_TARGETS
            logits = []
            for target in FORMATION_TARGETS:
                logits.append(logit[target])
                
            result = dict(zip(np.arange(len(FORMATION_TARGETS)), logits))
            
            return result
        
        else:
            raise ValueError(f"Invalid option: {option}")
        
    
    def upsample_nearest(self, data, new_length):
        """
        Upsample the input data to the specified new length using the nearest neighbor method.

        Parameters:
        - data (numpy.ndarray): The input data array to upsample.
        - new_length (int): The desired length of the upsampled data array.

        Returns:
        - numpy.ndarray: The upsampled data array.
        """
        original_len = len(data)
        if original_len == 0:
            raise ValueError("Input data array is empty.")

        # Calculate the ratio of the new length to the original length
        ratio = new_length / original_len

        # Create the target indices based on the desired length
        target_indices = np.arange(new_length)

        # Map the target indices to the nearest original indices
        nearest_indices = np.round(target_indices / ratio).astype(int)

        # Ensure indices are within the range of original indices
        nearest_indices = np.clip(nearest_indices, 0, original_len - 1)

        # Create the upsampled array with nearest neighbor values
        upsampled_data = data[nearest_indices]

        return upsampled_data

    
    #### old inference code to migrate to above infer methods ####
    def load_model(self, model_path, targets=None, formation_targets=None):
        # Load the label and formation models
        self.classifier = pickle.load(open(model_path, "rb"))
        return self.classifier

    def predict_label(self, data):
        all_features = self.preprocess_features(data)
        label_pred = self.model.predict_proba(all_features)
        
        
        label_result = {TARGETS[i]: float(label_pred[0][i]) for i in range(len(TARGETS))}
        return label_result

    def predict_formation(self, data):
        all_features = self.preprocess_features(data)
        formation_pred = self.formation_model.predict_proba(all_features)
        
        formation_result = {FORMATION_TARGETS[i]: float(formation_pred[0][i]) for i in range(len(FORMATION_TARGETS))}
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

def get_label_from_prediction(prediction,targets):
    # find the key with max value
    max_key = max(prediction, key=prediction.get)
    return targets[max_key]

def process_run_node(run_id, node_id, label_model_path, formation_model_path, targets, formation_targets, base_path, output_dir):
    logging.info(f"Processing run {run_id}, node {node_id}")

    

    label_inference = SimpleClassifier(
    classifier_config_file=label_model_path,
    formation_classifier_config_file=formation_model_path,
    modality='multimodal')
    
    
    # formation_inference = SimpleClassifier(classifier_config_file=formation_model_path,feature_twin=False, twin_model='multimodal', twin_buff_len=2, sync_interval=1, heartbeat_interval_s=5, modality='multimodal')
    geo_file = os.path.join(base_path, f'run{run_id}_gq-{node_id}_geo.parquet')
    mic_file = os.path.join(base_path, f'run{run_id}_gq-{node_id}_mic.parquet')

    if not (os.path.exists(geo_file) and os.path.exists(mic_file)):
        logging.info(f"Skipping run {run_id}, node {node_id} - files not found")
        return

    df_mic = pd.read_parquet(mic_file, engine='pyarrow')
    df_geo = pd.read_parquet(geo_file, engine='pyarrow')
    
    if TEST:
        seconds = 5
        df_mic = df_mic.head(16000 * seconds)
        df_geo = df_geo.head(200 * seconds)

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
        
        data = {
            'rs10/geo': {
                t_0: geo_packet,
                t_0 + 1: geo_packet
            },
            'rs10/mic': {
                t_0: audio_packet,
                t_0 + 1: audio_packet
            },
        }

        # Predict label and formation
        label_prediction = label_inference.infer(data, option='label')
        formation_prediction = label_inference.infer(data, option='formation')
        
        label_prediction = get_label_from_prediction(label_prediction, targets)
        formation_prediction = get_label_from_prediction(formation_prediction, formation_targets)

        label_predictions.append(label_prediction)
        formation_predictions.append(formation_prediction)

        # data_trace.append((audio_packet, geo_packet))
        
        # Save timestamped predictions
        timestamped_predictions[t_0] = {
            'label_prediction': label_prediction,
            'formation_prediction': formation_prediction
        }

    # Plot predictions and original data
    predicted_labels = pd.Series(label_predictions, name = 'label')
    df_formations = pd.Series(formation_predictions, name='formation')
    
    # predicted_labels = predicted_labels.idxmax(axis=1)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(predicted_labels, marker='o', linestyle='None')
    plt.yticks(range(len(targets)),targets)
    plt.title('Predicted Labels')
    plt.xlabel('Seconds')
    plt.ylabel('Labels')

    plt.subplot(4, 1, 2)
    plt.plot(df_formations, marker='o', linestyle='None')
    plt.yticks(range(len(formation_targets)), formation_targets)
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


def main_data(parallel=False):
    label_model_path = "/data/kara4/2023-graces-quarters/models/v2_label/final_model.pkl"
    formation_model_path = "/data/kara4/2023-graces-quarters/models/v2_formation/final_model.pkl"
    targets = TARGETS
    formation_targets = FORMATION_TARGETS

    base_path = '/data/kara4/2023-graces-quarters/raw'
    if TEST:
        output_dir = f'logs/predictions_test_{version}'
    else:
        output_dir = f'logs/predictions_{version}'
    os.makedirs(output_dir, exist_ok=True)

    if not parallel:
        print("Running in serial mode")
        for run_id in range(37):
            for node_id in range(1, 5):
                process_run_node(run_id, node_id, label_model_path, formation_model_path, targets, formation_targets, base_path, output_dir)

    else:
        print("Running in parallel mode")
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_run_node, run_id, node_id, label_model_path, formation_model_path, targets, formation_targets, base_path, output_dir)
                for run_id in range(37)
                for node_id in range(1, 5)
            ]
            for future in futures:
                future.result()  # Wait for all futures to complete

if __name__ == "__main__":
    main_data(parallel=PARALLEL)
    