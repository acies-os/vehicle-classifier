import asyncio
import json
from collections import deque
from typing import Dict
from typing import List

import click
import numpy as np
import torch
from acies.FoundationSense.inference import ModelForInference
from acies.node import Node
from acies.node import common_options
from acies.node import logger
from acies.vehicle_classifier.utils import TimeProfiler
from acies.vehicle_classifier.utils import classification_msg
from acies.vehicle_classifier.utils import get_time_range
from acies.vehicle_classifier.utils import normalize_key
from acies.vehicle_classifier.utils import update_sys_argv


class FoundationSenseClassifier(Node):
    def __init__(self, weight, *args, **kwargs):
        # pass other args to parent type
        super().__init__(*args, **kwargs)

        # your inference model
        self.model = self.load_model(weight)

        # buffer incoming messages
        self.buffs = {"sei": deque(), "aco": deque()}

        # how many input messages the model needs to run inference once
        # each message contains 1s of data:
        #     seismic  :    200 samples
        #     acoustic : 16_000 samples
        self.input_len = 2

        # the topic we publish inference results to
        self.pub_topic = f"{self.get_hostname()}/vehicle"

    def load_model(self, path_to_weight: str):
        """Load model from give path."""

        logger.info(f"{ModelForInference} load model from {path_to_weight}")

        # load weight and initialize your model
        model = ModelForInference(path_to_weight)
        return model

    def inference(self):
        # buffer incoming messages
        for k, q in self.queue.items():
            if not q.empty():
                logger.debug(f"enqueue: {k}")
                data = q.get(False)
                data = json.loads(data)
                mod, data = normalize_key(data)
                self.buffs[mod].append(data)

        # check if we have enough data to run inference
        if (
            len(self.buffs["sei"]) >= self.input_len
            and len(self.buffs["aco"]) >= self.input_len
        ):
            input_sei: List[Dict] = [
                self.buffs["sei"].popleft() for _ in range(self.input_len)
            ]
            input_aco: List[Dict] = [
                self.buffs["aco"].popleft() for _ in range(self.input_len)
            ]

            start_time, end_time = get_time_range(input_sei)

            # flatten
            input_sei = np.array([x["samples"] for x in input_sei]).flatten()
            input_aco = np.array([x["samples"] for x in input_aco]).flatten()
            assert len(input_sei) == 200 * self.input_len, f"input_sei={len(input_sei)}"
            assert (
                len(input_aco) == 16000 * self.input_len
            ), f"input_aco={len(input_aco)}"

            # down sampling
            input_sei = input_sei[::2]
            input_aco = input_aco[::2]
            assert len(input_sei) == 100 * self.input_len, f"input_sei={len(input_sei)}"
            assert (
                len(input_aco) == 8000 * self.input_len
            ), f"input_aco={len(input_aco)}"

            input_sei = torch.from_numpy(input_sei).float()
            input_sei = torch.reshape(input_sei, (1, 1, 10, 20))
            input_aco = torch.from_numpy(input_aco).float()
            input_aco = torch.reshape(input_aco, (1, 1, 10, 1600))

            data = {
                "shake": {
                    "audio": input_aco,
                    "seismic": input_sei,
                }
            }

            with TimeProfiler() as timer:
                result = self.model(data)
                class_names = self.model.args.dataset_config["vehicle_classification"]["class_names"]
                # TODO (Tommy): add label to the resulting scores
                result = [
                    {"label": class_names[i], "conf": score} for i, score in enumerate(result[0].tolist())
                ]
            logger.debug(f"Inference time: {timer.elapsed_time_ns / 1e6} ms")

            msg = classification_msg(start_time, end_time, result)
            logger.info(f"{self.pub_topic}: {msg}")
            self.publish(self.pub_topic, json.dumps(msg))

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
@click.option(
    "-w",
    "--weight",
    help="Model weight",
    type=str,
)
@click.argument("model_args", nargs=-1, type=click.UNPROCESSED)
def main(mode, connect, listen, key, weight, model_args):

    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    # initialize the class
    classifier = FoundationSenseClassifier(
        mode=mode,
        connect=connect,
        listen=listen,
        sub_keys=key,
        pub_keys=[],
        weight=weight,
    )
    asyncio.run(classifier.run())
