import asyncio
import json
from collections import deque

import click
import numpy as np
from acies.node import Node
from acies.node import common_options
from acies.node import logger
from acies.vehicle_classifier.utils import get_array


class DeepSenseClassifier(Node):
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
        self.input_len = 3

        # the topic we publish inference results to
        self.pub_topic = f"{self.get_hostname()}/classifier"

    def load_model(self, path_to_weight: str):
        """Load model from give path."""

        logger.info(f"load model from {path_to_weight}, but this is a dummy model")

        def dummy_model(*args, **kwargs):
            return [
                {"label": "AT-AT", "conf": 1.0},
                {"label": "Landspeeder", "conf": 0.0},
            ]

        return dummy_model

    def inference(self):
        # buffer incoming messages
        for k, q in self.queue.items():
            if not q.empty():
                logger.debug(f"enqueue: {k}")
                data = q.get(False)
                data = json.loads(data)
                mod, data = get_array(data)
                self.buffs[mod].append(data)

        # check if we have enough data to run inference
        if (
            len(self.buffs["sei"]) >= self.input_len
            and len(self.buffs["aco"]) >= self.input_len
        ):
            input_sei = [self.buffs["sei"].popleft() for _ in range(self.input_len)]
            input_aco = [self.buffs["aco"].popleft() for _ in range(self.input_len)]

            # flatten
            input_sei = np.array(input_sei).flatten()
            input_aco = np.array(input_aco).flatten()
            assert len(input_sei) == 200 * self.input_len, f"input_sei={len(input_sei)}"
            assert (
                len(input_aco) == 16000 * self.input_len
            ), f"input_aco={len(input_aco)}"

            # down sampling
            input_sei = input_sei[::2]
            input_aco = input_aco[::160]
            assert len(input_sei) == 100 * self.input_len, f"input_sei={len(input_sei)}"
            assert len(input_aco) == 100 * self.input_len, f"input_aco={len(input_aco)}"

            result = self.model(input_sei, input_aco)
            logger.info(f"{self.pub_topic}: {result}")
            self.publish(self.pub_topic, json.dumps(result))

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


@click.command()
@common_options
@click.option(
    "-w",
    "--weight",
    help="Model weight",
    type=str,
)
def main(mode, connect, listen, key, weight):
    classifier = DeepSenseClassifier(
        mode=mode,
        connect=connect,
        listen=listen,
        sub_keys=key,
        pub_keys=[],
        weight=weight,
    )
    asyncio.run(classifier.run())
