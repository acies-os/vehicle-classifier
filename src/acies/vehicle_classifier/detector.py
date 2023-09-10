import asyncio
import json
import logging
import os
from collections import deque
from typing import Dict
from typing import List
from typing import Tuple

import click
import numpy as np
from acies.node import Node
from acies.node import common_options

LOG_LEVEL = os.environ.get("ACIES_LOG", "error").upper()
logging.basicConfig(level=LOG_LEVEL)


class Detector(Node):
    def __init__(self, weight, *args, **kwargs):
        # pass other args to parent type
        super().__init__(*args, **kwargs)

        # your inference logic
        self.model = self.load_model(weight)
        self.buffs = {"sei": deque(), "aco": deque()}

        # how many input messages the model needs
        # each message contains 1s of data:
        #     seismic  :    200 samples
        #     acoustic : 16_000 samples
        self.input_len = 3

    def load_model(self, path_to_weight: str):
        logging.info(f"load model from {path_to_weight}, but this is a dummy model")

        def dummy_model(*args, **kwargs):
            return [
                {"label": "AT-AT", "conf": 1.0},
                {"label": "Landspeeder", "conf": 0.0},
            ]

        return dummy_model

    @staticmethod
    def get_array(data: Dict) -> Tuple[str, List]:
        if "sh3" in data:
            return "sei", data["sh3"]
        elif "eh3" in data:
            return "sei", data["eh3"]
        elif "samples" in data:
            return "aco", data["samples"]
        else:
            raise KeyError(f"{data} should contain key: `sh3` or `eh3` or `samples`")

    def inference(self):
        # print("enter inference")
        # add messages to buffers
        for k, q in self.queue.items():
            if not q.empty():
                logging.debug(f"enqueu {k}")
                data = q.get(False)
                data = json.loads(data)
                mod, data = self.get_array(data)
                logging.debug(f"{mod}")
                self.buffs[mod].append(data)

        # check if we have enough data to run inference
        if (
            len(self.buffs["sei"]) >= self.input_len
            and len(self.buffs["aco"]) >= self.input_len
        ):
            logging.info("running inference")
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
            self.publish(f"{self.get_hostname()}/classifier", json.dumps(result))

    async def run(self):
        try:
            await self.add_callback(self.inference)
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
    detector = Detector(
        mode=mode,
        connect=connect,
        listen=listen,
        sub_keys=key,
        pub_keys=[],
        weight=weight,
    )
    asyncio.run(detector.run())
