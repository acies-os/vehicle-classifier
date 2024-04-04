import logging
import time
from datetime import datetime
from pathlib import Path

import click
from acies.node.net import common_options
from acies.node.service import Service
from acies.vehicle_classifier.buffer import StreamBuffer
from acies.vehicle_classifier.utils import TimeProfiler
from acies.vehicle_classifier.utils import update_sys_argv

logger = logging.getLogger("acies.infer")


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
        self.input_len = 3

        # the topic we publish inference results to
        self.pub_topic = self.ns_topic_str("vehicle")
        logger.info(f"classification result published to {self.pub_topic}")

        self._counter = 0

        # your inference model
        classifier_config_file = Path(classifier_config_file)
        assert (
            classifier_config_file.exists()
        ), f"{classifier_config_file} does not exist"
        self.model = self.load_model(classifier_config_file)

    def load_model(self, path_to_weight: Path):
        """Load model from give path."""
        raise NotImplementedError

    def run_inference(self):
        # # buffer incoming messages
        # for k, q in self.buffs.items():
        #     if not q.empty():
        #         logger.debug(f"enqueue: {k}")
        #         data = q.get(False)
        #         data = json.loads(data)
        #         mod, data = normalize_key(data)
        #         self.buffs[mod].append(data)

        # # check if we have enough data to run inference
        # if (
        #     len(self.buffs["sei"]) >= self.input_len
        #     and len(self.buffs["aco"]) >= self.input_len
        # ):
        #     input_sei: List[Dict] = [
        #         self.buffs["sei"].popleft() for _ in range(self.input_len)
        #     ]
        #     input_aco: List[Dict] = [
        #         self.buffs["aco"].popleft() for _ in range(self.input_len)
        #     ]

        #     start_time, end_time = get_time_range(input_sei)

        #     # flatten
        #     input_sei = np.array([x["samples"] for x in input_sei]).flatten()
        #     input_aco = np.array([x["samples"] for x in input_aco]).flatten()
        #     assert len(input_sei) == 200 * self.input_len, f"input_sei={len(input_sei)}"
        #     assert (
        #         len(input_aco) == 16000 * self.input_len
        #     ), f"input_aco={len(input_aco)}"

        #     # down sampling
        #     input_sei = input_sei[::2]
        #     input_aco = input_aco[::160]
        #     assert len(input_sei) == 100 * self.input_len, f"input_sei={len(input_sei)}"
        #     assert len(input_aco) == 100 * self.input_len, f"input_aco={len(input_aco)}"

        #     with TimeProfiler() as timer:
        #         result = self.model(input_sei, input_aco)
        #     logger.debug(f"Inference time: {timer.elapsed_time_ns / 1e6} ms")

        #     msg = classification_msg(start_time, end_time, result)
        #     logger.info(f"{self.pub_topic}: {msg}")
        #     self.publish(self.pub_topic, json.dumps(msg))
        keys = [self.ns_topic_str(x) for x in self.model.modalities]
        try:
            samples = self.buffer.get(keys, self.input_len)
        except ValueError:
            return

        with TimeProfiler() as timer:
            result = self.infer(samples)
        infer_time_ms = timer.elapsed_time_ns / 1e6
        log_msg = {"inference_time_ms": infer_time_ms}
        logger.debug(f"{log_msg}")
        msg = self.make_msg(
            "classification",
            result,
            meta={
                "timestamp": datetime.now().timestamp(),
                "inference_time_ms": infer_time_ms,
            },
        )
        self.send("vehicle", msg)

    def infer(self, samples):
        raise NotImplementedError()

    def handle_message(self):
        time.sleep(0.1)

    def run(self):
        while True:
            self._counter += 1
            self.handle_message()
            self.run_inference()


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option("--weight", help="Model weight", type=str)
@click.argument("model_args", nargs=-1, type=click.UNPROCESSED)
def main(mode, connect, listen, topic, namespace, weight, model_args):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    # initialize the class
    clf = Classifier(
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        classifier_config_file=weight,
    )

    # start
    clf.start()
