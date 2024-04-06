from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StreamBuffer:
    size: int
    # {'rs1/mic': {180000: np.ndarray([...]),
    #             },
    # }
    _data: dict[str, dict[int, np.ndarray]] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _meta: dict[str, dict[int, dict]] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _timestamps: Counter[int] = field(default_factory=Counter)

    def add(self, topic: str, timestamp: int, samples: np.ndarray, meta: dict):
        # topic: 'rs1/mic'
        self._data[topic][timestamp] = samples
        self._meta[topic][timestamp] = meta
        self._timestamps[timestamp] += 1

    def _check_size(self):
        while len(self._timestamps) > self.size:
            # find and delete the oldest timestamp
            t = min(self._timestamps.keys())

            for v in self._data.values():
                # v = {t1: np.ndarray([...]), t2: np.ndarray([...])}
                _ = v.pop(t, None)
                self._timestamps[t] -= 1

            for v in self._meta.values():
                # v = {t1: {'timestamp': t1, 'label': l1, ...}, t2: {'timestamp': t2, 'label': l2, ...}}
                _ = v.pop(t, None)

            if self._timestamps[t] == 0:
                del self._timestamps[t]

    def get(self, keys: list[str], n: int) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, dict]]]:
        """Get n-second of samples for all keys."""
        data = defaultdict(dict)
        data_meta = defaultdict(dict)

        for timestamp in sorted(self._timestamps):
            if all(timestamp - i in self._data[k] for k in keys for i in range(n)):
                for k in keys:
                    for i in reversed(range(n)):
                        t = timestamp - i
                        sample = self._data[k].pop(t)
                        sample_meta = self._meta[k].pop(t)

                        self._timestamps[t] -= 1
                        if self._timestamps[t] == 0:
                            del self._timestamps[t]

                        data[k][t] = sample
                        sample_meta['energy'] = np.std(sample)
                        data_meta[k][t] = sample_meta
                return data, data_meta
        raise ValueError('not enough data')
