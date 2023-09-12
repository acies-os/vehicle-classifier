from typing import Dict
from typing import List
from typing import Tuple


def normalize_key(data: Dict) -> Tuple[str, Dict]:
    if "sh3" in data:
        data["samples"] = data.pop("sh3")
        return "sei", data
    elif "eh3" in data:
        data["samples"] = data.pop("eh3")
        return "sei", data
    elif "samples" in data:
        return "aco", data
    else:
        raise KeyError(f"{data} should contain key: `sh3` or `eh3` or `samples`")


def get_time_range(data: List[Dict]) -> Tuple[int, int]:
    start = data[0]["timestamp"]
    end = data[-1]["timestamp"]
    return start, end


def classification_msg(start: int, end: int, result: List[Dict]) -> Dict:
    msg = {"start": start, "end": end, "result": result}
    return msg
