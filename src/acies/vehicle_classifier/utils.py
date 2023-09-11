from typing import Dict
from typing import List
from typing import Tuple


def get_array(data: Dict) -> Tuple[str, List]:
    if "sh3" in data:
        return "sei", data["sh3"]
    elif "eh3" in data:
        return "sei", data["eh3"]
    elif "samples" in data:
        return "aco", data["samples"]
    else:
        raise KeyError(f"{data} should contain key: `sh3` or `eh3` or `samples`")
