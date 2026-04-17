import json
from pathlib import Path
from typing import List

import math_verify


class Answer:
    def __init__(self, parsed_answer: List):
        self.data = parsed_answer

    def __repr__(self):
        return self.data.__repr__()

    def __hash__(self):
        return hash(self.data.__repr__())

    def __eq__(self, other):
        if isinstance(other, Answer):
            # order matters
            try:
                return math_verify.verify(self.data, other.data) and math_verify.verify(other.data, self.data)
            except ValueError:
                return False
        else:
            return False


def save_to_file(sample_list: List[dict], out_filename: str | Path, save_mode: str = 'w'):
    assert save_mode in ['w', 'a'], "Save mode should be either `w` or `a`."

    if len(sample_list) == 0:
        return

    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_filename, save_mode) as f:
        f.write("\n".join(sample_str_list) + "\n")
