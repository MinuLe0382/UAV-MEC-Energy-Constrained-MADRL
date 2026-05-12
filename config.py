import json
import pathlib


def load_config(path: str = "config.json") -> dict:
    return json.loads(pathlib.Path(path).read_text())


CONFIG = load_config()
SYS = CONFIG["system"]
RL = CONFIG["rl"]
UNSPEC = CONFIG["unspecified_in_paper"]
