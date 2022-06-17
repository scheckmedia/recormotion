from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from omegaconf import OmegaConf


@dataclass
class Resolution:
    width: int = 1280
    height: int = 720


@dataclass
class Video:
    input: Union[str, int]
    max_fps: int
    resolution: Resolution
    buffer_size: int


@dataclass
class Detection:
    torchserve_url: str
    labels: Dict[int, str]
    sampling_rate: int = 2
    debug: bool = False
    trigger_classes: List[str] = field(default_factory=lambda: [])


@dataclass
class Config:
    video: Video
    detection: Detection


class Configuration(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            schema = OmegaConf.structured(Config)
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            config = OmegaConf.load(config_path)
            cls._config = OmegaConf.merge(schema, config)

        return cls._instance

    @property
    def config(self) -> Config:
        return self._config
