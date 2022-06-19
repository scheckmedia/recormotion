from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import OmegaConf


@dataclass
class Http:
    """Dataclass that holds settings for Flask"""

    host: str = "0.0.0.0"
    port: int = 80
    debug: bool = False
    threaded: bool = True


@dataclass
class Logging:
    """Dataclass representing logging settings

    :param level: log level, see python docs for possible values
    :type level: str
    :param file: log file path
    :type file: Optional[str]
    """

    level: str = "info"
    file: Optional[str] = None


@dataclass
class Resolution:
    """Dataclass that represents settings for a  video resolution

    :param width: width of the input / output video
    :type width: int
    :param height: height of the input / output video
    :type height: int
    """

    width: int = 1280
    height: int = 720


@dataclass
class VideoInput:
    """Dataclass that represents settings for video / stream input

    :param source: source of the video, could be file path or device number
    :type source: Union[str, int]
    :param max_fps: max number of FPS to read from a stream file
    :type max_fps: int
    :param resolution: input resolution to request from a video device
    :type resolution: Resolution
    :param buffer_size: numbr of frames to buffer to have a history before a trigger occurs
    :type buffer_size: int
    :param timecode: enable/disable a timecode in the video frame
    :type timecode: bool
    :param fourcc: additiona fourcc code for OpenCV to request from a video device
    :type fourcc: str
    """

    source: Union[str, int]
    max_fps: int
    resolution: Resolution
    buffer_size: int
    timecode: bool
    fourcc: Optional[str] = None


@dataclass
class VideoOutput:
    """Dataclass that represents video output settings

    :param folder: folder a trigger event video should be save in
    :type folder: str
    :param resolution: resolution of the output video
    :type resolution: Resolution
    :param filename: filename of the recording, can contain strftime formatter
    :type filename: str
    :param encoder: ffmpeg encoder to use for writing the video
                    on a RPI this should be h264_v4l2m2m for hardware encoding
    :type encoder: str
    :param fps: FPS of the output video
    :type fps: int
    :param ffmpeg_args: additional ffmpeg args like `-preset veryfast` for libx264
    :type ffmpeg_args: Optional[List[str]]
    """

    folder: str
    resolution: Resolution
    filename: str = "%Y-%m-%d-%H:%M:%S.mp4"
    encoder: str = "libx264"
    fps: int = 25
    ffmpeg_args: Optional[List[str]] = None


@dataclass
class Video:
    """dataclass for video settings"""

    input: VideoInput
    output: VideoOutput


@dataclass
class Detection:
    """Dataclass that holds settings for the motion detection

    :param torchserve_url: URL to the torchserve entry point e.g. http://ip:port/predictions/modelname
    :type torchserve_url: str
    :param labels:
    :type labels: Dict[int,str]
    :param sampling_rate:
    :type sampling_rate: int
    :param debug: falg to signal debugging mode, what will result in all detections will be visible in a frame
    :type debug: int
    :param trigger_detection_threshold: minimum score a detection should have to be valid
    :type trigger_detection_threshold: int
    :param trigger_invalidate_duration: number of seconds without a trigger detection until the recording stops
    :type trigger_invalidate_duration: int
    :param trigger_classes: List of potential class names that triggers an recording event
    :type trigger_classes: List[str]
    """

    torchserve_url: str
    labels: Dict[int, str]
    sampling_rate: int = 2
    debug: bool = False
    trigger_detection_threshold: float = 0.6
    trigger_invalidate_duration: int = 10
    trigger_classes: List[str] = field(default_factory=lambda: [])


@dataclass
class Config:
    """dataclass that holds the application settings"""

    http: Http
    logging: Logging
    video: Video
    detection: Detection


class Configuration:
    """Singleton object that holds the application configuration"""

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
        """Getter for access the application configuration

        :return: application configuration
        :rtype: Config
        """
        return self._config
