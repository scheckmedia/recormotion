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

    Args:
        level (str): log level, see python docs for possible values
        file (Optional[str]): log file path
    """

    level: str = "info"
    file: Optional[str] = None


@dataclass
class Resolution:
    """Dataclass that represents settings for a  video resolution

    Args:
        width (int): width of the input / output video
        height (int): height of the input / output video
    """

    width: int = 1280
    height: int = 720


@dataclass
class VideoInput:
    """Dataclass that represents settings for video / stream input

    Args:
        source (Union[str, int]): source of the video, could be file path or device number
        max_fps (int): max number of FPS to read from a stream file
        resolution (Resolution): input resolution to request from a video device
        buffer_size (int): numbr of frames to buffer to have a history before a trigger occurs
        timecode (bool): enable/disable a timecode in the video frame
        fourcc (str): additiona fourcc code for OpenCV to request from a video device
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

    Args:
        folder (str): folder a trigger event video should be save in
        resolution (Resolution): resolution of the output video
        filename (str): filename of the recording, can contain strftime formatter
        encoder (str): ffmpeg encoder to use for writing the video
                        on a RPI this should be h264_v4l2m2m for hardware encoding
        fps (int): FPS of the output video, if -1 or 0 the average frame rate of the input stream will be used
        ffmpeg_args (Optional[List[str]]): additional ffmpeg args like `-preset veryfast` for libx264
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

    Args:
        torchserve_url (str): URL to the torchserve entry point e.g. http://ip:port/predictions/modelname
        labels (Dict[int,str]): label list to decode the dection ids into class names
        sampling_rate (int): frequence how often in a second detections from TorchServe will be requested
        max_recording_time (int): maximum length in seconds a recording can have
        debug (int): flag to signal debugging mode, what will result in all detections will be visible in a frame
        trigger_save (str): if enabled, the initial trigger frame and detections will be saved to output folder
        trigger_path (str): file path were the trigger image will be saved
        trigger_detection_threshold (int): minimum score a detection should have to be valid
        trigger_invalidate_duration (int): number of seconds without a trigger detection until the recording stops
        trigger_classes (List[str]): List of potential class names that triggers an recording event
    """

    torchserve_url: str
    labels: Dict[int, str]
    sampling_rate: int = 2
    max_recording_time: int = 5 * 60
    debug: bool = False
    trigger_save: bool = False
    trigger_path: str = "%Y-%m-%d-%H:%M:%S.jpg"
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

        Retruns:
            Config: application configuration
        """
        return self._config
