import shutil
import sys
from cyuai.config import frame_processors
from cyuai.processors.core import get_frame_processors_modules


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        print('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        print('ffmpeg is not installed.')
        return False
    return True


def init() -> bool:
    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_check():
            return False
    return True


def swap_face(source_image, target_image, result_image):
    shutil.copy2(target_image, result_image)
    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_start(source_image, target_image):
            return False
        frame_processor.process_image(source_image, result_image, result_image)
