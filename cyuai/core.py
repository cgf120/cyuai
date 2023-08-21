import shutil
import sys
from cyuai.config import frame_processors
from cyuai.processors.core import get_frame_processors_modules
from cyuai.processors.face_analyser import get_one_face
from cyuai.typing import Frame
from cyuai.utilities import create_temp, extract_frames, detect_fps, get_temp_frame_paths, create_video, restore_audio, \
    clean_temp, move_temp, current_time


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


def swap_face_image(source_image, target_image, result_image):
    shutil.copy2(target_image, result_image)
    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_start(source_image, target_image):
            return False
        frame_processor.process_image(source_image, result_image, result_image)


def has_face(frame: Frame):
    return get_one_face(frame) is not None


def swap_face_video(source_image, target_video, result_video):
    print(f"创建临时目录:{current_time()}")
    create_temp(target_video)
    print(f"获取原视频的帧率:{current_time()}")
    fps = detect_fps(target_video)
    print(f"抽帧:{current_time()}")
    extract_frames(target_video, fps)
    print(f"读取抽帧图片:{current_time()}")
    temp_frame_paths = get_temp_frame_paths(target_video)

    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_start(source_image, target_video):
            return False
        frame_processor.process_video(source_image, target_video, result_video)
        if temp_frame_paths:
            print(f"帧处理:{frame_processor.__class__.__name__}开始：{current_time()}")
            frame_processor.process_video(source_image, temp_frame_paths)
        print(f"合成视频:{current_time()}")
        create_video(target_video)
        move_temp(target_video, result_video)
        print(f"处理音频:{current_time()}")
        restore_audio(target_video, result_video)
        print(f"清理临时文件:{current_time()}")
        clean_temp(target_video)
