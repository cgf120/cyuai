import time
from typing import Any, List
import cv2
from gfpgan.utils import GFPGANer
from cyuai.processors import core
from cyuai.processors.face_analyser import get_many_faces
from cyuai.typing import Frame, Face
from cyuai.utilities import conditional_download, resolve_relative_path, is_image

FACE_ENHANCER = None


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    if FACE_ENHANCER is None:
        model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
        FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1)
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/GFPGANv1.4.pth'])
    return True


def pre_start(source_path, target_path) -> bool:
    if not is_image(target_path):
        print('换脸对象必须是一个图片.')
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        _, _, temp_face = get_face_enhancer().enhance(
            temp_face,
            paste_back=True
        )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame


def process_frame(temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(target_frame)
    cv2.imwrite(output_path, result)


def process_frames(source_path: str, temp_frame_paths: List[str]) -> None:
    for temp_frame_path in temp_frame_paths:
        start = time.time()
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(temp_frame)
        cv2.imwrite(temp_frame_path, result)
        print(f"执行时间{time.time() - start}")


def process_video(temp_frame_paths: List[str]) -> None:
    core.process_video(None, temp_frame_paths, process_frames)
