from typing import Any, List
import cv2
import insightface
from cyuai.processors.face_analyser import get_one_face, find_similar_face
from cyuai.processors.face_reference import clear_face_reference, get_face_reference, set_face_reference
from cyuai.typing import Face, Frame
from cyuai.utilities import conditional_download, resolve_relative_path, is_image
from cyuai.processors import core

FACE_SWAPPER = None
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = resolve_relative_path('../models/inswapper_128.onnx')
        print(model_path)
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path,
                         ['https://huggingface.co/ashleykleynhans/inswapper/tree/main/inswapper_128.onnx'])
    return True


def pre_start(source_path, target_path) -> bool:
    if not is_image(source_path):
        print('输入脸对象必须是一个图片')
        return False
    elif not get_one_face(cv2.imread(source_path)):
        print('输入脸图片不存在人脸.')
        return False
    if not is_image(target_path):
        print('换脸对象必须是一个图片')
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    target_face = find_similar_face(temp_frame, reference_face)
    if target_face:
        temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = get_one_face(target_frame, 0)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_frames(source_path: str, temp_frame_paths: List[str]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        print(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[0])
        reference_face = get_one_face(reference_frame, 0)
        set_face_reference(reference_face)
    core.process_video(source_path, temp_frame_paths, process_frames)
