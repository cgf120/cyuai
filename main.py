import time
import cv2
import numpy as np
from flask import Flask, request
from cyuai import core
from flask_loguru import logger

app = Flask(__name__)

core.init()
core.pre_check()


@app.route('/has_face', methods=['POST'])
def process_image():
    logger.info(f'接口开始时间：{time.time()}')
    result = None
    # 检查请求中是否有文件
    file = request.files['image']
    if file and file.mimetype and file.mimetype.startswith('image/'):
        # 检查文件名是否为空
        if file.filename == '':
            result = '文件名为空'
        else:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            result = {'has_face': core.has_face(img)}
    else:
        result = '文件格式错误'
    logger.info(f'接口结束时间：{time.time()}')
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
