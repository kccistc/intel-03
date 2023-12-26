""" Smart Factory 를 실행시키기 위한 factory.py 파일  """
# !/usr/bin/env python3
# pylint: disable=no-member

import os
import sys
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
import openvino as ov

from iotdemo import ColorDetector, FactoryController, MotionDetector

# from openvino.inference_engine import IECore


def thread_cam1(q, force_stop):
    """ 불량을 검출할 thread  """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "detected")

    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model("resources/openvino.xml")

    # Load and initialize Video
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    start_flag = True
    while not force_stop:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))

        # Inference OpenVINO
        input_tensor = np.expand_dims(detected, 0)

        if start_flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)

            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

            ppp.input().preprocess()\
                .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)

            model = ppp.build()

            compiled_model = core.compile_model(model, "CPU")
            start_flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        if probs[0] > 0.0:
            print("Bad Item!")
        else:
            print("Good Item!")

        # Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q, force_stop):
    """ 색 구분을 위한 thread  """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "detected")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    video_file = "resources/conveyor.mp4"
    cap = cv2.VideoCapture(video_file)

    while not force_stop:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        info = ("VIDEO:Cam2 live", frame)
        q.put(info)

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == "blue" and ratio > .5:
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    """ 메인 UI  """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])

    cv2.imshow(title, frame)


def main():
    """ Factory.py 를 실행할 때 실행되는 Main 함수  """

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # Create a Queue
    q = Queue()

    force_stop = False

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q, force_stop))
    t2 = threading.Thread(target=thread_cam2, args=(q, force_stop))

    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not force_stop:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue.
            # You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = q.get_nowait()
            except Empty:
                continue

            # show videos with titles of
            # 'Cam1 live' and 'Cam2 live' respectively.
            name, data = event
            if name.startswith("VIDEO:"):
                imshow(name[6:], data)
            elif name == "PUSH":
                # Control actuator, name == 'PUSH'
                ctrl.push_actuator(data)
            elif name == 'DONE':
                force_stop = True

            q.task_done()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occured: {e}")
        os._exit()
