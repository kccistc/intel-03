"""!/usr/bin/env python3"""

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
import openvino as ov

from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    """Thread for abnormal classification"""
    # MotionDetector
    det = MotionDetector()
    det.load_preset('./resources/motion.cfg')

    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model("./resources/openvino.xml")

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')
    
    start_flag = True
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))

        # abnormal detect
        # Inference OpenVINO
        input_tensor = np.expand_dims(detected, 0)

        if start_flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400
            ppp.input().preprocess() \
                .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)

            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            start_flag = False
            
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        # Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1
        if probs[0] > 0.0:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    """Thread for color classification"""
    # MotionDetector
    det = MotionDetector()
    det.load_preset('./resources/motion.cfg')

    # ColorDetector
    color = ColorDetector()
    color.load_preset('./resources/color.cfg')

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))   # 튜플로 만들어서 데이터 두개 넣기

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected))

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
    exit()


def imshow(title, frame, pos=None):
    """Function for image show"""
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """Main"""
    global FORCE_STOP

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

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue. Use exceptions for handle.
            # de-queue name and data
            try:
                event = q.get_nowait()
            except Empty:
                continue

            # show videos with titles of 'Cam1 live' and 'Cam2 live'
            name, data = event
            if name.startswith("VIDEO:"):
                imshow(name[6:], data)
            elif name == "PUSH":
                ctrl.push_actuator(data)

            # Control actuator, name == 'PUSH'
            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("EXCEPTION : KeyboardInterrupt")
    finally:
        os._exit()
