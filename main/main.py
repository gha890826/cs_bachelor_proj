from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
import threading
import numpy as np
# from threading import Thread, enumerate
from queue import Queue

'''
stringdoc for main.py.
main.py start ->deal with args -> make network -> get a frame from get_frame() -> throw frame to detector() and get results list
-> throw results list to make_result() to cut target out and save -> call show_result() to show cuted out result
get_frame() - return frame pos can read
'''

default_weight_path="yolov_last.weights"
default_config_file_path="./datas/yolov4.cfg"
default_data_file="./datas/obj.data"
default_camera="1"
threads=[]


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=default_camera,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("-w","--weights", default=default_weight_path,
                        help="yolo weights path")
    parser.add_argument("-ds","--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--no_ext_output", action='store_false',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default=default_config_file_path,
                        help="path to config file")
    parser.add_argument("--data_file", default=default_data_file,
                        help="path to data file")
    parser.add_argument("-th","--thresh", type=float, default=.35,
                        help="remove detections with confidence below this value")
    parser.add_argument("-ne","--no_expand", action='store_false',
                        help="display expand windows show the detected objects")
    return parser.parse_args()
    

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("not ret")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.no_ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            ori_image=frame
            img_crop=darknet.crop_boxes(detections_adjusted, frame, class_colors)
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            # print(img_crop)
            if not args.dont_show:
                cv2.imshow('Inference', image)
                # for show_image in img_crop:
                #     cv2.imshow("str(i)", show_image)
                # for i in range(0,len(img_crop)):
                #     cv2.imshow(str(i), image[i])
                if len(img_crop)!=0:
                    Verti = np.vstack(img_crop)
                    cv2.imshow("cropped",Verti)
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    while cap.isOpened():
        cap.release()
        video.release()
        cv2.destroyAllWindows()
    exit(0)


def detector(frame):
    pass


def set_video(file_pos):
    __video_pos=file_pos


def make_result(frame,results):
    pass


def show_result():
    pass


if __name__=='__main__':
    print("main.py start")
    #os.chdir('./darknet')
    #os.system("python ./darknet_images.py --input ./data/dog.jpg")
    
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    #process args
    args = parser()
    check_arguments_errors(args)

    #load network
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    print("network load success")

    #set the input stream camera/video
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("input set success")

    #get frames
    threads.append(threading.Thread(target=video_capture, args=(frame_queue, darknet_image_queue)))
    threads[0].daemon = True

    #detect
    threads.append(threading.Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)))

    #drowing and show
    threads.append(threading.Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)))
    threads[2].daemon = True

    for t in threads:
        t.start()