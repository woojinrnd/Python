# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from ast import keyword
from collections import deque
from logging import Logger, LogRecord
from pathlib import Path
from pickle import APPEND
from re import A
from tabnanny import filename_only
#import time #kwj
from time import time
from tracemalloc import start
from urllib.request import AbstractDigestAuthHandler

import cv2  # kwj
import numpy as np
import torch
import torch.backends.cudnn as cudnn

#import keyboard
from gui_buttons import Buttons

#Initialize Buttons 
button = Buttons()
button.add_button("wound", 400, 20)
button.add_button("dot", 400, 80)

colors = button.colors


global im0
wound_que = deque(maxlen = 4)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# Button
def click_button(event, x, y, flags, params):
    global button_wound
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)
        #print(x,y)
    # global button_dot
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     button.button_click(x, y)
        
        
@torch.no_grad()
def run(
        #weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        weights=ROOT / 'best.pt',  # model.pt path(s)
        #source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        source = cv2.CAP_DSHOW + 0,
        #data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        data=ROOT / 'data/data.yaml',
        #imgsz=(640, 640),  # inference size (height, width)
        imgsz = (416, 416),
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=7,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inferenceqq
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size



    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        start_time = time() # kwj_fps
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0 #정규화
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        active_buttons = button.active_buttons_list() #Button

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                false_frame = [dataset.count] # kwj
                avg_frame = int(sum(false_frame) / len(false_frame)) # kwj
                # print(false_frame)
                # print(avg_frame)
                p, im0, frame = path[i], im0s[i].copy(), avg_frame #kwj
                # p, im0, frame = path[i], im0s[i].copy(), dataset.count # Origin
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 1)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string #창의 크기 출력
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            cv2.putText(im0, str(frame), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2) # 몇 번째 frame인지(frame)

            end_time = time() # kwj_fps
            fps = 1 / np.round(end_time - start_time, 2) #kwj_fps
            cv2.putText(im0, f'FPS: {int(fps)}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) #kwj_fps



            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n >= 0)}, "  # add to string
                    


                # Get active buttons list
                #active_buttons = button.active_buttons_list()
                #print("Active buttons", active_buttons)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        #label = None if hiqde_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') 
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                        
                        # fps = 1 / np.round(end_time - start_time, 2) #kwj_fps
                        # cv2.putText(im0, f'FPS: {int(fps)}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) #kwj_fps

                        # # node 프레임마다 감지 
                        # count = 0
                        # node_point = 5


                        

                        # if ((int(frame)+1) % node_point == 0 ):
                        #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 라벨링
                        #     annotator.box_label(xyxy, label, color=colors(c, True)) # 박스 그리기
                        #     LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') #kwj 몇개, 탐지 시간 출력
                        #     cv2.putText(im0, f'{s}Done',(150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # kwj_total_wound_per_frame
                        #     count += 1

                        #     #print("n_type :", n.dtype)
                        #     #print("n :", n)
                        #     node_frame_wound = n.tolist()
                        #     #print("node_frame_wound : ", type(node_frame_wound))
                            
                        #     # wound_que.append(node_frame_wound)

                        #     if (node_frame_wound == 0):
                        #         wound_que.append[(0)]
                        #     else:
                        #         wound_que.append(node_frame_wound)


                        #     print("---------길이------------ length of list :", len(wound_que))
                        #     print("frame_wound : ", wound_que)
                            

                        # # [5frame_wound, 10frame_wound, 15frame_wound 20frame_wound]
                        # if ((int(frame)+1) % 20 == 0):
                            
                        #     wound_per_20frame = sum(wound_que)
                        #     print("-------20프레임합계----------- 20frame_sum_total :", wound_per_20frame)
                        #     cv2.putText(im0, f'20frame_wound : {int(wound_per_20frame)}', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,200), 2) # kwj_sum_wound for 20frame
                        #     #print(type(frame_wound))
                        #     if (len(wound_que) != 0):
                        #         wound_que.clear() # que초기화
                        #     print("------초기화------reset_wound: ", wound_que)


                        #     #품질
                        #     if (wound_per_20frame <= 3):
                        #         print("-------------상------------- 총 개수 : ", wound_per_20frame)
                        #         cv2.putText(im0, f'BEST', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 상

                        #     elif (wound_per_20frame <= 5):
                        #         print("-------------중------------- 총 개수 : ", wound_per_20frame)
                        #         cv2.putText(im0, f'BETTER', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 중
                            
                        #     else:
                        #         print("-------------하-------------총 개수 : ", wound_per_20frame)
                        #         cv2.putText(im0, f'GOOD', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 하



                            
                            #0914
                            # print("n_type :", n.dtype)
                            # print("n :", n)

                            # n_kwj = n.tolist()

                            # print("n_kwj_type : ", type(n_kwj))
                            # print("n_kwj : ", n_kwj)

                            # frame_wound = [n_kwj]
                            # print("frame_wound_1 : ", type(frame_wound))
                            # #frame_wound.append(n_kwj)
                            # #print("frame_wound_2 : ", type(frame_wound))
                            # print("frame_wound : ", frame_wound)
                            # # print(len(frame_wound))
                            
                            

                            #total_wound = [n_kwj] #kwj_total wound

                            # if (count % 5 == 0):
                            #     total_wound.append(n_kwj(0))
                            #     print(total_wound)
                                
                            # elif (count % 10 == 0):
                            #     total_wound.append(n_kwj(0))
                                
                            # total_wound.append(total_wound(0))
                            #real_total = sum(total_wound)
                            #print(real_total)
                            # LOGGER.info(f'{int(real_total)}')
                        # Button
                        if names[c] in active_buttons:

                            
                            #kwj_origin
                            # #c = int(cls)  # integer class
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 라벨링
                            # annotator.box_label(xyxy, label, color=colors(c, True)) # 박스 그리기
                            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') #kwj
                            # cv2.putText(im0, f'{s}Done',(150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # kwj_total_wound_per_frame

                            # 20 frame 의 사진 저장
                            # count = 0 
                            # if (int(frame) % 20 == 0):
                            #     print('Saved frame number : ' + str(int(frame)))
                            #     cv2.imwrite("images/20frame/frame%d.jpg" % count, im0)
                            #     print('Saved frame%d.jpg' % count)
                            #     count += 1


                            # node 프레임마다 감지 
                            count = 0
                            node_point = 5
                            total_frame = 20
                            
                            


                            # #kwj_1212
                            # frame_box = []
                        
                            # frame_box.append(int(frame))
                            # current_frame = frame_box[0]
                            # print(frame_box)
                            # print("current frame : ", current_frame)
                            
                        

                            # if ((int(frame)+1) % node_point == 0 ):
                            if ((int(frame)+1) % node_point == 0 ):
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 라벨링
                                annotator.box_label(xyxy, label, color=colors(c, True)) # 박스 그리기
                                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') #kwj 몇개, 탐지 시간 출력
                                cv2.putText(im0, f'{s}Done',(150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # kwj_total_wound_per_frame
                                count = count + 1

                                # print("n_type :", n.dtype)
                                # print("n :", n)
                                node_frame_wound = n.tolist()
                                # print("node_frame_wound : ", type(node_frame_wound))
                                
                                # wound_que.append(node_frame_wound)

                                if (node_frame_wound == 0):
                                    wound_que.append[(0)]
                                else:
                                    wound_que.append(node_frame_wound)


                                print("---------길이------------ length of list :", len(wound_que))
                                print("frame_wound : ", wound_que)
                                

                            # [5frame_wound, 10frame_wound, 15frame_wound 20frame_wound]
                            if ((int(frame)+1) % total_frame == 0):
                                
                                wound_per_20frame = sum(wound_que)
                                print("-------20프레임합계----------- 20frame_sum_total :", wound_per_20frame)
                                cv2.putText(im0, f'20frame_wound : {int(wound_per_20frame)}', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,200), 2) # kwj_sum_wound for 20frame
                                #print(type(frame_wound))
                                wound_que.clear() # que초기화


                                # if (len(wound_que) != 0):
                                    # wound_que.clear() # que초기화
                                    # wound_que = deque(maxlen = 4)
                                print("------초기화------reset_wound: ", wound_que)

                                
                                #품질
                                if (wound_per_20frame <= 10):
                                    print("-------------상------------- 총 개수 : ", wound_per_20frame)
                                    # 일정 프레임동안 화면상에 표시하고 싶음
                                    cv2.putText(im0, f'BEST', (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4) # 상

                                elif (wound_per_20frame <= 5):
                                    print("-------------중------------- 총 개수 : ", wound_per_20frame)
                                    cv2.putText(im0, f'BETTER', (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4) # 중
                                
                                else:
                                    print("-------------하-------------총 개수 : ", wound_per_20frame)
                                    cv2.putText(im0, f'GOOD', (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4) # 하


                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    
                        
            
            #Stream results
            im0 = annotator.result()
            if view_img:
                #cv2.imshow(str(p), im0)

                # # Get active buttons list
                # active_buttons = button.active_buttons_list()
                # print("Active buttons", active_buttons)

                
                
                # Display buttons
                button.display_buttons(im0)
                cv2.namedWindow("woojin")
                cv2.setMouseCallback("woojin",click_button)
                cv2.imshow("woojin",im0)

                cv2.waitKey(1)  # 1 millisecond
                

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            #fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            fps = frame
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = fps, im0.shape[1], im0.shape[0]  #kwj_fps_save
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


        # Print time (inference-only)
        print("frame : " , frame)
        print("Active buttons", active_buttons) 
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
               
               
    
        
        

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning).


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    #parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default= cv2.CAP_DSHOW + 0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=7, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

a = 'finish' # 마지막 출력 문장 여기서 조정

def main(opt):
    #print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    return a


#실행을 담당 # 마우스 눌렀을 때의 이벤트 추가 (ex. detect 껐다 켜기)
if __name__ == "__main__": 
    opt = parse_opt()
    _main = main(opt)
    print(_main)




# 키보드 / 마우스를 눌렀을 때의 detecting 껐다가 키기
# 그러기 위해서는 model(video)의 방법에 대해서 알고 있어야 함
# model()의 인자? / 인자들은 어떻게 설정하나?


    


    # def mouse_click(event):
    #     if event == cv2.EVENT_FLAG_LBUTTON:
    #         a = 1
    #         hide_labels = True
    #         hide_conf = True
    #     elif event == cv2.EVENT_FLAG_RBUTTON:
    #         a = 0
    #         hide_labels = False
    #         hide_conf = False
    # cv2.setMouseCallback('0', mouse_click, run.im0)      

#예시
# while True:
#     if keyboard.is_pressed("1"):
#         print("hello")
#         break


#kwj
# CAMERA_ID = 0

# capture = cv2.VideoCapture(CAMERA_ID)
# if capture.isOpened() == False: # 카메라 정상상태 확인
#     print(f'Can\'t open the Camera({CAMERA_ID})')
#     exit()

# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
# capture.set(cv2.CAP_PROP_FPS, 30)

# prev_time = 0
# total_frames = 0
# start_time = time.time()
# while cv2.waitKey(1) < 0:
#     curr_time = time.time()

#     ret, frame = capture.read()
#     total_frames = total_frames + 1

#     term = curr_time - prev_time
#     fps = 1 / term
#     prev_time = curr_time
#     fps_string = f'term = {term:.3f},  FPS = {fps:.2f}'
#     print(fps_string)

#     cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
#     cv2.imshow("VideoFrame", frame)

# end_time = time.time()
# fps = total_frames / (start_time - end_time)
# print(f'total_frames = {tocd..tal_frames},  avg FPS = {fps:.2f}')

# capture.release()
# cv2.destroyAllWindows()



# def mouse_click(event):
#     if event == cv2.EVENT_FLAG_LBUTTON:
#         hide_labels = True
#         hide_conf = True
#     elif event == cv2.EVENT_FLAG_RBUTTON:
#         hide_labels = False
#         hide_conf = False
#cv2.setMouseCallback('0', mouse_click, im0)

#Stream results # 캡쳐화면 q 
            # im0 = annotator.result()
            # if view_img:
            #     while(True):
            #         cv2.imshow(str(p), im0)
            #         cv2.putText(im0, str(frame) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
            #         #cv2.waitKey(1)  # 1 millisecond
            #         if cv2.waitKey() == ord('q'):
            #             break
            #     cv2.destroyAllWindows() #kwj
