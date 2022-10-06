# from inspect import classify_class_attrs
# from re import I
# from unittest import result
# import cv2
# import torch
#import numpy as np
#from PIL import Image

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ모델 불러오기ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

## Load YOLOv5 with PyTorch Hub
## https://github.com/ultralytics/yolov5/issues/36
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\yolov5-master\kwj\dnn_model\best.pt', autoshape = False) 

#model = torch.hub.load(r'C:\yolov5-master', 'custom', path=r'C:\yolov5-master\kwj\dnn_model\best.pt', source='local')

## net = cv2.dnn.readNet(r'C:\yolov5-master\kwj\dnn_model\best.onnx')
## model = cv2.dnn_DetectionModel(net)
## model.setInputParams(size = (416, 416), scale = 1/255) # opencv : 0~255 -> 0~1 정규화

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ모델 불러오기ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


## 예시 (되는 거 확인)
## Images
# img1 = Image.open(r'C:\yolov5-master\kwj\apple1.jpg')  # PIL image
# img2 = cv2.imread(r'C:\yolov5-master\kwj\apple10.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
# imgs = [img1, img2]  # batch of images


# Inference
# results = model(imgs, size = 416)  # includes NMS

# #Results
# results.print()  
# results.save()  # or .show()
# results.show() 
# #cv2.imshow('img',results) #kwj

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ



#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ영상ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

####Intialize camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# if cap.isOpened():
#     while True:
#         # Get frames
#         ret, frame = cap.read()
#         frame = [frame]
#         model(frame)
#         # for i in frame:
#         #     frame1 = frame[i]
            
#         # Object Detection
#         # (class_ids, scores, bboxes) = model.detect(frame) # class(클래스) / score(신뢰도) / boundingboxes(좌표)
#         # print("class ids : ", class_ids)
#         # print("scores : ", scores)
#         # print("bboxes : ", bboxes)
        
#         cv2.imshow("KWJ", frame)
 
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
# cap.release()
# cv2.destroyAllWindows()


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ영상ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ




#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ테스트ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


# imagePath = r'C:\yolov5-master\kwj\apple2.png'
# img = cv2.imread(imagePath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# pred = model(img, size = 416)

# cv2.imshow("kwj", img)
# cv2.imshow("kwj2", gray)
# cv2.imshow("P1", pred)

# cv2.waitKey()
# cv2.destroyAllWindows()

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ테스트ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ



#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ테스트ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
#https://github.com/niconielsen32/ComputerVision/blob/master/deployYoloModel.py
##이걸로 하자
# from multiprocessing.dummy import active_children
import torch
import numpy as np
import cv2
from time import time
from gui_buttons import Buttons

#Initialize Buttons
button = Buttons()
button.add_button("wound", 400, 20)





class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index): # 카메라 / 모델 / 클래스 / cpu
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index # 카메라 인덱스
        self.model = self.load_model() # 모델 불러오기
        self.classes = self.model.names # 클래스 이름
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # cpu쓸거야
        print("Using Device: ", self.device)

    def get_video_capture(self): # 이미지 불러오기
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self): # 모델 불러오기
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load(r'C:\yolov5-master', 'custom', path=r'C:\yolov5-master\kwj\dnn_model\best.pt', source='local')
        return model

    def score_frame(self, frame):  # model 적용
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame] # frame을 list형태로 저장
        results = self.model(frame)  # model적용
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1] 
        return labels, cord

    def class_to_label(self, x): #라벨링
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame): # 라벨링 박스 그리기
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        active_buttons = button.active_buttons_list()

        #global i
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.7: # Confidence Score 신뢰도를 담당
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)

                if self.class_to_label in active_buttons:
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) # 박스 그리기
                    cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2) # class name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) # 박스 그리기
                    # detecting class : 정확도 출력
                    print(self.class_to_label(labels[i]), ':', row[4])

                #print(row[4])

        return frame

    def __call__(self): # 카메라 키고, fps출력
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        

        cap = self.get_video_capture()
        assert cap.isOpened()
        
        # 카메라 속성
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        cap.set(cv2.CAP_PROP_FPS, 15)

        


        # Mouse Event
        def click_button(event, x, y, flags, params):
            # global button_wound
            if event == cv2.EVENT_LBUTTONDBLCLK:
                button.button_click(x, y)
                # print(x, y)
                # polygon = np.array([[(540, 20), (620, 20), (620, 60), (540, 60)]])

                # is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
                # if is_inside > 0:
                #     print("Click")
                #     cv2.fillPoly(frame, polygon, (200, 0, 0))

                #     if button_wound is False:
                #         button_wound = True
                #     else:
                #         button_wound = False

                #     print("Now button wound is : ", button_wound)
                    




        # create Window
        cv2.namedWindow("woojin")
        cv2.setMouseCallback("woojin", click_button)


      # Get Frame
        while True:
          
            start_time = time()
            
            # Get active buttons list
            active_buttons = button.active_buttons_list()
            print("Active buttons", active_buttons)

            
            ret, frame = cap.read()
            assert ret
            
            results = self.score_frame(frame) # result =  모델 적용
            frame = self.plot_boxes(results, frame) # frame = 라벨링 박스
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            
            # #Print Result
            # labels, cord = results
            # n = len(labels)
            # x_shape, y_shape = frame.shape[1], frame.shape[0]
            

            
            #Create Button
            #cv2.rectangle(frame, (540,20), (620, 60), (0,0,200), -1)
            # polygon = np.array([[(540, 20), (620, 20), (620, 60), (540, 60)]])
            # cv2.fillPoly(frame, polygon, (0, 0, 200))
            # cv2.putText(frame, "Swith", (550,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))

            # Display buttons
            button.display_buttons(frame)
            
            cv2.imshow('woojin', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
      
        cap.release()
        
        
#Create a new object and execute.
detector = ObjectDetection(capture_index=0)
detector()

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ테스트ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# import torch
# import numpy as np
# import cv2
# #import pafy
# import time


# class ObjectDetection:
#     """
#     Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
#     """
    
#     def __init__(self):
#         """
#         Initializes the class with youtube url and output file.
#         :param url: Has to be as youtube URL,on which prediction is made.
#         :param out_file: A valid output file name.
#         """
#         self.model = self.load_model()
#         self.classes = self.model.names
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("\n\nDevice Used:",self.device)



#     def load_model(self):
#         """
#         Loads Yolo5 model from pytorch hub.
#         :return: Trained Pytorch model.
#         """
#         model = torch.hub.load(r'C:\yolov5-master', 'custom', path=r'C:\yolov5-master\kwj\dnn_model\best.pt', source='local')
#         return model


#     def score_frame(self, frame):
#         """
#         Takes a single frame as input, and scores the frame using yolo5 model.
#         :param frame: input frame in numpy/list/tuple format.
#         :return: Labels and Coordinates of objects detected by model in the frame.
#         """
#         self.model.to(self.device)
#         frame = [frame]
#         results = self.model(frame)
     
#         labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#         return labels, cord


#     def class_to_label(self, x):
#         """
#         For a given label value, return corresponding string label.
#         :param x: numeric label
#         :return: corresponding string label
#         """
#         return self.classes[int(x)]


#     def plot_boxes(self, results, frame):
#         """
#         Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
#         :param results: contains labels and coordinates predicted by model on the given frame.
#         :param frame: Frame which has been scored.
#         :return: Frame with bounding boxes and labels ploted on it.
#         """
#         labels, cord = results
#         n = len(labels)
#         x_shape, y_shape = frame.shape[1], frame.shape[0]
#         for i in range(n):
#             row = cord[i]
#             if row[4] >= 0.7: #신뢰도
#                 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
#                 bgr = (0, 255, 0)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#                 cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

#         return frame


#     def __call__(self):
#         """
#         This function is called when class is executed, it runs the loop to read the video frame by frame,
#         and write the output into a new file.
#         :return: void
#         """
#         cap = cv2.VideoCapture(0)

#         while cap.isOpened():
            
#             start_time = time.perf_counter()
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             results = self.score_frame(frame)
#             frame = self.plot_boxes(results, frame)
#             end_time = time.perf_counter()
#             fps = 1 / np.round(end_time - start_time, 3)
#             cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
#             cv2.imshow("img", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break


# # Create a new object and execute.
# detection = ObjectDetection()
# detection()
