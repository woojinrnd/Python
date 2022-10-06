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
button.add_button("wound", 200, 10)
button.add_button("Dot", 200, 40)



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
      
        return cv2.VideoCapture(cv2.CAP_DSHOW + self.capture_index)

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
        labels= results.xyxyn[0][:, -1] # 라벨 (label name)
        cord = results.xyxyn[0][:, :-1] # 정확도 (confidence)
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
            if row[4] >= 0.6: # Confidence Score 신뢰도를 담당
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) #왼쪽 꼭지점  # 오른쪽 꼭지점
                bgr = (0, 255, 0)



                if self.class_to_label(labels[i]) in active_buttons: #Detecting Switch
                    cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2) # class name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) # 박스 그리기 # (왼쪽 꼭지점), (오른쪽 꼭지점) 
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
        
        # Intialize Camera
        cap = self.get_video_capture()
        assert cap.isOpened()
        
        # 카메라 속성
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        cap.set(cv2.CAP_PROP_FPS, 15)
        

        # Mouse Event
        def click_button(event, x, y, flags, params):
            global button_wound
            if event == cv2.EVENT_LBUTTONDBLCLK:
                button.button_click(x, y)
                #print(x, y)
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