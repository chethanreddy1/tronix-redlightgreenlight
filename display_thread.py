import cv2
import numpy as np
import threading
import time

import socket
import struct
import pickle


from display import Video,People
DOLL_VIDEO_PATH='C:\\Users\\sujal\\Downloads\\Untitled video - Made with Clipchamp (1).mp4'
IMAGE_FOLDER_PATH="C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\images"

def display_thread(vid_obj,people_obj):
    while True:
        vid_obj.display_doll()
        vid_obj.hightlight_on_webcam(people_obj.curr_keypoints_cam1)
        vid_obj.player_status=people_obj.l
        vid_obj.show_player_status()

        cv2.imshow('Frame',cv2.resize(vid_obj.display_frame,(1080,720)))
        if cv2.waitKey(1)==ord('q'):
            break
        print(people_obj.l)

def back_camera_thread(vid_obj):
    print("back thread")
    
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.0.101'  # Use the server's IP address
    port = 8088 # Use the same port as on the server

    client_socket.connect((host_ip, port))

    data = b""
    payload_size = struct.calcsize("<Q")

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)  # Adjust the buffer size as needed
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("<Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)  # Adjust the buffer size as needed
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        # print(time.time())
        vid_obj.change_back_frame(frame)
        # cv2.imshow('Frame3',vid_obj.frame_b)
        # if cv2.waitKey(1) == ord('q'):
        #     break

def front_thread(v_obj):
    print("front thread running")
    cam_obj = cv2.VideoCapture(0)
    while True:
        ret , frame = cam_obj.read()
        v_obj.change_front_frame(frame)
        if not ret:
            break
        # cv2.imshow('Frame2',v_obj.frame_f)
        # if cv2.waitKey(1)==ord('q'):
        #     break



def testing_thread(vid_obj,people_obj):
    for i in range(1000):
        if(i==100):
            vid_obj.mode='r'
            people_obj.l=[1,0,1,1,1]
            time.sleep(3)
        if(i==200):
            vid_obj.mode='g'
            people_obj.l=[0,0,1,1,1]
            time.sleep(3)
        
        if(i==300):
            vid_obj.mode='r'
            people_obj.l=[0,0,1,1,1]
            time.sleep(3)
            people_obj.change_keypoints(changed_keypoints_1,changed_keypoints_2)
            
        if(i==400):
            vid_obj.mode='g'
            people_obj.l=[0,0,0,1,1]
            time.sleep(3)
        if(i==500):
            vid_obj.mode='r'
            people_obj.l=[1,0,1,1,1]
            time.sleep(3)
        if(i==600):
            vid_obj.mode='g'
            people_obj.l=[0,0,0,1,1]
            time.sleep(3)
        
        

per_keypoints=np.array([[     462.82,      834.28,     0.94828],
       [     458.88,      835.75,     0.92352],
       [     458.27,      831.67,     0.90224],
       [     458.33,      834.18,     0.81058],
       [     455.66,      821.08,        0.89],
       [     477.73,      818.68,     0.95413],
       [      479.1,      837.37,     0.94124],
       [     476.44,      800.54,     0.93741],
       [     510.04,      842.17,      0.9098],
       [     499.46,       781.8,     0.86429],
       [     537.27,      846.97,     0.89841],
       [     512.78,      773.41,     0.54808],
       [     539.43,      828.61,     0.85455],
       [     539.03,      807.71,     0.88112],
       [     539.52,      818.33,     0.90238],
       [     584.81,       827.9,     0.87716],
       [     579.78,      812.18,     0.91905],
       [     630.71,       822.3,     0.88984],
       [     611.75,      802.68,     0.86489],
       [     638.42,      828.67,     0.87141],
       [     636.16,      831.58,      0.9184],
       [     635.47,      819.23,     0.89662],
       [     630.58,      802.91,     0.98351],
       [     628.63,       797.3,     0.86932],
       [     614.01,      801.87,     0.70369]], dtype=np.float32)
per_keypoints=per_keypoints/2
per_keypoints_2=per_keypoints+10
per_keypoints_3=per_keypoints+30
per_keypoints_4=per_keypoints+50
per_keypoints_5=per_keypoints+100

keypoints_1={0:per_keypoints,1:per_keypoints_2,2:per_keypoints_3,3:per_keypoints_4,4:per_keypoints_5}
keypoints_2=keypoints_1.copy()
changed_keypoints_1={0:per_keypoints+50,1:per_keypoints_2+50,2:per_keypoints_3+50,3:per_keypoints_4+50,4:per_keypoints_5+50}
changed_keypoints_2=changed_keypoints_1.copy()

vid_obj=Video(DOLL_VIDEO_PATH,IMAGE_FOLDER_PATH)

people_obj=People()
people_obj.change_keypoints(keypoints_1,keypoints_2)
t1=threading.Thread(target=display_thread,args=(vid_obj,people_obj))
t1.start()
t2=threading.Thread(target=testing_thread,args=(vid_obj,people_obj))
t2.start()
t3=threading.Thread(target=front_thread,args=(vid_obj,))
t3.start()
t4=threading.Thread(target=back_camera_thread,args=(vid_obj,))




t4.start()

# back_camera_thread(vid_obj)
