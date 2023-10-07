import cv2
import numpy as np
import time



class Video:
    def __init__(self,doll_video_path,image_folder_path,webcam):
        self.display_frame=np.zeros((1080,1920,3),dtype=np.uint8)
        self.mode='g' 
        self.prev_mode=self.mode
        self.count_frames_of_doll=0
        self.list_of_frames_backward=[]
        self.list_of_frames_forward=[]
        self.DOLL_VIDEO_PATH=doll_video_path
        self.IMAGE_FOLDER_PATH=image_folder_path
        self.webcam_object=webcam
        self.player_status=[1,1,1,1,1] #1 indicates all players are alive
        self.alive1=cv2.imread(IMAGE_FOLDER_PATH+'\\alive'+str(1)+'.png')
        self.alive2=cv2.imread(IMAGE_FOLDER_PATH+'\\alive'+str(2)+'.png')
        self.alive3=cv2.imread(IMAGE_FOLDER_PATH+'\\alive'+str(3)+'.png')
        self.alive4=cv2.imread(IMAGE_FOLDER_PATH+'\\alive'+str(4)+'.png')
        self.alive5=cv2.imread(IMAGE_FOLDER_PATH+'\\alive'+str(5)+'.png')
        self.list_of_photos_alive=[self.alive1,self.alive2,self.alive3,self.alive4,self.alive5]
        self.dead1=cv2.imread(IMAGE_FOLDER_PATH+'\\DEAD'+str(1)+'.png')
        self.dead2=cv2.imread(IMAGE_FOLDER_PATH+'\\DEAD'+str(2)+'.png')
        self.dead3=cv2.imread(IMAGE_FOLDER_PATH+'\\DEAD'+str(3)+'.png')
        self.list_of_photos_dead=[self.dead1,self.dead2,self.dead3]

        #creating list of frames of doll video
        cap=cv2.VideoCapture(self.DOLL_VIDEO_PATH)
        while True:
            ret, frame = cap.read()
            if not ret: break # break if no next frame
            resized_frame= cv2.resize(frame, (640, 720))
            self.list_of_frames_backward.append(resized_frame) # append frame to the list of frames
        
        self.list_of_frames_forward=self.list_of_frames_backward.copy()
        self.list_of_frames_forward.reverse()

        #intialising the player bar



    def display_doll(self):
        if(self.prev_mode!=self.mode):
            self.count_frames_of_doll=len(self.list_of_frames_backward)-self.count_frames_of_doll-1
            self.prev_mode=self.mode
        if(self.mode=='g'):
            display_list=self.list_of_frames_backward

        else:
            display_list=self.list_of_frames_forward
        
        self.display_frame[360:,1280:,:]=display_list[self.count_frames_of_doll]
        if not self.count_frames_of_doll == (len(display_list)-5):
            self.count_frames_of_doll+=10

        return
    
    def show_player_status(self):
        left =0
        right =384
        for num,i in enumerate(self.player_status):
            if i==1:
                custom_image=self.list_of_photos_alive[num]
                
            else:
                custom_image=self.list_of_photos_dead[num]
                # print(np.shape(custom_image))
            custom_image = cv2.resize(custom_image, (384, 360))
            self.display_frame[0:360,left:right, :] = custom_image
            left=right
            right+=384

    def hightlight_on_webcam(self,keypoints):
        _,image=self.webcam_object.read()

        ALPHA=0.8 #level of transparency

        if(len(keypoints.items())>5):
            raise Exception("More than 5 persons in the frame")
        i=0
        for id,person_keypoints in keypoints.items():
            # the vertices of the polygon is stored in points
            list1=[3,1,2,4,7,9,11,13,16,18,17,15,12,10,8,6]
            points=[person_keypoints[i] for i in list1]
            points=np.array(points)
            # points=np.int32(points)  #converting points(type=list) to np.array of int32
            points=points[:,:2] #removing the accuracy column

            #the model returns y,x so exchange first and second column using temp

            temp=np.zeros_like(points)

            temp[:,0]=points[:,0]    #temp stores the first col
            points[:,0]=points[:,1]  #first col is made sec col
            points[:,1]=temp[:,0]    #first col is filled in sec col
            points_2=np.int32(points)

            #generating a mask
            mask = np.zeros(np.shape(image), dtype = np.uint8)
            cv2.fillPoly(mask,pts=[points_2],color=(255,255,255))
            mask=mask/255
            mask=mask*(ALPHA-1)+1
            #multiplying the mask to the original frame to reduce the transparency
            image=np.multiply(mask,image)
            #generating a second mask
            mask_2=np.zeros(np.shape(image),dtype=np.uint8)

            if(self.player_status[i]==1):
                colour=(0,255,0)  #1 represents the person is not yet out , so highlight with green
            else:
                colour=(0,0,255)  # highlight with red
            i+=1
            cv2.fillPoly(mask_2,pts=[points_2],color=colour)
            mask_2=mask_2*(1-ALPHA)
            image=image+mask_2

        self.display_frame[360:,:1280,:]=cv2.resize(image,(1280,720))





DOLL_VIDEO_PATH='C:\\Users\\sujal\\Downloads\\Untitled video - Made with Clipchamp (1).mp4'
IMAGE_FOLDER_PATH="C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\images"
cam = cv2.VideoCapture(0)
vid_obj=Video(DOLL_VIDEO_PATH,IMAGE_FOLDER_PATH,cam)

# keypoints=np.load("C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\testing\\d_video.npy",allow_pickle=True)
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

keypoints={0:per_keypoints,1:per_keypoints_2,2:per_keypoints_3,3:per_keypoints_4,4:per_keypoints_5}

i=0

while 500:
    t=time.time()
    vid_obj.display_doll()
    vid_obj.hightlight_on_webcam(keypoints)
    vid_obj.show_player_status()
    # vid_obj.show_player_status([0,0,0,0,0])
    cv2.imshow('Frame',cv2.resize(vid_obj.display_frame,(1080,720)))
    if cv2.waitKey(1)==ord('q'):
        break
    if(i==80):
        vid_obj.mode='r'
    if(i==100):
        vid_obj.player_status[2]=0
        vid_obj.mode='g'
    if(i==180):
        vid_obj.mode='r'
    if(i==200):
        vid_obj.player_status[1]=0
        vid_obj.mode='g'
    if(i==380):
        vid_obj.mode='r'
    if(i==400):
        vid_obj.player_status[0]=0
        vid_obj.mode='g'
    
    print(time.time()-t)
    i+=1
