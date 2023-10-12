import numpy as np
import cv2
import socket
import pickle
import struct
import time
import threading


DOLL_VIDEO_PATH='C:\\Users\\sujal\\Downloads\\Untitled video - Made with Clipchamp (1).mp4'
IMAGE_FOLDER_PATH="C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\images"
UPFRONT2666_PATH="C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\upfront2666.mp4"

class People:
    def __init__(self,no_play):
        self.past_keypoints_cam1 = {}
        self.curr_keypoints_cam1 = {}
        self.past_keypoints_cam2 = {}
        self.curr_keypoints_cam2 = {}
        self.sort_curr_keypts_cam1={}
        self.sort_curr_keypts_cam2={}
        self.sort_prev_keypts_cam1={}
        self.sort_prev_keypts_cam2={}
        self.playerstatus=[1]*no_play
        self.no_play = no_play
        self.flag = 1                                #Flag if people are more than 
        self.er = 0.003
        self.begin1 = True
        self.begin2 = True
        self.imp1 = []
        self.imp2 = []
        self.s=[]
        

    def change_keypoints(self,keypoints1,keypoints2):
        if(self.begin1 and self.begin2):
            if(len(list(keypoints1.keys())) == self.no_play):
                self.curr_keypoints_cam1 = keypoints1
                self.past_keypoints_cam1 = keypoints1
                self.imp1=list(self.curr_keypoints_cam1.keys())
                self.sort_curr_keypts_cam1=dict([(i,self.curr_keypoints_cam1[self.imp1[i]]) for i in range(len(self.curr_keypoints_cam1))])
                self.sort_prev_keypts_cam1=self.sort_curr_keypts_cam1.copy()
                self.begin1=False
            else:
                self.begin1 = True
            if(len(list(keypoints2.keys())) == self.no_play):
                
                #keypoints2 are reversed

                mapping=list(keypoints2.keys())
                for i in range(int(len(mapping)/2)):
                    temp=keypoints2[mapping[i]]
                    keypoints2[mapping[i]]=keypoints2[mapping[len(mapping)-i-1]]
                    keypoints2[mapping[len(mapping)-i-1]]=temp

                self.curr_keypoints_cam2 = keypoints2
                self.past_keypoints_cam2 = keypoints2
                self.imp2=list(self.curr_keypoints_cam2.keys())
                self.sort_curr_keypts_cam2=dict([(i,self.curr_keypoints_cam2[self.imp2[i]]) for i in range(len(self.curr_keypoints_cam2))])
                self.sort_prev_keypts_cam2=self.sort_curr_keypts_cam2.copy()
                self.begin2=False
            else:
                self.begin2 = True

        #game began
        else:
            if(len(keypoints1) > self.no_play or len(keypoints2) > self.no_play):
                assert(0),'Greater than'+str(self.no_play)+ 'detected'
            
            # if(len(keypoints1)<self.no_play -1):
            #     assert(0),"Front Camera less than 3"

            #reversing keypoints2
            mapping=list(keypoints2.keys())
            for i in range(int(len(mapping)/2)):
                temp=keypoints2[mapping[i]]
                keypoints2[mapping[i]]=keypoints2[mapping[len(mapping)-i-1]]
                keypoints2[mapping[len(mapping)-i-1]]=temp

            

            self.past_keypoints_cam1 = self.curr_keypoints_cam1
            self.curr_keypoints_cam1 = keypoints1
            self.sort_prev_keypts_cam1=self.sort_curr_keypts_cam1

            self.past_keypoints_cam2 = self.curr_keypoints_cam2
            self.curr_keypoints_cam2 = keypoints2
            self.sort_prev_keypts_cam2=self.sort_curr_keypts_cam2
            

            keyps1 = list(self.curr_keypoints_cam1.keys())
            keyps2 = list(self.curr_keypoints_cam2.keys())
            past_keyps1 = list(self.past_keypoints_cam1.keys())
            past_keyps2 = list(self.past_keypoints_cam2.keys())

            if(len(self.curr_keypoints_cam1)<self.no_play):
                missing_in_cam1=list(set(self.imp1)-set(self.curr_keypoints_cam1))

                if(len(missing_in_cam1)==1):
                    self.curr_keypoints_cam1[missing_in_cam1[0]]=self.past_keypoints_cam1[missing_in_cam1[0]]
                elif len(missing_in_cam1)>1:
                    assert(0),">=2 ids changed at a single time in camera 1"
                
            if(len(self.curr_keypoints_cam2)<self.no_play):
                missing_in_cam2=list(set(self.imp2)-set(self.curr_keypoints_cam2))
                
                if(len(missing_in_cam2)==1):
                    self.curr_keypoints_cam2[missing_in_cam2[0]]=self.past_keypoints_cam2[missing_in_cam2[0]]
                elif len(missing_in_cam2)>1:
                    assert(0),">= 2 ids changed at a single time in camera 2"

            if(len(self.curr_keypoints_cam1)==self.no_play):
                if(set(self.imp1)!=set(self.curr_keypoints_cam1.keys())):
                    
                    added_in_cam1=list(set(self.curr_keypoints_cam1)-set(self.imp1))
                    missing_in_cam1=list(set(self.imp1)-set(self.curr_keypoints_cam1))
                    print(added_in_cam1)
                    print(missing_in_cam1)
                    if(len(added_in_cam1) !=1 and len(missing_in_cam1)!=1):
                        assert(0),"no of people added and missing are not equal and greater than 1 in camera 1"

                    index_of_missing=self.imp1.index(missing_in_cam1[0])
                    self.imp1[index_of_missing]=added_in_cam1[0]

            if(len(self.curr_keypoints_cam2)==self.no_play):
                if(set(self.imp2)!=set(self.curr_keypoints_cam2.keys())):
                    added_in_cam2=list(set(self.curr_keypoints_cam2)-set(self.imp2))
                    missing_in_cam2=list(set(self.imp2)-set(self.curr_keypoints_cam2))

                    if(len(added_in_cam2) !=1 and len(missing_in_cam2)!=1):
                        assert(0),"no of people added and missing are not equal and greater than 1 in camera 2"

                    index_of_missing=self.imp2.index(missing_in_cam2[0])
                    self.imp2[index_of_missing]=added_in_cam2[0]

            self.sort_curr_keypts_cam1=dict([(i,self.curr_keypoints_cam1[self.imp1[i]]) for i in range(len(self.curr_keypoints_cam1))])

            self.sort_curr_keypts_cam2=dict([(i,self.curr_keypoints_cam2[self.imp2[i]]) for i in range(len(self.curr_keypoints_cam2))])

            list_of_keys_cam1=list(self.sort_curr_keypts_cam1.values())
            list_of_keys_cam1.sort(key=lambda x:np.mean(x[:,1:2]))

            self.sort_curr_keypts_cam1=dict([(i,list_of_keys_cam1[i]) for i in range(len(list_of_keys_cam1))])

            list_of_keys_cam2=list(self.sort_curr_keypts_cam2.values())
            list_of_keys_cam2.sort(key=lambda x:np.mean(x[:,1:2]),reverse=True)

            self.sort_curr_keypts_cam2=dict([(i,list_of_keys_cam2[i]) for i in range(len(list_of_keys_cam2))])
    
    def eliminate(self,th):

        if len(self.sort_curr_keypts_cam1)==5 and len(self.sort_curr_keypts_cam2)==5 and len(self.sort_prev_keypts_cam1)==5 and len(self.sort_prev_keypts_cam2)==5:
            def calculate_cosine_similarity(dictpr1,dictpr2, dictpa1,dictpa2,confidence_threshold=0.6):
                sim=[]
                for person_id in dictpa1:
                    if (np.sum(self.sort_curr_keypts_cam1[person_id][:,2])>=np.sum(self.sort_curr_keypts_cam2[person_id][:,2])):
                        dict1=dictpa1
                        dict2=dictpr1
                        
                    else:
                        dict1=dictpa2
                        dict2=dictpr2

                    pose1 = dict1[person_id]
                    pose2 = dict2[person_id]

                    if pose1.shape != pose2.shape:
                        return None 

                    
                    coords1 = pose1[:, :2] 
                    coords2 = pose2[:, :2] 

                    confidence1 = pose1[:, 2]
                    confidence2 = pose2[:, 2]

                    # Check if both keypoints have confidence > confidence_threshold
                    valid_keypoints = (confidence1 > confidence_threshold) & (confidence2 > confidence_threshold)

                    if not np.any(valid_keypoints):
                        # If no valid keypoints, similarity is 1
                        sim.append(1.0)
                    else:
                        # Extract and normalize only the valid keypoints
                        normalized_coords1 = (coords1[valid_keypoints] - np.min(coords1[valid_keypoints], axis=0)) / (np.max(coords1[valid_keypoints], axis=0) - np.min(coords1[valid_keypoints], axis=0))
                        normalized_coords2 = (coords2[valid_keypoints] - np.min(coords1[valid_keypoints], axis=0)) / (np.max(coords1[valid_keypoints], axis=0) - np.min(coords1[valid_keypoints], axis=0))

                        # Calculate the cosine similarity between the normalized coordinate arrays
                        dot_product = np.sum(normalized_coords1 * normalized_coords2)
                        norm_coords1 = np.linalg.norm(normalized_coords1)
                        norm_coords2 = np.linalg.norm(normalized_coords2)

                        similarity = dot_product / (norm_coords1 * norm_coords2)

                        sim.append(1 - similarity)
                return [100.0 if s == 1 else s * 100 for s in sim]
        
            self.s=calculate_cosine_similarity(      self.sort_curr_keypts_cam1,
                                                self.sort_curr_keypts_cam2,
                                                self.sort_prev_keypts_cam1,
                                                self.sort_prev_keypts_cam2,)
            
            # new= ((np.array(s)<th)*1)*(np.array(self.playerstatus))
            new= ((np.array(self.s)<th)*1)
            self.playerstatus=list(new)
    
class Video:
    def __init__(self,doll_video_path,image_folder_path):
        self.display_frame=np.zeros((1080,1920,3),dtype=np.uint8)
        self.frame_f=np.zeros((480,640,3),dtype=np.uint8)
        self.frame_b=np.zeros((480,640,3),dtype=np.uint8)
        self.mode='g';
        self.prev_mode=self.mode
        self.count_frames_of_doll=0
        self.list_of_frames_backward=[]
        self.list_of_frames_forward=[]
        self.DOLL_VIDEO_PATH=doll_video_path
        self.IMAGE_FOLDER_PATH=image_folder_path
        self.player_status=[1,1,1,1,1] #1 indicates all players are alive
        self.alive1=cv2.imread(self.IMAGE_FOLDER_PATH+'\\alive'+str(1)+'.png')
        self.alive2=cv2.imread(self.IMAGE_FOLDER_PATH+'\\alive'+str(2)+'.png')
        self.alive3=cv2.imread(self.IMAGE_FOLDER_PATH+'\\alive'+str(3)+'.png')
        self.alive4=cv2.imread(self.IMAGE_FOLDER_PATH+'\\alive'+str(4)+'.png')
        self.alive5=cv2.imread(self.IMAGE_FOLDER_PATH+'\\alive'+str(5)+'.png')
        self.list_of_photos_alive=[self.alive1,self.alive2,self.alive3,self.alive4,self.alive5]
        self.dead1=cv2.imread(self.IMAGE_FOLDER_PATH+'\\DEAD'+str(1)+'.png')
        self.dead2=cv2.imread(self.IMAGE_FOLDER_PATH+'\\DEAD'+str(2)+'.png')
        self.dead3=cv2.imread(self.IMAGE_FOLDER_PATH+'\\DEAD'+str(3)+'.png')
        self.dead4=cv2.imread(self.IMAGE_FOLDER_PATH+'\\DEAD'+str(4)+'.png')
        self.dead5=cv2.imread(self.IMAGE_FOLDER_PATH+'\\DEAD'+str(5)+'.png')
        self.list_of_photos_dead=[self.dead1,self.dead2,self.dead3,self.dead4,self.dead5]
        self.doll_situation='b'

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
                self.doll_situation='b'
        if(self.mode=='g'):
            display_list=self.list_of_frames_backward

        else:
            display_list=self.list_of_frames_forward
        
        self.display_frame[360:,1280:,:]=display_list[self.count_frames_of_doll]
        if self.count_frames_of_doll <= (len(display_list)-41):         
            self.count_frames_of_doll+=40
        else:
            if(self.mode=='r'):
                self.doll_situation='f'
            
        


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
        image=self.frame_b

        ALPHA=0.8 #level of transparency

        if(len(keypoints.items())>5):
            raise Exception("More than 5 persons in the frame")
        i=0
        print(len(keypoints))
        for id,person_keypoints in keypoints.items():
            # the vertices of the polygon is stored in points
            list1=[0,5,4,3,15,14,13,9,12,11,10,2,1]
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
    
    def change_front_frame(self,frame):
        self.frame_f=frame

    def change_back_frame(self,frame):
        self.frame_b=frame

def display_thread(vid_obj,people_obj):
    while True:
        vid_obj.display_doll()
        print(people_obj.sort_curr_keypts_cam1)
        vid_obj.hightlight_on_webcam(people_obj.sort_curr_keypts_cam1)
        vid_obj.player_status=people_obj.playerstatus
        vid_obj.show_player_status()

        cv2.imshow('Display Frame',cv2.resize(vid_obj.display_frame,(1080,720)))
        if cv2.waitKey(1)==ord('q'):
            break

def front_thread(v_obj):
    print("front thread running")
    cam_obj = cv2.VideoCapture(UPFRONT2666_PATH)
    while True:
        ret , frame = cam_obj.read()
        v_obj.change_front_frame(frame)
        v_obj.change_back_frame(frame)
        if not ret:
            break

def model_thread(people_obj):
    i=0
    numpyfile=np.load("C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\upfront2666.npy",allow_pickle=True)
    while i<len(numpyfile):
        t=time.time()
        people_obj.change_keypoints(numpyfile[i],numpyfile[i])
        time.sleep(1/100)
        i+=1

def error_fn_thread(people_obj,vid_obj):
    while True:
        if(vid_obj.doll_situation=='b'):
            people_obj.eliminate(0.1)


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

if __name__=="__main__":
    vid_obj=Video(DOLL_VIDEO_PATH,IMAGE_FOLDER_PATH)
    people_obj=People(5)
    keypoints_loaded=np.load("C:\\Users\\sujal\\Desktop\\NITK\\RedLight_GreenLight\\upfront2666.npy",allow_pickle=True)
    print(len(keypoints_loaded[500]))
    t1=threading.Thread(target=front_thread,args=(vid_obj,))
    t2=threading.Thread(target=display_thread,args=(vid_obj,people_obj))
    t3=threading.Thread(target=model_thread,args=(people_obj,))
    t4=threading.Thread(target=error_fn_thread,args=(people_obj,vid_obj))
    t1.start()
    t2.start()
    t3.start()
    t4.start()


