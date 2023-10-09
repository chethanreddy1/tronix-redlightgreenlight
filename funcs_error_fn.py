def read(p1,v1):
    while(True):
        if(v1.r_flag):
            p1.error_fn2()
        else:
            pass

#Threading for error function with People and Video 
import threading
p1 = People(past_keypoint_cam1,curr_keypoint_cam1,past_keypoint_cam2,curr_keypoint_cam2)
v1 = Video()
t1=threading.Thread(target = read,args=(p1,v1))
t1.start()
t1.join()