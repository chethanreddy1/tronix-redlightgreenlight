class people():
    def __init__(self,past_keypoint_cam1,curr_keypoint_cam1,past_keypoint_cam2,curr_keypoint_cam2):
        self.past_keypoint_cam1 = past_keypoint_cam1
        self.curr_keypoint_cam1 = curr_keypoint_cam1
        self.past_keypoint_cam2 = past_keypoint_cam2
        self.curr_keypoint_cam2 = curr_keypoint_cam2
        self.l = [1,1,1,1,1]
        self.flag = 1                                #Flag if people are more than 5
        self.del = 0.003

    if(assert min(len(curr_keypoints_cam1),len(curr_keypoints_cam2) >= 5):
       flag=0
       
    self.past_keypoint_cam2 = past_keypoint_cam2
    self.curr_keypoint_cam2 = curr_keypoint_cam2
       
    self.curr_keypoint_cam2[0], self.curr_keypoint_cam2[4] = self.curr_keypoint_cam2[4], self.curr_keypoint_cam2[0]
    self.curr_keypoint_cam2[1], self.curr_keypoint_cam2[3] = self.curr_keypoint_cam2[3], self.curr_keypoint_cam2[1]
    
    self.prev_keypoint_cam2[0], self.prev_keypoint_cam2[4] = self.prev_keypoint_cam2[4], self.prev_keypoint_cam2[0]
    self.prev_keypoint_cam2[1], self.prev_keypoint_cam2[3] = self.prev_keypoint_cam2[3], self.prev_keypoint_cam2[1]
    
    def error_fn2(self):
        error = []

        for j in range(min(len(curr_keypoints_cam1),len(curr_keypoints_cam2)):#Total number of players
            if(np.std(curr_keypoints_cam1)>np.std(curr_keypoints_cam2)):
                       prev_keypoints=self.past_keypoint_cam1
                       curr_keypoints=self.curr_keypoint_cam1
            else:
                       prev_keypoints=self.past_keypoint_cam2
                       curr_keypoints=self.curr_keypoint_cam2
            
            prev_arr = np.array(prev_keypoints[j])
            curr_arr = np.array(curr_keypoints[j])

            x_prev = prev_arr[:,0]
            y_prev = prev_arr[:,1]
            p_prev = np.concatenate((x_prev,y_prev))

            p1 = np.sum(p_prev - np.mean(p_prev))/np.std(p_prev)

            x_curr = curr_arr[:,0]
            y_curr = curr_arr[:,1]
            p_curr = np.concatenate((x_curr,y_curr))

            p2 = np.sum(p_curr - np.mean(p_curr))/np.std(p_curr)

            error.append(np.mean(np.square(np.subtract(p1,p2))))
            print(j," : ",error[j])
            if(error[j]>del):
                       l[j]=0