class People:
    def __init__(self):
        self.past_keypoints_cam1 = []
        self.curr_keypoints_cam1 = []
        self.past_keypoints_cam2 = []
        self.curr_keypoints_cam2 = []
        self.l = [1,1,1,1,1]
        self.flag = 1                                #Flag if people are more than 5
        self.er = 0.003
        self.begin = True

    def change_keypoints(self,keypoints1,keypoints2):
        if(not self.begin):
           self.past_keypoints_cam1 = self.curr_keypoints_cam1
           self.curr_keypoints_cam1 = keypoints1
           self.past_keypoints_cam2 = self.curr_keypoints_cam2
           self.curr_keypoints_cam2 = keypoints2
        else:
            self.curr_keypoints_cam1 = keypoints1
            self.curr_keypoints_cam2 = keypoints2
            self.past_keypoints_cam1 = keypoints1
            self.past_keypoints_cam2 = keypoints2

        keyps1 = self.curr_keypoints_cam1.keys()
        keyps2 = self.curr_keypoints_cam2.keys()
    
        if(min(len(self.curr_keypoints_cam1),len(self.curr_keypoints_cam2)) >= 5):
            assert(0),"Greater than 5 players detected"
            self.flag=0
        
        self.curr_keypoints_cam2[keyps2[0]], self.curr_keypoints_cam2[keyps2[4]] = self.curr_keypoints_cam2[keyps2[4]], self.curr_keypoints_cam2[keyps2[0]]
        self.curr_keypoints_cam2[keyps2[1]], self.curr_keypoints_cam2[keyps2[3]] = self.curr_keypoints_cam2[keyps2[3]], self.curr_keypoints_cam2[keyps2[1]]
    
        self.past_keypoints_cam2[keyps2[0]], self.past_keypoints_cam2[keyps2[4]] = self.past_keypoints_cam2[keyps2[4]], self.past_keypoints_cam2[keyps2[0]]
        self.past_keypoints_cam2[keyps2[1]], self.past_keypoints_cam2[keyps2[3]] = self.past_keypoints_cam2[keyps2[3]], self.past_keypoints_cam2[keyps2[1]]    

    def error_fn(self):
        error =[]
        if len(self.curr_keypoints_cam1)<len(self.curr_keypoints_cam2):
            dict_of_key=self.curr_keypoints_cam1
        else:
            dict_of_key=self.curr_keypoints_cam2
        for j,key in enumerate(dict_of_key): #Iterating through all the players
            past_arr1 = np.array(self.past_keypoints_cam1[key])
            curr_arr1 = np.array(self.curr_keypoints_cam1[key])

            past_arr2 = np.array(self.past_keypoints_cam2[key])
            curr_arr2 = np.array(self.curr_keypoints_cam2[key])
            
            x_prev1 = past_arr1[:,0]
            y_prev1 = past_arr1[:,1]
            x_curr1 = curr_arr1[:,0]
            y_curr1 = curr_arr1[:,1]
            conf_curr1 = curr_arr1[:,3]

            x_prev2 = past_arr2[:,0]
            y_prev2 = past_arr2[:,1]
            x_curr2 = curr_arr2[:,0]
            y_curr2 = curr_arr2[:,1]
            conf_curr2 = curr_arr2[:,3]

            if(np.mean(conf_curr1)>np.mean(conf_curr2)):    
                p_prev = np.concatenate((x_prev1,y_prev1))
                p_curr = np.concatenate((x_curr1,y_curr1))
            else:
                p_prev = np.concatenate((x_prev2,y_prev2))
                p_curr = np.concatenate((x_curr2,y_curr2))
            
            p1 = np.sum(p_prev - np.mean(p_prev))/np.std(p_prev)
            p2 = np.sum(p_curr - np.mean(p_curr))/np.std(p_curr)

            error.append(np.mean(np.square(np.subtract(p1,p2))))
            print(j," : ",error[j])
            if(error[j]>self.er):
                self.l[j]=0
