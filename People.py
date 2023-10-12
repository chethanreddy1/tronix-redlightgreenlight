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
    
