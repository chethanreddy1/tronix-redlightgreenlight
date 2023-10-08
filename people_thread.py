def people_thread(cam_obj1,cam_obj2,people_object,model1,model2):
    while True:
        frame1 = cam_obj1.read()
        frame2 = cam_obj2.read()
        keypoints1 =model1.inference(frame1)
        keypoints2= model2.inference(frame2)
        people_object.change_keypoints(keypoints1,keypoints2)
        
    
        
        
    
