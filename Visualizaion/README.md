# CLASP1 Person-Bin Association and Theft Detection
## 1. Get all pose detection information
For this part, we depend on OpenPose to get the positions of human hands in the scenes. After this, we will have the pose for every frame stored in image_dir_output folder, in form of json.
## 2. Get all hands information file **hands_noid.pkl** using **draw_hands.py**
Use this script to get the hands' positions of every frame, which can then be used to get the association
## 3. Associate Bin and Person using persons' hands positions and IOU of bins' bounding boxes and people's bounding boxes
**association.py** and **association_11.py**  
Basically association loads people, bin, and hands data and computes the intersection among the three things. Based on that, makes a list of possible events.  

Some important things:  
 
association.py  
Line 222 saves a dictionary with all the camera 9 events that will be later used by camera 11
The code is actually very simple, the thing is that because of the demo thing, there are a lot of things that are just for drawing in the overlay, and freeze the screen and so on, so don´t freak out when you see all the code haha
  
As you will see in the script, there are some paths that refer to:  
- The input directory with all images  
- The detections of the people  
- Bin detections  
- Hands detection  
  
They are not arguments, they are just variables that you can change  

# CLASP2 Person-Bin Association and Theft Detection
## 1. Pose Detection for the video clip or image directory
For this part, we depend on OpenPose to get the positions of human hands in the scenes. After this, we will have the pose for every frame stored in image_dir_output folder, in form of json.
## 2. Associate hands with person detection results
### **2.1 opps_joint2mat.py**: 
Store all the poses corresponding with hands information into one .mat file
### **2.2 hands_association.py**:
Associate detected hands with person ID using IOU between skeletons and person detection bounding boxes
## 3. Associate Bin and Person using persons' hands positions and bins' positions
**association_handsID.py**  
Use the bins' bounding boxes as the main body to do the event detection every frame
If one's interaction count with one bin is over the threshold, he/she would be the owner of this bin. And the interaction count for other people would not accumulate any more to fix the owner of the bin.