
from operator import index
import numpy as np
import pandas as pd
import cv2 
from cv2 import compare
############################## 
############################## 
# SOME USEFUL PARAMETERS
classes_file_path = "./dataset/classes.names"
dataset_classes = []
dataset_file_names =[]
confige_file_path = "./darknet/cfg/yolov4-custom.cfg"
output_weights = "./Outputs/backup/yolov4-custom_60000.weights"
test_dataset_directory = "./dataset/test2.txt"
BOLB_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
###################################################################
############################ Functions ############################
###################################################################



def open_files():
    with open(classes_file_path, "rt") as f:
        dataset_classes = f.read().rstrip("\n").split("\n")
        df = pd.DataFrame()
        print(f"DataFrame:\n{df}\n")
        return dataset_classes

def network_initialize(confige_file_path,output_weights):
    net = cv2.dnn.readNetFromDarknet(confige_file_path, output_weights) # Creating darknet network
    # set configuration for darknet
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)    
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def read_test_dataset():
    with open(test_dataset_directory, "rt") as f:
        images_file_path = f.read().rstrip("\n").split("\n")
        print("dataset size = ", len(images_file_path),"\n\n ###################### \n ###################### \n")
        
        return images_file_path

def image_preprocessing(image_path, net):
    print('image_path:' ,image_path)
    frame = cv2.imread(image_path)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(BOLB_SIZE, BOLB_SIZE), mean=(0, 0, 0),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    output = net.forward(out_names)
    
    return output,frame

def creating_confusion_matrix_datafram(classes):
    M =pd.DataFrame(data = np.zeros(shape=[len(classes)+1,len(classes)+1],dtype=np.int8),columns=classes + ['NR'],index = classes + ['NL'] )
    return M
def color_rule(val):
    return ['background-color: red' if isinstance(x, int) and x > 0 else 'background-color: yellow' for x in val]

def find_objects(output, img, sample_name,df):
    img_h, img_w, img_c = img.shape
    bboxes = []
    class_ids = []
    confidences = []
    for cell in output:
        for detect_vector in cell:  # output[0][0] ... output[0][299], output[1][0]...output[1][1199]
            scores = detect_vector[5:]  # [x,y,w,h,Pc,c1,c2,..] ==> [c1,...] probability of output detection
            
            
            class_id = np.argmax(scores)  # get the related max probability index
            
            confidence = scores[class_id]  # get the value of max probability.
            
            if confidence > CONFIDENCE_THRESHOLD:
                #print(confidence)
                w, h = (detect_vector[2] * img_w), (detect_vector[3] * img_h)
                x, y = int((detect_vector[0] * img_w) - w / 2), int((detect_vector[1] * img_h) - h / 2)
                bboxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    indices = [indices]
    
    print (indices[0], len(indices[0]) , "NL" in sample_name)
    if "NL" in sample_name and len(indices[0]) == 0:
        print(df['NR']['NL'])
        df['NR']['NL'] = df['NR']['NL'] +1
        return 'NL'
    else:
        for i in indices:
            #print(type(i))
            if  np.size(i):
                
                i = i[0]
                
                #bbox = bboxes[i]
                #x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                
                if classes[class_ids[i]] in classes:
                    
                    #print("frame:",sample_name.split('/')[1].split('_')[0],"      rec:",classes[class_ids[i]])
                    return classes[class_ids[i]]
            else:
                break
            
            
        
        df['NR'][sample_name.split('/')[-1].split('_')[0]] = df['NR'][sample_name.split('/')[-1].split('_')[0]] +1
        print("BYE")
        return 'Not recognize'

        #cv2.rectangle(img, (x+5, y+5), (x + int(w)-5, y + int(h)-5), (0, 255, 0), 2)
        #cv2.putText(img, f'{coco_classes[class_ids[i]].upper()} ',
              #      (x, y+int(h/4) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)





###################################################################
################ functions executions and codes  ##################
################################################################### 


# open clasess files
classes = open_files()
print('classes_names are: \n',classes)

# network_initialize
net = network_initialize(confige_file_path,output_weights)
# read_test_dataset path
images_file_path = read_test_dataset()
df = creating_confusion_matrix_datafram(classes)
print(df)
#image_preprocessing(images_file_path,net)

#print(images_file_path[0].split('/')[1].split('_')[0])
#df['C00']['C01'] = 1 # first is column second is rows
#print(df)
#print(df[images_file_path[0].split('/')[1].split('_')[0]][images_file_path[0].split('/')[1].split('_')[0]] )
# if images_file_path[0].split('/')[1].split('_')[0] in df.columns:
#     print(images_file_path[0].split('/')[1].split('_')[0])

count = 0
for sample_name in images_file_path:
    
    if count % 50 == 0:
        print("################################# \n ################################# \n ################################# \n We are checking sample: ",count)
        
    obj = ''
    output,frame = image_preprocessing(sample_name,net)
    obj = find_objects(output, frame , sample_name , df)
    
    print('obj found = ',obj , "   ground_truth = ",sample_name.split('/')[-1].split('_')[0])
    if obj == "NL":
        print("array[NR]","[NL] = ",df["NR"]["NL"])
        print("NO object is in pic \n ########### \n ###########")

    if sample_name.split('/')[-1].split('_')[0] == obj and obj !='NL':
        
        print(df[obj][sample_name.split('/')[-1].split('_')[0]])
        df[obj][sample_name.split('/')[-1].split('_')[0]]= df[obj][sample_name.split('/')[-1].split('_')[0]] + 1
        print("array[",obj,"]","[",sample_name.split('/')[-1].split('_')[0],"] = ",df[obj][sample_name.split('/')[-1].split('_')[0]])
        print("\n We had TP \n ########### \n ###########")
        # if count % 100 == 0:
        #     print("array[",obj,"],[",obj,"] = ",df[obj][obj])
        #     print("\n We had TP \n ########### \n ###########")

    if obj == "Not recognize":
        print("array[NR]","[",sample_name.split('/')[-1].split('_')[0],"] = ",df["NR"][sample_name.split('/')[-1].split('_')[0]])
        print("there is object but no recognition found \n ########### \n ###########")
    
    if sample_name.split('/')[-1].split('_')[0] != obj and obj != "Not recognize":
        print("salam")
        df[obj][sample_name.split('/')[-1].split('_')[0]]= df[obj][sample_name.split('/')[-1].split('_')[0]] + 1 #colomn / row 
        print("array[",obj,"],[",sample_name.split('/')[-1].split('_')[0],"] = ",df[obj][sample_name.split('/')[-1].split('_')[0]])
        print('we have TN \n ########### \n ###########')
    
    count = count+1
#print(df)
print(df.sum())
total = 0
for i in range(0,len(df.columns)):
    total = total + int(df.sum()[i])

print("the total number of experiments = ",total)

# save to exel
html_column = df.style.apply(color_rule, axis=1)
html_column.to_excel(r'./codes/Evaluations/styled.xlsx')

# save to csv
df = df.replace(0,"")
df.to_csv(r'./codes/Evaluations/confusion_M.csv', sep=',', mode='a')

























