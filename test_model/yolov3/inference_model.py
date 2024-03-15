
import cv2

import numpy as np
import os
cap = cv2.VideoCapture(0)
coco_file = "classes.names"
coco_classes = []
net_conf = "cfg/yolov4-custom.cfg"
net_weights = "cfg/yolov4-custom_60000.weights"
blob_size = 320
confidence_threshold = 0.5
nms_threshold = 0.3

with open(coco_file, "rt") as f:
    coco_classes = f.read().rstrip("\n").split("\n")
print(coco_classes)
# print(len(coco_classes))

net = cv2.dnn.readNetFromDarknet(net_conf, net_weights) # Creating darknet network
# set configuration for darknet
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)    
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def find_objects(output, img):
    img_h, img_w, img_c = img.shape
    bboxes = []
    class_ids = []
    confidences = []
    for cell in output:
        for detect_vector in cell:  # output[0][0] ... output[0][299], output[1][0]...output[1][1199]
            scores = detect_vector[5:]  # [x,y,w,h,Pc,c1,c2,..] ==> [c1,...] probability of output detection
            
            
            class_id = np.argmax(scores)  # get the related max probability index
            
            confidence = scores[class_id]  # get the value of max probability.
            
            if confidence > confidence_threshold:
                #print(confidence)
                w, h = (detect_vector[2] * img_w), (detect_vector[3] * img_h)
                x, y = int((detect_vector[0] * img_w) - w / 2), int((detect_vector[1] * img_h) - h / 2)
                if h > 0.1 and w >0.1: # in order to remove small bounding box on same region of interest
                    bboxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    #print(coco_classes[class_ids[indices[0]]],coco_classes[class_ids[indices[1]]],coco_classes[class_ids[indices[2]]],coco_classes[class_ids[indices[3]]],coco_classes[class_ids[indices[4]]],coco_classes[class_ids[indices[5]]],coco_classes[class_ids[indices[6]]],coco_classes[class_ids[indices[7]]],coco_classes[class_ids[indices[8]]],coco_classes[class_ids[indices[9]]])
    # np_array=np.asarray(indices)
    
    # indices = np.reshape(np_array,(2,-1))
    # indices = indices[0]
    #print('indices.shape',indices.shape)
    
    #print('indices is: ', indices,coco_classes[class_ids[indices[0]]])
    #print(coco_classes[class_ids[indices[0]]],coco_classes[class_ids[indices[1]]],coco_classes[class_ids[indices[2]]],coco_classes[class_ids[indices[3]]],coco_classes[class_ids[indices[4]]])
    #print(bboxes[indices[0][0]],bboxes[indices[0][1]],bboxes[indices[0][2]],bboxes[indices[0][3]],bboxes[indices[0][4]])
    
    for i in indices:
        
        #print(i.size, type(i), i.shape, i)
        if i.size == 1:
            #print('raft tu in')
            # max_i.append(i)
            # if i < max(max_i):
            #     i = max(max_i)
            #     print("max ", i)
            
            #print('i is:',i, class_ids[i])
            bbox = bboxes[i]
            #print("bbox",bbox)
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            cv2.rectangle(img, (x+5, y+5), (x + int(w)-5, y + int(h)-5), (0, 255, 0), 2)
            cv2.putText(img, f'{coco_classes[class_ids[i]].upper()} ',
                    (x, y-int(h/16) ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,)
            
        
        else:
            i = i.max()
            
            #print('i is:',i)
            bbox = bboxes[i]
            print("bbox",bbox)
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            cv2.rectangle(img, (x+5, y+5), (x + int(w)-5, y + int(h)-5), (0, 255, 0), 2)
            cv2.putText(img, f'{coco_classes[class_ids[i]].upper()} ',
                    (x, y-int(h/4) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


count = 0
for files in os.listdir():
    if files.endswith('.jpg') or files.endswith('.png'):
        print(files)
        frame = cv2.imread(files)


        #success, frame = cap.read()
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for x in range(frame.shape[0]):
            for y in range (frame.shape[1]):
                for c in range (frame.shape[2]):
                    if frame[x][y][c] <= 50:
                        frame[x][y][c] = 0
                    else:
                        frame[x][y][c] = 255
        

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(blob_size, blob_size), mean=(0, 0, 0),
                                    swapRB=True, crop=False)

        net.setInput(blob)
        out_names = net.getUnconnectedOutLayersNames()
        output = net.forward(out_names)
        find_objects(output, frame)
        cv2.imwrite('./saves/aaimgs_'+str(count)+'.jpg',frame)
        count = count +1
        cv2.imshow("Webcam", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
