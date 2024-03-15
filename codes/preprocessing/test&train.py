import os
import numpy as np
from sklearn.model_selection import train_test_split
full_path_to_images = r'C:/Users/ASUS/Desktop/yolov4/sdp_classifier/dataset'
os.chdir(full_path_to_images)
train_set = []
test_set = []
for path in os.listdir():
    
    os.chdir(path)
    for file in os.listdir():
        if path == 'train':
            if file.endswith('.jpg'):
                path_to_save_into_txt_files = full_path_to_images+'/'+path+'/'+file
                train_set.append(path_to_save_into_txt_files+'\n')
        if path == 'test':
            if file.endswith('.jpg'):
                path_to_save_into_txt_files = full_path_to_images+'/'+path+'/'+file
                test_set.append(path_to_save_into_txt_files+'\n')
        # if path == 'NL':
        #     if file.endswith('.jpg'):
        #         path_to_save_into_txt_files = full_path_to_images+'/'+path+'/'+file
                # test_set.append(path_to_save_into_txt_files+'\n')
    os.chdir('../')
# for current_dir, dirs, files in os.walk('.'):
#     # Going through all files
#     for f in files:
#       if f.endswith('.jpg'):
#         path_to_save_into_txt_files = full_path_to_images + '/' + f
#         train.append(path_to_save_into_txt_files + '\n')

# dividing dataset and test set as we had before

# for i in range (0,len(dataset)+1):
#     if i % 100 == 0:       
#         test_set.append(dataset[i])  # 10% for test
#         train_set= train_set + dataset[i-90:i] + '\n'  # 90 % for train
    
print('Data size: ',len(train_set)+len(test_set))
print("Test size: ",len(test_set), "percentage of whole:",len(test_set)/(len(train_set)+len(test_set)))
print("Train size: ",len(train_set))

os.chdir(full_path_to_images)
# Creating file train.txt and writing 80% of lines in it
# with open('train.txt', 'w') as train_txt:
#     for i in train_set:
#         train_txt.write(i)

# Creating file test.txt and writing 20% of lines in it
with open('test.txt', 'w') as test_txt:
    for i in test_set:
        test_txt.write(i)