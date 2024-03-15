# import image module from pillow
#from PIL import Image
import math
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
#from skimage import io, color
#from pandas import array

class_number = 6
Original_Component_directory = r'C:\Users\ASUS\Desktop\YOLOV3\Project\di-club-technical\dataset\C0%s\Original'%str(class_number)
Saved_image_directory = r'C:\Users\ASUS\Desktop\YOLOV3\Project\di-club-technical\dataset\C0%s'%str(class_number)
label_directory = r'C:\Users\ASUS\Desktop\YOLOV3\Project\di-club-technical\dataset\C0%s\Labels'%str(class_number)
status = 'C%s'%str(class_number)  # name of component

threshold = 45  # A good number to use in different situation


print("class_number is: ",class_number)

class Line:
    def __init__(self,start_point,end_point, thickness) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.color = (0,0,0)
        self.thickness =  thickness

    def figure_the_line(self,image):
        line = cv2.line(image, self.start_point, self.end_point, self.color, self.thickness)
        return line

def load_imges():
    os.chdir(Original_Component_directory)
    img_list = []
    list_of_original_img = []
    for file_names in os.listdir():
        list_of_original_img.append(file_names)
        img_list.append( np.array(cv2.imread(file_names)))
    return list_of_original_img,img_list


def creat_white_image(image):
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            image[j,i]=255
    return image

def show_img(img,number):
    
    cv2.imshow(str(number),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image,count,save_directory,status):
    os.chdir(save_directory)
    cv2.imwrite(save_directory+'\{symbol_name}_'.format(symbol_name=status)+ str(count)+'.jpg',image)

def create_labels(count,x,y,w,h, label_directory,status,label_name):
    os.chdir(label_directory)
    f= open("{symbol_name}_".format(symbol_name=status)+str(count)+".txt","w+")
    f.write("{label_assigend_number} {x_loc} {y_loc} {w_rec} {h_rec}".format(label_assigend_number= label_name,x_loc= x,y_loc= y, w_rec= w,h_rec= h ))

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img



def sp_noise(image,prob):

    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output   
def get_boundary(image):
    return image

def transformation(boundary,count,status,x,y,w,h):
    # Adding Noise
    noise_img = sp_noise(get_boundary(boundary),0.02)
    
    save_image(noise_img,count,Saved_image_directory,status + '_noise')
    create_labels(count,x,y,w,h,label_directory,status + '_noise',class_number)

    # Bluring img
    #random_kernel_size = random.randrange(5, 15, 5)
    blur_img = cv2.GaussianBlur(get_boundary(boundary), (7,7),0) 
    save_image(blur_img,count,Saved_image_directory,status + '_Blur')
    create_labels(count,x,y,w,h,label_directory,status + '_Blur',class_number)

    # Rotating img
    
    

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def drowing_rotation(white_img, rotate_images, minimum_x,maximum_x,minimum_y,maximum_y):
    count = 1
    i = 0
    for image in rotate_images:   
        image = rgb2gray(image)
        if i % 2 == 0 :
            print("Process image ",images_list[i+4])
        for x in range(minimum_x,maximum_x):
            for y in range (minimum_y,maximum_y):
                if x % 10 == 0 and y % 30 == 0 and count <251:
                    white_img[y:y+image.shape[0],x:x+image.shape[1]] = image
                    boundary = cv2.rectangle(white_img, (minimum_x//2,minimum_y//2), (white_img.shape[0]-(minimum_x//2),white_img.shape[1]-(minimum_y//2)), (0,0,0), 1)
                        
                    
                    # provide YOLO label structure
                    x_boundary,y_boundary = ((x+ (image.shape[0]/2))/white_img.shape[0]),    ((y+ (image.shape[1]/2))/white_img.shape[1])
                    w_boundary,h_boundary = ((image.shape[0])/white_img.shape[0]),((image.shape[1])/white_img.shape[1])
                    x_boundary= (math.floor((x_boundary+ 0.00001)*1000000)/1000000) 
                    y_boundary= (math.floor((y_boundary+ 0.00001)*1000000)/1000000) 
                    w_boundary= (math.floor((w_boundary + 0.02001)*1000000)/1000000) 
                    h_boundary= (math.floor((h_boundary+ 0.02001)*1000000)/1000000) 
                    save_image(boundary,count,Saved_image_directory,status+'_rotate')
                    create_labels(count,x_boundary,y_boundary,w_boundary,h_boundary,label_directory,status+'_rotate',class_number)
                    white_img = creat_white_image(white_img)
                    # if count % 100 == 0:
                    #     print(count)
                    count = count+1
        i+=1


################################################ ************************* ##################################
################################################ ************************* ##################################
################################################ ************************* ##################################
################################################ ************************* ##################################
# Implementations:


#Load original images:
images_list,images = load_imges()
rotated_images = []
rotated_images = images[5:-1] 
rotated_images.append(images[-1])
images = images[0:5]

# Creat a white image 416*416
background = np.ones((416,416), dtype=np.uint8)
white_img = creat_white_image(background)

# knowing maximum dimantions of original images
tmp = []
for image in images_list:
    img = cv2.imread(image)
    img = np.array(img[0:,0:,1])
    tmp.append(img.shape)
max_dimention = max(tmp)
print(max_dimention)
# define max dimentions(boundary) for shifting image
maximum_x = white_img.shape[0] - max_dimention[0] - threshold 
minimum_x = 2*threshold 
maximum_y = white_img.shape[1] - max_dimention[1] - threshold 
minimum_y = 2*threshold 
print("length of area that images are shifted in (" ,maximum_x - minimum_x,", ",maximum_y- minimum_y,")")

############################
############################
############################
############################
############################
############################
############################
############################

# for img in images:
#     cv2.imshow("input",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    

count = 1
i = 0
for image in images:   
    image = rgb2gray(image)
    if i % 2 == 1:
        print("Process image ",images_list[i])
    for x in range(minimum_x,maximum_x):
        for y in range (minimum_y,maximum_y):
            if x % 10 == 0 and y % 30 == 0 and count <251:
                white_img[y:y+image.shape[0],x:x+image.shape[1]] = image
                
                

                ###########################
                # drowing the above conection (vertically)
                #random_tickness = random.randrange(4, 6, 2)
                # vertical_random_x =  random.randrange((image.shape[0]-(random.randrange(2, 4, 1))), (image.shape[0]+(random.randrange(2, 4, 1))), 1)
                # vertical_above_line_obj = Line((x-(image.shape[0]//2)+vertical_random_x-2,y),(x-(image.shape[0]//2)+vertical_random_x-2,minimum_y//2), 6)
                # #vertical_above_line_obj = Line((x+1,y),(x+1,minimum_y//2), random_tickness)
                # vertical_above_line = vertical_above_line_obj.figure_the_line(white_img)
                ######################
                # drowing the down conection (vertically)
                # vertical_random_x =  random.randrange((image.shape[0]-(1)), (image.shape[0]+(1)), 1)
                # vertical_down_line_obj = Line((x-(image.shape[0]//2)+vertical_random_x-2,y+image.shape[1]),(x-(image.shape[0]//2)+vertical_random_x-2,white_img.shape[1]-(minimum_y//2)), 6)
                # #vertical_down_line_obj = Line((x+1,y+image.shape[1]),(x+1,white_img.shape[1]-(minimum_y//2)), random_tickness)
                # vertical_above_line = vertical_down_line_obj.figure_the_line(vertical_above_line)
                ##################

                boundary = cv2.rectangle(white_img, (minimum_x//2,minimum_y//2), (white_img.shape[0]-(minimum_x//2),white_img.shape[1]-(minimum_y//2)), (0,0,0), 1)
                        
                save_image(boundary,count,Saved_image_directory,status)
                
                # provide YOLO label structure
                x_boundary,y_boundary = ((x+ (image.shape[0]/2))/white_img.shape[0]),    ((y+ (image.shape[1]/2))/white_img.shape[1])
                w_boundary,h_boundary = ((image.shape[0])/white_img.shape[0]),((image.shape[1])/white_img.shape[1])
                x_boundary= (math.floor((x_boundary+ 0.00001)*1000000)/1000000) 
                y_boundary= (math.floor((y_boundary+ 0.00001)*1000000)/1000000) 
                w_boundary= (math.floor((w_boundary + 0.02001)*1000000)/1000000) 
                h_boundary= (math.floor((h_boundary+ 0.02001)*1000000)/1000000) 

                create_labels(count,x_boundary,y_boundary,w_boundary,h_boundary,label_directory,status,class_number)
                        
                transformation(boundary,count,status,x_boundary,y_boundary,w_boundary,h_boundary)
                white_img = creat_white_image(white_img)
                # if count % 100 == 0:
                #     print(count)
                count = count+1
                
                
    i+=1
        # if count % 250 == 0:
        #     break
##############################################
##############################################


white_img = creat_white_image(white_img)
drowing_rotation(white_img, rotated_images, minimum_x,maximum_x,minimum_y,maximum_y)

