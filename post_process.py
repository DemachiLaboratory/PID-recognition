import os
import cv2 as cv
import argparse
import numpy as np
import math
import linecache
from pre_process import show,clear


#rootdir = "D:/Download/PID_YOLO/train"
#savepath = "D:/Download/PID_YOLO/train/return"
rootdir = "/mnt/data/datasets/PID_YOLO/divide"  #images+labels acquire from 
savepath = "/mnt/data/datasets/PID_YOLO/divide/return"  # images+labels save in 
filedir = rootdir+"/coordinate"

#rootdir refers to the cropped output ,savepath refers to the path for saving postprocess
#filedir refers to the filedir saving the infomation of coordinate for each adaptive cropped images 


#for post-process: re-calculate the coordinatate and then recover original image
def main():
    if os.path.isdir(savepath):
        clear(savepath) 
    os.makedirs(savepath) 
    #First, read the absolute coordinate of cropped image from filedir
    filelist = os.listdir(rootdir) 
    normalization = np.zeros(4)
    n = 0
    name = []
    #the normalization refers to the original size of image
    #n refer to the order of original image, name refers to the name of each
    for filename in filelist:  
        if filename[-1] == "g" :
            print(filename)
            i = int(filename[-6])
            j = int(filename[-5])
            image = cv.imread(os.path.join(rootdir, filename))
            
            #if i=0 and j=0. a new original image is loaded
            #save the former completied read image
            #then record the new name, initilize the size of new image and build a new imgae as background
            if i ==0 and j==0 :
                name.append(filename[:-7]) 
                print(name)
                if n != 0 :
                    save = savepath + '/' + name[n-1] +'.jpg'
                    cv.imwrite(save, final_matrix)
                maximum = np.zeros(4)
                #norm function is to calculate the size of each orginal image according to the cropped image 
                norm(name[n],maximum)
                #new image
                final_matrix = np.zeros((int(maximum[3]), int(maximum[1]), 3), np.uint8)
                n+=1
            
            #every original image calculate size once at the begining of loading
            imw = maximum[1] - maximum[0]
            imh = maximum[3] - maximum[2]

            #paste every cropped image on the background image according to the coordinate file
            #in every coordinate file, it only contains one line for the image four coordinate imformation
            #the minimum and maximum x-coordinate and the minimum and maximum y-coordinate of cropped image
            filename1 = filename[:-3] + 'txt'
            f = open(os.path.join(filedir,filename1),'r') 
            lines = f.readlines()
            img = lines[0].strip().split(" ")
            img = list(map(float,img))
            #imx_min, imx_max refers to the minimum and maximum x-coordinate of each image on the orignal image respectively
            #imy_min, imy_max refers to the minimum and maximum y-coordinate of each image on the orignal image respectively
            imx_min = int(img[0])
            imx_max = int(img[1])
            imy_min = int(img[2])
            imy_max = int(img[3])
            w = imx_max - imx_min
            h = imy_max - imy_min
            #paste every cropped image, no need to worry about interception because it will cover automatically
            final_matrix[imy_min:imy_max, imx_min:imx_max] = image
            f.close()

            #then, re-calculate the related coordinate for the orginal images and save into according label file
            file = open(savepath +'/' + name[n-1]+'.txt','a+') 
            f = open(os.path.join(rootdir,filename1),'r') 
            lines = f.readlines()
            #calculte every bounding box related coordination to the absolute coordinate on original image
            #x_min, x_max refers to the minimum and maximum x-coordinate of each bounding box on the orignal image respectively
            #y_min, y_max refers to the minimum and maximum y-coordinate of each bounding box on the orignal image respectively
            for line in lines : 
                box = line.strip().split(" ")
                box = list(map(float,box))
                xmin = (box[1]-box[3]/2)*w + imx_min
                ymin = (box[2]-box[4]/2)*h + imy_min
                xmax = (box[1]+box[3]/2)*w + imx_min
                ymax = (box[2]+box[4]/2)*h + imy_min                  
                #char is the related coordinate of each bounding box for original image as the YOLO format
                char = np.zeros(5)
                char[0] = int(box[0])
                char[1] = (xmin + xmax)/(2*imw)
                char[2] = (ymin + ymax)/(2*imh)
                char[3] = (xmax - xmin)/imw 
                char[4] = (ymax - ymin)/imh 
                char = list(map(str,char))
                file.write(' '.join(char))
                file.write('\n')
            file.close()
            f.close()
    save = savepath + '/' + name[n-1] +'.jpg'
    cv.imwrite(save, final_matrix)    
 #   show(savepath)

def norm(name,maximum):
    filelist = os.listdir(rootdir)
    for filename in filelist:
        if filename[:-7] == name and filename[-1] == "g":
            filename1 = filename[:-3] + 'txt'
            f = open(os.path.join(filedir,filename1),'r') 
            lines = f.readlines()
            img = lines[0].strip().split(" ")
            img = list(map(float,img))
            for m in range(4):
                if m %2 == 0 :
                    maximum[m] = min(maximum[m],img[m])
                else :
                    maximum[m] = max (maximum[m],img[m])
          


if __name__ == '__main__':
    main()
          


    
