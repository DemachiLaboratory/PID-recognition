import os
import cv2 as cv
import argparse
import numpy as np
import math
import shutil


rootdir = "/mnt/data/datasets/PID_YOLO"  #images+labels acquire from 
savepath = "/mnt/data/datasets/PID_YOLO/divide"  # images+labels save in 
#rootdir = "D:/Download/PID_YOLO"
#savepath = "D:/Download/PID_YOLO/train"

#the pre-set final size for task w-width h-height , for P&ID maps : split into 1280*1280, origin is 4961*3580
w_after = 1280 
h_after = 1280
ratio = 0.15 #interception ration 

#first calculate as setting , and then adapt to different size of cropped images

def main():
    #clear the last time output and make the save file folder 
    if os.path.isdir(savepath):
        clear(savepath) 
    os.makedirs(savepath) 
#    show(rootdir)  #show all the labels on origin image 

    #list from the original image
    filelist = os.listdir(rootdir)  
    for filename in filelist:   
        if filename[-1] == "g" :
            image = cv.imread(os.path.join(rootdir, filename))
            print(filename)
            #original information of images  
            size = image.shape
            w = size [1]
            h = size [0]     
            #enlarge the size of orgin images to ensure the feasibility of picture cropping 
            # w_actual h_actual refers to the actual number which orgin image can split horizontally and vertically respectively 
            # w_count  h_count refers to the rounded up result of the w_actual and h_actual respectively  
            w_count = math.ceil(((w-w_after)/((1-ratio)*w_after)) + 1 )
            w_actual = ((w-w_after)/((1-ratio)*w_after)) + 1 
            h_count = math.ceil(((h-h_after)/((1-ratio)*h_after)) + 1 ) 
            h_actual = ((h-h_after)/((1-ratio)*h_after)) + 1 
            if w_count > w_actual or h_count > h_actual:
                image  = cv.copyMakeBorder(image,0, int(h_after*h_count-h-ratio*h_after*(h_count-1)),
                                           0, int(w_after*w_count-w-ratio*w_after*(w_count-1)),cv.BORDER_CONSTANT,value=0)
            
             #save the enlarged image into the enlarge folder 
                save = savepath + '/enlarge/' + filename
                if not os.path.isdir(savepath + '/enlarge/'):
                    os.makedirs(savepath + '/enlarge/')
                cv.imwrite(save, image)
			#loop in turns of cropped images name, open the original label file one time  
            for i in range(h_count) :                    
                for j in range(w_count) :
                    f = open(savepath +'/' +filename[:-4]+'-'+str(i)+str(j)+'.txt','a+') 
                    f.close()
                    maximum = np.zeros(5)
                    #imx_min and imy_min refers to minimum vertex (x,y) absolute coordinate of each cropped picture
                    #the maximum vertex absolute coordinate is the minimum one adding the width or height 
                    if i == 0 and j == 0:
                        imx_min=0
                        imy_min=0

                    elif i == 0 and j != 0:
                        imx_min=(1-ratio)*w_after*j
                        imy_min=0

                    elif i != 0 and j == 0 :
                        imx_min=0
                        imy_min=(1-ratio)*h_after*i

                    else :
                        imx_min=(1-ratio)*w_after*j
                        imy_min=(1-ratio)*h_after*i

                    imx_max = imx_min + w_after
                    imy_max = imy_min + h_after

                    maximum[1] = imx_min 
                    maximum[2] = imx_max
                    maximum[3] = imy_min 
                    maximum[4] = imy_max    

                    file = open(rootdir + "/" + filename[:-3]+'txt')
                    lines = file.readlines()
                    #every line in the original label files referring to one bounding box
                    #there are 5 numbers in every line:
                    #the class of object, the related coordinate of the target center (x,y), the related width and height of bounding box   
                    for line in lines :
                        flag = 0
                        #calculate the absolute coordinate of bounding box in original picture 
                        #xmin and xmax refers to the minimum and maximum absolute x-axis coordinate of the bounding box respectively 
                        #the same as ymin and ymax
                        char = line.strip().split(" ")
                        char = list(map(float,char))
                        xmin = (char[1]-char[3]/2)*w
                        ymin = (char[2]- char[4]/2)*h
                        xmax = (char[1]+char[3]/2)*w
                        ymax = (char[2]+char[4]/2)*h   
                        #decide whether every bounding box is subjected to all cropped pictures or not     
                        #there are two situations: 
                        
                        #one vertex , two vertices and four vertices of the bounding box in the cropped image 
                        # x1 y1 & x1 y2 & x2 y1 & x2 y2                    
                        if ((imx_max>xmin>imx_min) or (imx_min<xmax<imx_max)) and ((imy_max>ymin>imy_min) or (imy_min<ymax< imy_max)) : 
                            flag = 1 

                        #no vertex in the cropped image but still the bounding box through the image
                        # x1 x2 & y1 y2
                        elif  (xmin>imx_min) and (xmax<imx_max) and  (ymin<imy_min) and (ymax>imy_max):
                            flag = 1 
                        elif  (ymin>imy_min) and (ymax< imy_max) and (xmin<imx_min) and (xmax>imx_max):
                            flag = 1 
                        
                        #find out max and mini coordinate of x&y in subsjected bounding box for adaptation
                        if flag == 1 :
                            char[1] = xmin
                            char[2] = xmax
                            char[3] = ymin
                            char[4] = ymax
                            for m in range(5) :
                                if m%2 == 0 :
                                    maximum [m] = max(maximum[m],char[m])
                                else :
                                    maximum [m] = min(maximum[m],char[m])                                   
                    w_after_adapted = maximum[2]-maximum[1]
                    h_after_adapted = maximum[4]-maximum[3]

                    #split up every picture into the required adapted size and save 
                    cropped= image[int(maximum[3]):int(maximum[4]),int(maximum[1]):int(maximum[2])]
                    save_dir = savepath +'/'+filename[:-4]+'-'+str(i)+str(j)+".jpg"
                    cv.imwrite(save_dir, cropped)
                        
                    #save the coordinate information of adapted images for post-process
                    abs_coordinate = list(map(str,maximum[1:]))
                    if not os.path.isdir(savepath + '/coordinate/'):
                        os.makedirs(savepath + '/coordinate/')
                    f = open(savepath +'/coordinate/'+filename[:-4]+'-'+str(i)+str(j)+'.txt','a+') 
                    f.write(' '.join(abs_coordinate))
                    f.write('\n')
                    f.close()  
                        
                    #save the cropped labels information according to adaptation information 
                    for line in lines :
                        #calculate the absolute coordinate of the original picture 
                        #xmin and xmax refers to the minimum and maximum absolute x-axis coordinate of the bounding box respectively
                        #the same as ymin and ymax
                        char = line.strip().split(" ")
                        char = list(map(float,char))
                        x = char[1]*w
                        y = char[2]*h
                        xmin = (char[1]-char[3]/2)*w
                        ymin = (char[2]- char[4]/2)*h
                        xmax = (char[1]+char[3]/2)*w
                        ymax = (char[2]+char[4]/2)*h   

                        #decide whether every bounding box is subjected to this cropped pictures or not
                        #there are two sitiations:
                            
                        #if the four vertices of bounding box all included
                        #save the coordinate directly                   
                        if xmin>maximum[1] and xmax<maximum[2] and ymin>maximum[3] and ymax<maximum[4]:
                            flag = 1 
                            
                        #if part of vertices included, and others are not
                        #reset the excluded coordinate to ensure all including
                        elif ((maximum[2]>xmin>maximum[1]) or (maximum[1]<xmax<maximum[2])) and ((maximum[4]>ymin>maximum[3]) or (maximum[3]<ymax< maximum[4])) : 
                            xmin = max(maximum[1],xmin) 
                            xmax = min(maximum[2],xmax)
                            ymin = max(maximum[3],ymin) 
                            ymax = min(maximum[4],ymax)
                            flag = 1 
                        #change the labels into the related one for cropped image
                        if flag == 1 :
                            char[0] = int(char[0])
                            char[1] = ((xmin+xmax)/2 - maximum[1])/w_after_adapted
                            char[2] = ((ymin+ymax)/2 - maximum[3])/h_after_adapted
                            char[3] = (xmax-xmin)/w_after_adapted
                            char[4] = (ymax-ymin)/h_after_adapted 
                            char = list(map(str,char))
                            f = open(savepath +'/' +filename[:-4]+'-'+str(i)+str(j)+'.txt','a+') 
                            f.write(' '.join(char))
                            f.write('\n')
                            f.close()
                    file.close()
    
    show(savepath) #show the output of afapted images  
          

#show the picture with labels and save the output picture

def show(rootdir):
    filelist = os.listdir(rootdir) 
    savepath = rootdir + '/show/'
    if not os.path.isdir(rootdir + '/show/'):
        os.makedirs(savepath)
    #draw the bounding boxes on crossponding image
    for filename in filelist:   
        if filename[-1] == "g" :
            image = cv.imread(os.path.join(rootdir, filename))
            size = image.shape
            w = size [1]
            h = size [0]
            #calculate the absolute coordinate of bounding box on cropped images
            file = open(rootdir + "/" + filename[:-3]+'txt')
            lines = file.readlines()
            for line in lines:
                char = line.strip().split(" ")
                char = list(map(float,char))
                xmin = int((char[1]-char[3]/2)*w)
                ymin = int((char[2]- char[4]/2)*h)
                xmax = int((char[1]+char[3]/2)*w)
                ymax = int((char[2]+char[4]/2)*h)
                rec = cv.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                save = savepath + filename
                cv.imwrite(save, rec)

                
#clear all the former output files
def clear(savepath):
    shutil.rmtree(savepath)

# read file as rows to find out the size after adaptation
def readrow(savepath,filename,maximum):
    file= open(savepath + "/" + filename)       
    lines = file.readlines()
    for line in lines :
        char = line.strip().split(" ")#tab split the number apart
        char = list(map(float,char))
        for i in range(5) :
            if i%2 == 0 :
                maximum [i] = max(maximum[i],char[i])
            else :
                maximum [i] = min(maximum[i],char[i])                   
    file.close()
       


if __name__ == '__main__':
    main()

                   
    
