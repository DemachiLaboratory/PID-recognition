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
    show(rootdir)  #show all the labels on origin image 

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


            #divide every bounding box label file with the setting of cropped pictures as well as the situation of adaptation
            file = open(rootdir + "/" + filename[:-3]+'txt')
            lines = file.readlines()
            #every line in the original label files referring to one bounding box
            #there are 5 numbers in every line:
            #the class of object, the related coordinate of the target center (x,y), the related width and height of bounding box   
            for line in lines :
                #calculate the absolute coordinate of the original picture 
                #xmin and xmax refers to the minimum and maximum absolute x-axis coordinate of the bounding box respectively , the same as ymin and ymax
                char = line.strip().split(" ")
                char = list(map(float,char))
                x = char[1]*w
                y = char[2]*h
                xmin = (char[1]-char[3]/2)*w
                ymin = (char[2]- char[4]/2)*h
                xmax = (char[1]+char[3]/2)*w
                ymax = (char[2]+char[4]/2)*h   
                # decide whether every bounding box is subjected to all cropped pictures or not 
                for i in range(h_count) :                    
                    for j in range(w_count) :     
                        flag = 0
                        #imx_min and imy_min refers to the minimum (x,y) absolute coordinate of each cropped picture
                        #the maximum absolute coordinate is the minimum one adding the width or height 
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

                        # if the bounding box is intercept with cropped images, save the absolute coordinate of the bounding box
                        #一个点、两个点或者全都在框中 x1 y1 & x1 y2 & x2 y1 & x2 y2                    
                        if ((imx_max>xmin>imx_min) or (imx_min<xmax<imx_max)) and ((imy_max>ymin>imy_min) or (imy_min<ymax< imy_max)) : 
                            flag = 1 
                        #没有一个点在框中但是整个框过这个图片 x1 x2 & y1 y2
                        elif  (xmin>imx_min) and (xmax<imx_max) and  (ymin<imy_min) and (ymax>imy_max):
                            flag = 1 
                        elif  (ymin>imy_min) and (ymax< imy_max) and (xmin<imx_min) and (xmax>imx_max):
                            flag = 1 
                        if flag == 1 :
                            f = open(savepath+'/'+filename[:-4]+'-'+str(i)+str(j)+'v1.txt','a+') 
                            char[0] = int(char[0])
                            char[1] = xmin
                            char[2] = xmax
                            char[3] = ymin
                            char[4] = ymax
                            char = list(map(str,char))                      
                            f.write(' '.join(char))
                            f.write('\n')
                            f.close()

            #then process the pictrue cropped size to adaptation according to the adapted labels
            filelist1 = os.listdir(savepath)
            for filename1 in filelist1:
                if filename1[-1] == "t" :
                    print(filename1)
                    #find out the xmin ymin and xmax ymin  of every cropped pictures
                    maximum = np.zeros(5)
                    #calculate the orginal setting of every cropped picture
                    i = int(filename1[-8])
                    j = int(filename1[-7])
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

                    #initialize the supposed cropped size of picture 
                    maximum[1] = imx_min 
                    maximum[2] = imx_min + w_after
                    maximum[3] = imy_min 
                    maximum[4] = imy_min + h_after

                    #calculate the adapted cropped size compared to the setting one 
                    #the result will be saved in maximum[]
                    #maximum[1] to maximum[4] refers to the minimum of x , maximum of x ,the minimum of y , maximum of y  
                    readrow(savepath,filename1,maximum)
                    ab_coordinate = list(map(str,int(maximum[1:])))
                    f = open(savedir +'/coordinate/'+filename[:-7]+'.txt','a+') 
                    f.write(' '.join(char))
                    f.write('\n')
                    f.close()                    
                    #after adopptation, the cropped picture width and height are calculated as w_after_adapted and h_after_adapted
                    w_after_adapted = maximum[2]-maximum[1]
                    h_after_adapted = maximum[4]-maximum[3]

                    # then do the normalization for every line in label file and save all into the adapted fold 
                    file1 = open(savepath + "/" + filename1)                    
                    lines1 = file1.readlines()

                    if not os.path.isdir(savepath + '/adapted/'):
                        os.makedirs(savepath + '/adapted/')  
                
                    for line in lines1 :
                        char = line.strip().split(" ")
                        char = list(map(float,char))
                        char1 = np.zeros(5)
                        char1[0] = int(char[0])
                        char1[1] = ((char[1]+char[2])/2 - maximum[1])/w_after_adapted
                        char1[2] = ((char[3]+char[4])/2 - maximum[3])/h_after_adapted
                        char1[3] = (char[2]-char[1])/w_after_adapted
                        char1[4] = (char[4]-char[3])/h_after_adapted
                        char = list(map(str,char1))
                        f = open(savepath +'/adapted/' +filename1[:-6]+'.txt','a+') 
                        f.write(' '.join(char))
                        f.write('\n')
                        f.close()
                    file1.close()
    
            
                    #split up every picture into the required size and save 
                    image1 = cv.imread(savepath + '/enlarge/'+filename1[:-9]+'.jpg')
                    cropped = image1[int(maximum[3]):int(maximum[4]),int(maximum[1]):int(maximum[2])]
                    save_dir = savepath + '/adapted/'+filename1[:-6]+".jpg"
                    cv.imwrite(save_dir, cropped)
    show(savepath +'/adapted/') #show the output of afapted images  
          



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

                   
    