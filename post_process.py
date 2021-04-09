import os
import cv2 as cv
import argparse
import numpy as np
import math
import linecache
from pre_process import show,clear


rootdir = "D:/Download/PID_YOLO/train"
savepath = "D:/Download/PID_YOLO/train/return"
#rootdir = "/mnt/data/datasets/PID_YOLO/divide/adapted"  #images+labels acquire from 
#savepath = "/mnt/data/datasets/PID_YOLO/divide/adapted/return"  # images+labels save in 
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
    show(savepath)

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
          


    







'''
# combine with YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        #在每层输出上循环
        for output in layerOutputs:
            # 对每个检测进行循环
            for detection in output:

                scores = detection[5:]

                classID = np.argmax(scores)
                confidence = scores[classID]

                #过滤掉那些置信度较小的检测结果
                if confidence > 0.9:
                    #框后接框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    #边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                # 批量检测图片注意此处的boxes在每一次遍历的时候要初始化，否则检测出来的图像框会叠加
                    boxes.append([x, y, int(width), int(height)])
                    #print(boxes)
                    confidences.append(float(confidence))
                    #print(confidences)
                    classIDs.append(classID)
        print('boxes:',boxes)
        print('confidences:',confidences)
        print(type(boxes),type(confidences))
        # 极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
        k = -1
        if len(idxs) > 0:
            # for k in range(0,len(boxes)):
            for i in idxs.flatten() :
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # 在原图上绘制边框和类别
                color = [int(c) for c in COLORS[classIDs[i]]]
                # image是原图，     左上点坐标， 右下点坐标， 颜色， 画线的宽度
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
                print('type:',LABELS[classIDs[i]])
                savepath = "F:/Python/ModelArts/image/output/test2"  # 图像保存地址
                savepath=savepath+'/'+LABELS[classIDs[i]]
                # 如果输出的文件夹不存在，创建即可
                if not os.path.isdir(savepath):
                    os.makedirs(savepath)
                # 各参数依次是：图片，添加的文字，左上角坐标(整数)，字体，        字体大小，颜色，字体粗细
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                # 图像裁剪注意坐标要一一对应
                # 图片裁剪 裁剪区域【Ly:Ry,Lx:Rx】
                cut = image[y:(y+h), x:(x + w)]
                #print(type(cut))
                if cut.size != 0:
            # boxes的长度即为识别出来的车辆个数，利用boxes的长度来定义裁剪后车辆的路径名称
                    if k < len(boxes):
                        k = k+1
                # 从字母a开始每次+1
                    t = chr(ord("a")+k)
                    
                    print(filename)
                    print(filename.split(".")[0]+"_"+t+".jpg") 
                cv2.imwrite(savepath+"/"+filename.split(".")[0]+"_"+t+".jpg",cut)

'''