import os
import cv2 as cv
import argparse
import numpy as np
import math
import linecache
from pre_process import show

rootdir = "/mnt/data/datasets/PID_YOLO/divide/adapted"  #images+labels acquire from 
savepath = "/mnt/data/datasets/PID_YOLO/divide/adapted/return"  # images+labels save in 
filedir = rootdir+"/coordinate"
class = 2  #read from the yaml in YOLO , here is just the setting

def main():
    if os.path.isdir(savepath):
        clear(savepath) 
    os.makedirs(savepath) 
    #First, read the absolute coordinate of cropped image
    filelist = os.listdir(rootdir) 
    for filename in filelist:  
        if filename[-1] == "g" :
            print(filename)
            i = int(filename1[-6])
            j = int(filename1[-5])
            filename1 = filename[-3:] + 'txt'
            f = open(os.path.join(filedir,filename1),'r') 
            lines = f.readlines()
            img = lines[0].strip().split(" ")
            imx_min = int(img[0])
            imx_max = int(img[1])
            imy_min = int(img[2])
            imy_max = int(img[3])
            w = imx_max - imx_min
            h = imy_max - imy_min
            f.close()

            #then, calculate the related coordinate of the whole orgin images
            f = open(os.path.join(rootdir,filename1),'r') 
            lines = f.readlines()
            for line in lines : 
                box = line.strip().split(" ")
                xmin = (box[1]-box[3]/2)*w
                ymin = (box[2]-box[4]/2)*h
                xmax = (box[1]+box[3]/2)*w
                ymax = (box[2]+box[4]/2)*h                  
                

                char[0] = ((xmin+xmax)/2 - maximum[1])/w_after_adapted
                char[1] = ((ymin+ymax)/2 - maximum[3])/h_after_adapted
                char[2] = (xmax-xmin)/w_after_adapted
                char[3] = (ymax-ymin)/h_after_adapted   

                #for this verision              
                #radomly set the confidence score of each bounding boxes(YOLO wii read from the)
                score = round(np.random.random,2)
                char[4] = score
                char[5] = int(box[0])

                #In YOLO
                #set the output of the final layer for every object is a vector named : det[x,y,w,h,scores[]]
#                confidence = det[4:]
#                classID = np.argmax(confidence)
#                score = confidence[classID]
#                char[4] = classID
#                char[5] = score

                char = list(map(str,char))
                filename_save = filename[:-7]+'.txt'
                file = open(os.path.join(savepath,filename_save),'a+') 
                file.write(' '.join(char))
                file.write('\n')

            f.close()                  

          
       
    # initiate the bounding boxes
        boxes = []
        confidences = []
        classIDs = []

def cal_absolute(i,j,w_after,h_after,w,h,char):
    #calculate the original absolute coordinate 
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
    imx = imx_min + w_after/2
    imy = imy_min + h_after/2  
    #then calculate the adapatated absolute coordinate 





def nms():

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