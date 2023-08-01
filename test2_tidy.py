
import sys
a = sys.version
print(a)
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
from scipy import stats

def seconds_to_hms(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)
 
def detection(img): #in: image, out: row for each obj (xmin    ymin    xmax   ymax  confidence  class    name)
    result = model(img)
    dataframe = result.pandas().xyxy[0]

    return dataframe

def shift5(arr, num, fill_value=np.zeros):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value(arr.shape[1])
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value(arr.shape[1])
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def find_actions(video_location,fps_param):
    #fps_param = 10
    
    model = torch.hub.load('C:/Users/Alex/yolov5', 'custom', source ='local', path='C:/Users/Alex/yolov5/runs/train/exp_3class_14/best.pt',force_reload=True)
    cap = cv2.VideoCapture('videos/fight_margaret.wmv')#warehouse.mp4')
    frame_counter = 0
    general_df = pd.DataFrame()
    while True:
    
        img = cap.read()[1]
        if img is None:
            break
        if frame_counter%fps_param == 0:

            result = model(img)
            df = result.pandas().xyxy[0] #df for each frame, row for each obj (xmin    ymin    xmax   ymax  confidence  class    name)


            for ind in df.index:
                x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
                x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
                label = df['name'][ind]
                conf = df['confidence'][ind]
                text = label + ' ' + str(conf.round(decimals= 2))
                
                #memory 

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)


            #Center dots function
            disable_background = False
            enable_center = True


            if disable_background == True:

                h, w, c = img.shape

                img = 255 * np.ones((h, w, 3), dtype = np.uint8)

            if enable_center == True:
                for ind in df.index:
                    x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
                    x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
                    #x1 + (x2-x1)/2,y1 + (y2-y1)/2

                    img = cv2.circle(img, (int(x1 + (x2-x1)/2),int(y1 + (y2-y1)/2)), radius=5, color=(0, 0, 255), thickness=-1)#object center dots on white screen


            df['time'] = seconds_to_hms(frame_counter/fps)#add time column
            general_df = pd.concat([general_df,df])


            #cv2.imshow('Video',img)
            cv2.imshow('Video',img)
            cv2.waitKey(10)
        frame_counter += 1



    return df

# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
#model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model = torch.hub.load('C:/Users/Alex/yolov5', 'custom', source ='local', path='C:/Users/Alex/yolov5/runs/train/exp_3class_14/best.pt',force_reload=True)
#cap = cv2.VideoCapture('videos/fight_margaret.wmv')#warehouse.mp4')
cap = cv2.VideoCapture('videos/yard.mp4')#warehouse.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
print("fps = ", fps)

frame_counter = 0
fps_param = 10 #detect once per fps_param frames
window_size = 9 #how many detections in a window
num_of_classes = 3
general_df = pd.DataFrame()
window_mem = np.zeros((window_size,num_of_classes))
prev_scene_objects = np.zeros((num_of_classes))
scene_objects = np.zeros((num_of_classes))



while True:
    
    img = cap.read()[1]
    if img is None:
        break
    if frame_counter%fps_param == 0:

        result = model(img)
        df = result.pandas().xyxy[0] #df for each frame, row for each obj (xmin    ymin    xmax   ymax  confidence  class    name)
        
        window_mem = shift5(window_mem, -1)
        for ind in df.index:
            x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
            x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
            label = df['name'][ind]
            conf = df['confidence'][ind]
            text = label + ' ' + str(conf.round(decimals= 2))
            #print(type(df['class'][ind]))
            window_mem[window_size-1][int(df['class'][ind])] += 1
            
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        
        
        scene_objects = stats.mode(window_mem, axis = 0, keepdims=True)[0][0]
        
        for idx, value in np.ndenumerate(scene_objects):
            if value != prev_scene_objects[idx[0]]:
                cv2.putText(img, str(idx), (500, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)


        cv2.putText(img, str(scene_objects), (500, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        prev_scene_objects = np.copy(scene_objects)
        
        #print(stats.mode(window_mem, axis = 0, keepdims=True)[0])
        
        
        #Center dots function
        disable_background = False
        enable_center = True
        
        
        if disable_background == True:
            
            h, w, c = img.shape
            
            img = 255 * np.ones((h, w, 3), dtype = np.uint8)
            
        if enable_center == True:
            for ind in df.index:
                x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
                x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
                #x1 + (x2-x1)/2,y1 + (y2-y1)/2
                
                img = cv2.circle(img, (int(x1 + (x2-x1)/2),int(y1 + (y2-y1)/2)), radius=5, color=(0, 0, 255), thickness=-1)#object center dots on white screen

        
        df['time'] = seconds_to_hms(frame_counter/fps)#add time column
        general_df = pd.concat([general_df,df])
        

        #cv2.imshow('Video',img)
        cv2.imshow('Video',img)
        cv2.waitKey(100)
    frame_counter += 1
print(general_df.to_string())
cap.release()