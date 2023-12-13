from re import I
import numpy as np
import cupy as cp
import cv2
import time
import os
import math
from tkinter import filedialog, simpledialog,messagebox,Label,Tk,Button,IntVar,Radiobutton
import json

# ==================================================== Numpy Methods ================================================================

def numpyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent):
    for row in range(ImgsRows):
        PoseDict = {
        "Up": 0,
        "Down": 0,
        "Left": 0,
        "Right": 0
        }
        output = imgs[ImgsCols*row]
        st = time.time()
        for col in range(1, ImgsCols):
            img1 = imgs[ImgsCols*row+col]
            
            w = img1.shape[1]
            
            tmpUP = np.zeros((PoseDict["Up"],w), np.uint8)
            tmpDOWN = np.zeros((PoseDict["Down"],w), np.uint8)
            img1 = cv2.vconcat([tmpDOWN,img1,tmpUP])
            
            h = img1.shape[0]
            
            img0_right_index = output.shape[1] - math.floor(w*overlapPercentageLR)
            img1_left_pixel = math.floor(w*(overlapPercentageLR/5))
           
            img0_right_cnt = output.shape[1] - img0_right_index
            x, align_img1, align_output = findHOverlapNotAlignIndex(output, img1, img1_left_pixel, 
                                                                      img0_right_cnt, math.floor(h*checkYAlignmentPercent),
                                                                      img0_right_index, PoseDict)
            OverlapXIndexList.append(int(x))
            tempOut = cv2.hconcat([align_output[:,:int(x)],align_img1])
            output = blendSeamlessCloneX(tempOut, align_output, x, align_output.shape[1])
         
        if row > 0:
            tmpLEFT = np.zeros((output.shape[0],PoseDictList[row-1]["Left"]), np.uint8)
            tmpRIGHT = np.zeros((output.shape[0],PoseDictList[row-1]["Right"]), np.uint8)
            output = cv2.hconcat([tmpRIGHT,output,tmpLEFT])
            
            PrevOutput_w = PrevOutput.shape[1]
            PrevOutput_h = PrevOutput.shape[0]
            output_w = output.shape[1]
            output_h = output.shape[0]
            
            diff = abs(output_w-PrevOutput_w)
            
            if PrevOutput_w > output_w:
                temp = np.zeros((output_h,diff), np.uint8)
                output = np.hstack([output,temp], np.uint8)
            else:
                temp = np.zeros((PrevOutput_h,diff), np.uint8)
                PrevOutput = np.hstack([PrevOutput,temp], np.uint8)
            
            img0_bottom_index = PrevOutput.shape[0] - math.floor(output.shape[0]*overlapPercentageTB)
            img1_top_pixel = math.floor(output.shape[0]*overlapPercentageTB/2)
        
            img0_bottom_cnt = PrevOutput.shape[0] - img0_bottom_index
            y, align_img1, align_output = findVOverlapNotAlignIndex(PrevOutput, output, 
                                                                      img1_top_pixel, img0_bottom_cnt, 
                                                                      math.floor(output.shape[1] * checkXAlignmentPercent), 
                                                                      img0_bottom_index, PoseDict)
            OverlapYIndexList.append(int(y))
            tempOut = cv2.vconcat([align_output[:y,:],align_img1])
            output = tempOut#blendSeamlessCloneY(tempOut, align_output, y, align_output.shape[0])
        
        PrevOutput = output
        PoseDictList.append(PoseDict)
        et = time.time()
        ProcessTimeList.append(et-st)
        st = time.time()
            
    return output
        
def findHOverlapIndex(output_right, img1_left, img1_left_pixel, img0_right_cnt):
    sqdif_arr = np.zeros(img0_right_cnt-img1_left_pixel, float)
    print("Finding X Overlap......")
    for x in range(img0_right_cnt - img1_left_pixel):
        diff = output_right[:,x:x+img1_left_pixel] - img1_left
        sum_sqdif = np.sum(diff*diff, dtype=np.int64)
        sqdif_arr[x] = sum_sqdif
    print()
    return np.where(sqdif_arr == sqdif_arr.min())[0][0]

def findHOverlapNotAlignIndex(output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index, PoseDict):
    output_right = output[PoseDict["Down"]:output.shape[0]-PoseDict["Up"],img0_right_index:]
    img1_left = img1[PoseDict["Down"]:img1.shape[0]-PoseDict["Up"],:img1_left_pixel]
    print("Start Check Y Align")
    sqdif_arr = np.zeros((check_y_align_pixel_cnt*2, 2), float)
    '''
    [y diff value, x -> the best overlap x position when in this y value]
    '''
    h = img1_left.shape[0]
    w = img1_left.shape[1]
    for j in range(check_y_align_pixel_cnt):
        print("Checking Y Align")
        img1DOWN = img1_left[:h-j,:] # cut down part (add black row at top) -> move down
        img1UP = img1_left[j:,:] # cut up part (add black row at bottom) -> move up
        if j > 0:
            temp = np.zeros((j,img1_left_pixel))
            img1DOWN = np.vstack([temp,img1DOWN])
            img1UP = np.vstack([img1UP,temp])
       
        xDOWN = findHOverlapIndex(output_right[j:,:], img1DOWN[j:,:], img1_left_pixel, img0_right_cnt)+img0_right_index
        xUP = findHOverlapIndex(output_right[:h-j,:], img1UP[:h-j,:], img1_left_pixel, img0_right_cnt)+img0_right_index
        
        diffDOWN = output_right[j:,xDOWN-img0_right_index:xDOWN-img0_right_index+img1_left_pixel] - img1DOWN[j:,:]
        diffUP = output_right[:h-j,xUP-img0_right_index:xUP-img0_right_index+img1_left_pixel] - img1UP[:h-j,:]
        
        sum_sqdifDOWN = np.sum(diffDOWN*diffDOWN)
        sum_sqdifUP = np.sum(diffUP*diffUP)
        
        sqdif_arr[j,0] = sum_sqdifDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,0] = sum_sqdifUP
        
        sqdif_arr[j,1] = xDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,1] = xUP
    
    index = int(np.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    
    if index >= check_y_align_pixel_cnt:
        temp1 = np.zeros((index-check_y_align_pixel_cnt,img1.shape[1]))
        temp2 = np.zeros((index-check_y_align_pixel_cnt,output.shape[1]))
        align_img1 = np.vstack([img1,temp1])
        align_output = np.vstack([temp2,output])
        PoseDict["Up"] += index-check_y_align_pixel_cnt
    else:
        temp1 = np.zeros((index,img1.shape[1]))
        temp2 = np.zeros((index,output.shape[1]))
        align_img1 = np.vstack([temp1,img1])
        align_output = np.vstack([output,temp2])
        PoseDict["Down"] += index
        
    return sqdif_arr[index,1], align_img1, align_output

def findVOverlapIndex(output_bottom, img1_top, img1_top_pixel, img0_bottom_cnt):
    sqdif_arr = np.zeros(img0_bottom_cnt-img1_top_pixel, float)
    print("Finding Y Overlap......")
    for y in range(img0_bottom_cnt - img1_top_pixel):
        diff = output_bottom[y:y+img1_top_pixel,:] - img1_top
        sum_sqdif = np.sum(diff*diff, dtype=np.int64)
        sqdif_arr[y] = sum_sqdif
    print()
    return np.where(sqdif_arr == sqdif_arr.min())[0][0]

def findVOverlapNotAlignIndex(output, img1, img1_top_pixel, img0_bottom_cnt, check_x_align_pixel_cnt, img0_bottom_index, PoseDict):
    output_bottom = output[img0_bottom_index:,PoseDict["Right"]:output.shape[1]-PoseDict["Left"]]
    img1_top = img1[:img1_top_pixel, PoseDict["Right"]:img1.shape[1]-PoseDict["Left"]]
    print("Start Check X Align")
    sqdif_arr = np.zeros((check_x_align_pixel_cnt*2, 2), float)
    '''
    [x diff value, y -> the best overlap y position when in this x value]
    '''
    h = img1_top.shape[0]
    w = img1_top.shape[1]
    for j in range(check_x_align_pixel_cnt):
        print("Checking X Align")
        img1RIGHT = img1_top[:,:w-j] # cut right part (add black row at left) -> move right
        img1LEFT = img1_top[:,j:] # cut left part (add black row at right) -> move left
        if j > 0:
            temp = np.zeros((img1_top_pixel,j),dtype=np.uint8)
            img1RIGHT = np.hstack([temp,img1RIGHT])
            img1LEFT = np.hstack([img1LEFT,temp])
       
        yRIGHT = findVOverlapIndex(output_bottom[:,j:], img1RIGHT[:,j:], img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        yLEFT = findVOverlapIndex(output_bottom[:,:w-j], img1LEFT[:,:w-j], img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        difRIGHT = output_bottom[yRIGHT-img0_bottom_index:yRIGHT-img0_bottom_index+img1_top_pixel,j:] - img1RIGHT[:,j:]
        diffLEFT = output_bottom[yLEFT-img0_bottom_index:yLEFT-img0_bottom_index+img1_top_pixel,:w-j] - img1LEFT[:,:w-j]
        
        sum_sqdifRIGHT = np.sum(difRIGHT*difRIGHT)
        sum_sqdifLEFT = np.sum(diffLEFT*diffLEFT)
        
        sqdif_arr[j,0] = sum_sqdifRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,0] = sum_sqdifLEFT
        
        sqdif_arr[j,1] = yRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,1] = yLEFT
    
    index = int(np.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    
    if index >= check_x_align_pixel_cnt:
        temp1 = np.zeros((img1.shape[0], index-check_x_align_pixel_cnt),dtype=np.uint8)
        temp2 = np.zeros((output.shape[0], index-check_x_align_pixel_cnt),dtype=np.uint8)
        align_img1 = np.hstack([img1,temp1])
        align_output = np.hstack([temp2,output])
        PoseDict["Left"] += index-check_x_align_pixel_cnt
    else:
        temp1 = np.zeros((img1.shape[0],index),dtype=np.uint8)
        temp2 = np.zeros((output.shape[0],index),dtype=np.uint8)
        align_img1 = np.hstack([temp1,img1])
        align_output = np.hstack([output,temp2])
        PoseDict["Right"] += index
        
    return sqdif_arr[index,1], align_img1, align_output

def blendSeamlessCloneX(tempOut, align_output, x, w):
    width_to_blend = math.floor(w*0.02)
    if x-width_to_blend < 0:
        width_to_blend = x;
    if x+width_to_blend >= w:
        width_to_blend = w-x;
    src = align_output[:,int(x)-width_to_blend:int(x)+width_to_blend]
    src = align_output[:,int(x)-100:int(x)+100]
    mask = 255*np.ones(src.shape, np.uint8)
    tempOut = tempOut.astype('uint8')
    src = src.astype('uint8')
    tempOut = cv2.cvtColor(tempOut,cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    output_img = cv2.seamlessClone(src,tempOut,mask,(int(x),tempOut.shape[0]//2),cv2.NORMAL_CLONE) #cv2.MONOCHROME_TRANSFER also can
    output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
    return output_img

def blendSeamlessCloneY(tempOut, align_output, y, h):
    height_to_blend = math.floor(h*0.1)
    if h-height_to_blend < 0:
        height_to_blend = y;
    if y+height_to_blend >= h:
        height_to_blend = h-y;
    src = align_output[int(y)-height_to_blend:int(y)+height_to_blend,:]
    mask = 255*np.ones(src.shape, np.uint8)
    tempOut = tempOut.astype('uint8')
    src = src.astype('uint8')
    tempOut = cv2.cvtColor(tempOut,cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    output_img = cv2.seamlessClone(src,tempOut,mask,(tempOut.shape[1]//2, int(y)),cv2.NORMAL_CLONE) #cv2.MONOCHROME_TRANSFER also can
    output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
    return output_img

# ==================================================== Cupy Methods =================================================================
def cupyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent):
    global processedRow
    global processedCol
    processedRow = 0
    for row in range(ImgsRows):
        PoseDict = {
        "Up": 0,
        "Down": 0,
        "Left": 0,
        "Right": 0
        }
        output =  cp.array(imgs[ImgsCols*row])
        st = time.time()
        processedCol = 0
        for col in range(1, ImgsCols):
            img1 = cp.array(imgs[ImgsCols*row+col])
            w = img1.shape[1]
            
            tmpUP = cp.zeros((PoseDict["Up"],w))
            tmpDOWN = cp.zeros((PoseDict["Down"],w))
            img1 = cp.vstack([tmpDOWN,img1,tmpUP])
            
            h = img1.shape[0]
            
            #img0_right_index = output.shape[1] - math.floor(w*overlapPercentageLR)
            img1_left_pixel = math.floor(w*(overlapPercentageLR/5))
            value = math.floor(w*overlapPercentageLR)
            img0_right_cnt = value if output.shape[1] - img1_left_pixel >= value else output.shape[1]
            img0_right_index = output.shape[1] - img0_right_cnt
            
            x, align_img1, align_output, movementY = findHOverlapNotAlignIndexCu(output, img1, img1_left_pixel, 
                                                                      img0_right_cnt, math.floor(h*checkYAlignmentPercent),
                                                                      img0_right_index, PoseDict)
            OverlapXIndexList.append(int(x))
            #tempOut = cp.hstack([align_output[:,:int(x)],align_img1])
            #output = blendSeamlessCloneXCu(tempOut, align_output, int(x), w, movementY, PoseDict)
            output = blendAlphaXCu(align_img1, align_output,x,w,movementY,PoseDict)
            
            processedCol+=1
            
        if row > 0:
            tmpLEFT = cp.zeros((output.shape[0],PoseDictList[row-1]["Left"]))
            tmpRIGHT = cp.zeros((output.shape[0],PoseDictList[row-1]["Right"]))
            output = cp.hstack([tmpRIGHT,output,tmpLEFT])
            
            PrevOutput_w = PrevOutput.shape[1]
            PrevOutput_h = PrevOutput.shape[0]
            output_w = output.shape[1]
            output_h = output.shape[0]
            
            diff = abs(output_w-PrevOutput_w)
            
            if PrevOutput_w > output_w:
                temp = cp.zeros((output_h,diff))
                output = cp.hstack([output,temp])
            else:
                temp = cp.zeros((PrevOutput_h,diff))
                PrevOutput = cp.hstack([PrevOutput,temp])
            
            img1_top_pixel = math.floor(output.shape[0]*overlapPercentageTB/5)
            value = math.floor(output.shape[0]*overlapPercentageTB)
            img0_bottom_cnt = value if PrevOutput.shape[0] - img1_top_pixel >= value else PrevOutput.shape[0]
            img0_bottom_index = PrevOutput.shape[0] - img0_bottom_cnt
            
            y, align_img1, align_output, movementX = findVOverlapNotAlignIndexCu(PrevOutput, output, 
                                                                      img1_top_pixel, img0_bottom_cnt, 
                                                                      math.floor(output.shape[1] * checkXAlignmentPercent), 
                                                                      img0_bottom_index, PoseDict)
            OverlapYIndexList.append(int(y))
            output = blendAlphaYCu(align_img1, align_output, y, output.shape[0], movementX, PoseDict, PoseDictList[row-1])
            #tempOut = cp.vstack([align_output[:y,:],align_img1])
            #output = blendSeamlessCloneYCu(tempOut, align_output, y, h, movementX, PoseDict)
        
        PrevOutput = output
        PoseDictList.append(PoseDict)
        et = time.time()
        ProcessTimeList.append(et-st)
        st = time.time()
        processedRow += 1
            
    return output

def findHOverlapIndexCu(output_right, img1_left, img1_left_pixel, img0_right_cnt, min_overlap_count = 0):
    sqdif_arr = cp.zeros(img0_right_cnt-img1_left_pixel-min_overlap_count, float)
    shape = img1_left.shape
    print("Finding Overlap using Cuda......")
    for x in range(min_overlap_count, img0_right_cnt - img1_left_pixel):
        #diff = output_right[:,x:x+img1_left_pixel] - img1_left
        #sum_sqdif = cp.sum(diff*diff)
        diffT = output_right[:math.floor(shape[0]*0.2),x:x+img1_left_pixel] - img1_left[:math.floor(shape[0]*0.2),:]
        diffM = output_right[int(shape[0]//2 - shape[0]*0.1):int(shape[0]//2 + shape[0]*0.1),x:x+img1_left_pixel] - img1_left[int(shape[0]//2 - shape[0]*0.1):int(shape[0]//2 + shape[0]*0.1),:]
        diffB = output_right[math.floor(shape[0] - shape[0]*0.2):,x:x+img1_left_pixel] - img1_left[math.floor(shape[0] - shape[0]*0.2):,:]
        sum_sqdifT = cp.sum(diffT*diffT)
        sum_sqdifM = cp.sum(diffM*diffM)
        sum_sqdifB = cp.sum(diffB*diffB)
        sum_sqdif = sum_sqdifT + sum_sqdifB + sum_sqdifM
        sqdif_arr[x-min_overlap_count] = sum_sqdif
    
    print()
    return int(cp.where(sqdif_arr == sqdif_arr.min())[0][0]) + min_overlap_count

def findHOverlapNotAlignIndexCu(output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index, PoseDict):
    output_right = output[PoseDict["Down"]:output.shape[0]-PoseDict["Up"],img0_right_index:]
    img1_left = img1[PoseDict["Down"]:img1.shape[0]-PoseDict["Up"],:img1_left_pixel]
    print("Start Check Align using Cuda")
    sqdif_arr = cp.zeros((check_y_align_pixel_cnt*2, 2), float)
    '''
    [y diff value, x -> the best overlap x position when in this y value]
    '''
    h = img1_left.shape[0]
    for j in range(check_y_align_pixel_cnt):
        print("Checking Align using Cuda")
        img1DOWN = img1_left[:h-j,:] # cut down part (add black row at top) -> move down
        img1UP = img1_left[j:,:] # cut up part (add black row at bottom) -> move up
        if j > 0:
            temp = cp.zeros((j,img1_left_pixel))
            img1DOWN = cp.vstack([temp,img1DOWN])
            img1UP = cp.vstack([img1UP,temp])
       
        xDOWN = findHOverlapIndexCu(output_right[j:,:], img1DOWN[j:,:], img1_left_pixel, img0_right_cnt, math.floor(img1.shape[1]*minOverlap))+img0_right_index
        xUP = findHOverlapIndexCu(output_right[:h-j,:], img1UP[:h-j,:], img1_left_pixel, img0_right_cnt, math.floor(img1.shape[1]*minOverlap))+img0_right_index
        '''
        s = time.time()
        #diffDOWN = output_right[j:j+math.floor(h*0.2),xDOWN-img0_right_index:xDOWN-img0_right_index+img1_left_pixel] - img1DOWN[j:j+math.floor(h*0.2),:]
        #diffUP = output_right[h - math.floor(h*0.2) - j:h-j,xUP-img0_right_index:xUP-img0_right_index+img1_left_pixel] - img1UP[h - math.floor(h*0.2) - j:h-j,:]
        
        sum_sqdifDOWN = cp.sum(diffDOWN*diffDOWN)
        sum_sqdifUP = cp.sum(diffUP*diffUP)
        
        test+=time.time()-s'''
        
        
        diffDOWN = output_right[j:,xDOWN-img0_right_index:xDOWN-img0_right_index+img1_left_pixel] - img1DOWN[j:,:]
        diffUP = output_right[:h-j,xUP-img0_right_index:xUP-img0_right_index+img1_left_pixel] - img1UP[:h-j,:]
        sum_sqdifDOWN = cp.sum(diffDOWN*diffDOWN)
        sum_sqdifUP = cp.sum(diffUP*diffUP)
        
        sqdif_arr[j,0] = sum_sqdifDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,0] = sum_sqdifUP
        
        sqdif_arr[j,1] = xDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,1] = xUP
    
    index = int(cp.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    
    movement = []
    if index >= check_y_align_pixel_cnt:
        v = index-check_y_align_pixel_cnt
        temp1 = cp.zeros((v,img1.shape[1]))
        temp2 = cp.zeros((v,output.shape[1]))
        align_img1 = cp.vstack([img1,temp1])
        align_output = cp.vstack([temp2,output])
        PoseDict["Up"] += v
        movement.append("Up")
        movement.append(v)
    else:
        temp1 = cp.zeros((index,img1.shape[1]))
        temp2 = cp.zeros((index,output.shape[1]))
        align_img1 = cp.vstack([temp1,img1])
        align_output = cp.vstack([output,temp2])
        PoseDict["Down"] += index
        movement.append("Down")
        movement.append(index)
        
    return sqdif_arr[index,1], align_img1, align_output, movement

def findVOverlapIndexCu(output_bottom, img1_top, img1_top_pixel, img0_bottom_cnt, min_overlap_count = 0):
    sqdif_arr = cp.zeros(img0_bottom_cnt-img1_top_pixel)
    print("Finding Y Overlap using Cuda......")
    shape = img1_top.shape
    value = math.floor(shape[1]*0.2)
    if value > 10000:
        value = 10000
    for y in range(min_overlap_count, img0_bottom_cnt - img1_top_pixel):
        '''
        #diff = output_bottom[y:y+img1_top_pixel,:] - img1_top
        sum_sqdif = cp.sum(diff*diff)
        '''
        diffL = output_bottom[y:y+img1_top_pixel,:value] - img1_top[:,:value]
        diffM = output_bottom[y:y+img1_top_pixel,shape[1]//2 - value//2:shape[1]//2 + value//2] - img1_top[:,shape[1]//2 - value//2:shape[1]//2 + value//2]
        diffR = output_bottom[y:y+img1_top_pixel, shape[1]-value:] - img1_top[:, shape[1]-value:]
        sum_sqdifL = cp.sum(diffL*diffL)
        sum_sqdifM = cp.sum(diffM*diffM)
        sum_sqdifR = cp.sum(diffR*diffR)
        sum_sqdif = sum_sqdifL+sum_sqdifM+sum_sqdifR
        sqdif_arr[y+min_overlap_count] = sum_sqdif
    print()
    return int(cp.where(sqdif_arr == sqdif_arr.min())[0][0])+min_overlap_count

def findVOverlapNotAlignIndexCu(output, img1, img1_top_pixel, img0_bottom_cnt, check_x_align_pixel_cnt, img0_bottom_index, PoseDict):
    output_bottom = equaliseCupyImage(output[img0_bottom_index:,PoseDict["Right"]:output.shape[1]-PoseDict["Left"]])
    img1_top = equaliseCupyImage(img1[:img1_top_pixel, PoseDict["Right"]:img1.shape[1]-PoseDict["Left"]])
    
    print(str(processedRow)+"-"+str(processedCol),": Start Check X Align")
    sqdif_arr = cp.zeros((check_x_align_pixel_cnt*2, 2), float)
    '''
    [x diff value, y -> the best overlap y position when in this x value]
    '''
    w = img1_top.shape[1]
    for j in range(check_x_align_pixel_cnt):
        print(str(processedRow)+"-"+str(processedCol),": Checking X Align")
        img1RIGHT = img1_top[:,:w-j] # cut right part (add black row at left) -> move right
        img1LEFT = img1_top[:,j:] # cut left part (add black row at right) -> move left
        if j > 0:
            temp = cp.zeros((img1_top_pixel,j))
            img1RIGHT = cp.hstack([temp,img1RIGHT])
            img1LEFT = cp.hstack([img1LEFT,temp])
            
        yRIGHT = findVOverlapIndexCu(output_bottom[:,j:], img1RIGHT[:,j:], img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        yLEFT = findVOverlapIndexCu(output_bottom[:,:w-j], img1LEFT[:,:w-j], img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        diffRIGHT = output_bottom[yRIGHT-img0_bottom_index:yRIGHT-img0_bottom_index+img1_top_pixel,j:] - img1RIGHT[:,j:]
        diffLEFT = output_bottom[yLEFT-img0_bottom_index:yLEFT-img0_bottom_index+img1_top_pixel,:w-j] - img1LEFT[:,:w-j]
        
        sum_sqdifRIGHT = cp.sum(diffRIGHT*diffRIGHT)
        sum_sqdifLEFT = cp.sum(diffLEFT*diffLEFT)
       
        
        sqdif_arr[j,0] = sum_sqdifRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,0] = sum_sqdifLEFT
        
        sqdif_arr[j,1] = yRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,1] = yLEFT
       
    index = int(cp.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    movement = []
    # Align by adding black pixel at left/right
    if index >= check_x_align_pixel_cnt:
        v = index-check_x_align_pixel_cnt
        temp1 = cp.zeros((img1.shape[0], v))
        temp2 = cp.zeros((output.shape[0], v))
        align_img1 = cp.hstack([img1,temp1])
        align_output = cp.hstack([temp2,output])
        PoseDict["Left"] += v
        movement.append("LEFT")
        movement.append(v)
    else:
        temp1 = cp.zeros((img1.shape[0],index))
        temp2 = cp.zeros((output.shape[0],index))
        align_img1 = cp.hstack([temp1,img1])
        align_output = cp.hstack([output,temp2])
        PoseDict["Right"] += index
        movement.append("Right")
        movement.append(index)
    
    return sqdif_arr[index,1], align_img1, align_output, movement

'''
SRC:
0      10      5       15      20    30  
0      10      15      30      20    35
       U       U       D       U     D
       10      5       15      5     15
       U10     U15     U15     U20   U20   
       D0      D0      D15     D15   D30	

U - WHEN U, U CURR + D TOTAL; WHEN D, D TOTAL
D - WHEN D, D CURR + U TOTAL; WHEN U, U TOTAL

'''
def blendSeamlessCloneXCu(tempOut, align_output, x, w, movementY, PoseDict):
    width_to_blend = math.floor(w*0.015)
    if movementY[0] == "UP":
        U = movementY[1] + PoseDict["Down"]
        D = align_output.shape[0] - PoseDict["UP"]
    else:
        U = PoseDict["Down"]
        D = align_output.shape[0] - movementY[1]

    if x-width_to_blend < 0:
        width_to_blend = x;
    if x+width_to_blend >= align_output.shape[1]:
        width_to_blend = align_output.shape[1]-x;
    
    src = align_output[U:D,int(x)-width_to_blend:int(x)+width_to_blend]
    mask = 255*np.ones(src.shape, np.uint8)
    tempOut = cp.asnumpy(tempOut)
    tempOut = tempOut.astype('uint8')
    src = cp.asnumpy(src)
    src = src.astype('uint8')
    tempOut = cv2.cvtColor(tempOut,cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    output_img = cv2.seamlessClone(src,tempOut,mask,(int(x),(U+D)//2),cv2.NORMAL_CLONE) #cv2.MONOCHROME_TRANSFER also can
    output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
    output_img = output_img.astype('float64')
    output = cp.array(output_img)
    return output

def blendSeamlessCloneYCu(tempOut, align_output, y, h, movementX, PoseDict):
    height_to_blend = math.floor(h*0.015)
    if movementX[0] == "Left":
        L = movementX[1] + PoseDict["Right"]
        R = align_output.shape[1] - PoseDict["Left"]
    else:
        L = PoseDict["Right"]
        R = align_output.shape[1] - movementX[1]    

    if h-height_to_blend < 0:
        height_to_blend = y;
    if y+height_to_blend >= align_output.shape[0]:
        height_to_blend = align_output.shape[0]-y;

    src = align_output[int(y)-height_to_blend:int(y)+height_to_blend,L:R]
    mask = 255*np.ones(src.shape, np.uint8)
    tempOut = cp.asnumpy(tempOut)
    tempOut = tempOut.astype('uint8')
    src = cp.asnumpy(src)
    src = src.astype('uint8')
    tempOut = cv2.cvtColor(tempOut,cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    output_img = cv2.seamlessClone(src,tempOut,mask,((L+R)//2, int(y)),cv2.NORMAL_CLONE) #cv2.MONOCHROME_TRANSFER also can
    output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
    output_img = output_img.astype('float64')
    output = cp.array(output_img)
    return output

def blendAlphaXCu(align_img1, align_output, x, w, movementY, PoseDict):
    width_to_blend = math.floor(w*0.1)
    if movementY[0] == "UP":
        U = movementY[1] + PoseDict["Down"]
        D = align_output.shape[0] - PoseDict["UP"]
    else:
        U = PoseDict["Down"]
        D = align_output.shape[0] - movementY[1]

    if x-width_to_blend < 0:
        width_to_blend = x;
    if x+width_to_blend >= align_output.shape[1]:
        width_to_blend = align_output.shape[1]-x;
    
    src = align_output[U:D,int(x):int(x)+width_to_blend]
    src_shape = src.shape
    
    target = align_img1[U:D,:src_shape[1]]
    target_shape = target.shape
    
    #mask1 = np.repeat(np.tile(np.linspace(1, 0, overlap1.shape[1]), (overlap1.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
    #mask2 = np.repeat(np.tile(np.linspace(0, 1, overlap2.shape[1]), (overlap2.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
    mask1 = cp.linspace(1, 0, src_shape[1])
    mask2 = 1-mask1
    final = src * mask1 + target * mask2
    
    output = cp.hstack([align_output[:,:x],align_img1])
    output[U:D,int(x):int(x)+src_shape[1]] = final
   
    return output

def blendAlphaYCu(align_img1, align_output, y, h, movementX, PoseDict, PrevPoseDict):
    height_to_blend = math.floor(h*0.1)
    ignorePrev = max(PoseDict["Down"],PoseDict["Up"])
    ignorePrev = ignorePrev if ignorePrev <= 10 else ignorePrev-10
    if movementX[0] == "Left":
        L = movementX[1] + PoseDict["Right"]
        R = align_output.shape[1] - PoseDict["Left"]
    else:
        L = PoseDict["Right"]
        R = align_output.shape[1] - movementX[1]

    if h-height_to_blend < 0:
        height_to_blend = y;
    if y+height_to_blend >= align_output.shape[0]:
        height_to_blend = align_output.shape[0]-y;
    
    src = align_output[int(y):int(y)+height_to_blend,L:R]
    src_shape = src.shape
    
    target = align_img1[:src_shape[0],L:R]
    
    mask1 = cp.swapaxes(cp.tile(cp.linspace(1, 0, src_shape[0]), (src_shape[1], 1)),0,1)
    mask1 = getBlackMask(src,mask1,ignorePrev,src_shape[0])
    mask2 = 1-mask1
    
    final = src * mask1 + target * mask2
    output = cp.vstack([align_output[:y,:],align_img1])
    output[int(y):int(y)+height_to_blend,L:R] = final
    return output

#  ======================================================== Others ==================================================================
def getBlackMask(img,mask,ignorePrev,h):
    '''
    _, alpha = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    alpha = np.zeros(alpha.shape, dtype=np.uint8)
    alpha = cv2.fillPoly(alpha,[contours[-1]],(255),1)
    '''
    alpha = img[h-ignorePrev:]<=1
    temp = mask[h-ignorePrev:]
    temp[alpha] = 0
    mask[h-ignorePrev:] = temp
    return mask
    
def prepareImages(path = None, gray = False):
    if path is None:
        selection = askSelection("Folder or Images?",["Folder","Images"],"Select Input Types",0)
        if selection is None:
            return None,None
        elif selection == "Folder":
            path = filedialog.askdirectory()
            if path == "":
                return None,None
        else:
            imgs_list = filedialog.askopenfilenames()
            path = imgs_list
            
        
    if selection == "Folder":    
        file_list = os.listdir(path)
        imgs_list = [
                    path+"\\"+f
                    for f in file_list
                    if os.path.isfile(os.path.join(path, f))
                    and f.lower().endswith((".png", ".gif",".jpg",".jpeg",".bmp"))
                ]
        
    imgs=[]
    for img in imgs_list:
        print("LOADING IMAGE: "+os.path.basename(img)+"......")
        if gray:
            imgs.append(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
        else:
            imgs.append(cv2.imread(img))
    return path, imgs

def imageResize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def equaliseCupyImage(image_gray):
    image_gray = cp.asnumpy(image_gray)
    image_gray = image_gray.astype('uint8')
    image_gray = clahe.apply(image_gray)
    image_gray = image_gray.astype('float64')
    return cp.asarray(image_gray)

def showCupyImage(name, img, height = 400, width = None):
    imgNUMPY = cp.asnumpy(img)
    imgNUMPY = imgNUMPY.astype('uint8')
    cv2.imshow(name, imageResize(imgNUMPY,width,height))

def askSelection(prompt, options,title=None, defaultSelectionIndex = 2):
    root = Tk()
    root.minsize(300,100) 

    if title:
        root.title(title)
    if prompt:
        Label(root, text=prompt).pack()
    
    v = IntVar()
    v.set(defaultSelectionIndex)
    for i, option in enumerate(options):
        Radiobutton(root, text=option, variable=v, value=i).pack(side="top",anchor="w")
    
    global status
    status = False
    def select():
        global status
        status = True
        root.destroy()
        
    Button(root, text="Confirm", command=select).pack(side="bottom",pady=5)
    
    root.mainloop()
    
    if status:
        return options[v.get()]
    else:
        return None    

def saveStitchedImage():
    f = filedialog.asksaveasfile(mode='w', 
                                 defaultextension=".png",
                                 filetypes=[("png file", ".png"),("bmp file", ".bmp")],
                                 initialfile="Stitched_"+"_".join(str(time.asctime().replace(":","")).split()))
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    if f:
        return f.name
    f.close()
    
def saveStitchedImageJSON(inputPath, imgPath, overlapPercentageLR, overlapPercentageTB, 
                          checkYAlignmentPercent, checkXAlignmentPercent, ImgsRows, ImgsCols, 
                          PoseDictList, OverlapXIndexList, OverlapYIndexList, ProcessTimeList, processTime):
    # Data to be written
    data = {
        "Input Path": inputPath,
        "Output Path": imgPath,
        "Left-Right Overlap %": overlapPercentageLR,
        "Top-Bottom Overlap %": overlapPercentageTB,
        "Check Y Alignment Percent": checkYAlignmentPercent,
        "Check X Alignment Percent": checkXAlignmentPercent,
        "Input Images Rows": ImgsRows,
        "Input Images Cols": ImgsCols,
        "Number of Pixel Shift Up": str([PoseDict["Up"] for PoseDict in PoseDictList]),
        "Number of Pixel Shift Down": str([PoseDict["Down"] for PoseDict in PoseDictList]),
        "Number of Pixel Shift Left": str([PoseDict["Left"] for PoseDict in PoseDictList]),
        "Number of Pixel Shift Right": str([PoseDict["Right"] for PoseDict in PoseDictList]),
        "Overlap X Indices": str(OverlapXIndexList),
        "Overlap Y Indices": str(OverlapYIndexList),
        "Process Times": str(ProcessTimeList),
        "Total Process Time": processTime
    }
 
    # Serializing json
    json_object = json.dumps(data, indent=len(data))
 
    # Writing to sample.json
    with open(os.path.join(os.path.dirname(imgPath),os.path.splitext(os.path.basename(imgPath))[0]+"_JSON.json"), "w") as outfile:
        outfile.write(json_object)

def getInputInfo():
    outputListNum = 10
    inputPath, imgs = prepareImages(gray = True)
    if imgs is None or len(imgs) == 0:
        return False, [None]*outputListNum
    
    inputLength = len(imgs)

    # Declare Variable
    checkYAlignmentPercent = 0
    checkXAlignmentPercent = 0
    overlapPercentageTB = 0.015 
    overlapPercentageLR = 0.015

    # Set Col Num and Row Num
    while True:
        ImgsCols = simpledialog.askinteger("Column (Input Length: "+str(inputLength)+")", "Enter your column number:(Combine left-right)",
                                           initialvalue=2,
                                           minvalue=1)
        if ImgsCols is None:
            return False, [None]*outputListNum
        if ImgsCols <= 0:
            messagebox.showerror("Column Number cannot smaller than/equal to 0")
            continue
            
        while True:
            ImgsRows = simpledialog.askinteger("Row (Input Length: "+str(inputLength)+"), Cols: "+str(ImgsCols), 
                                               "Enter your row number:(Combine top-bottom)",
                                               initialvalue=1,
                                               minvalue=1)
            if ImgsRows is None:
                return False, [None]*outputListNum
            if ImgsRows <= 0:
                messagebox.showerror("Row Number cannot smaller than/equal to 0")
            else:
                break
            
        if ImgsCols * ImgsRows > len(imgs):
            messagebox.showerror("ERROR", str(ImgsCols)+" * "+str(ImgsRows)+" = "+str(ImgsCols * ImgsRows)+", but input images only have "+str(len(imgs)))
        else:
            break
    
    # Set Overlap Percentage
    if ImgsCols > 1:
        while True:
            overlapPercentageLR = simpledialog.askfloat("Left-Right Overlap", "Enter the percentage of overlap (left-right):",
                                                        initialvalue=0.2,
                                                        minvalue=0.1,
                                                        maxvalue=0.9)
            if overlapPercentageLR is None:
                return False, [None]*outputListNum
            if overlapPercentageLR > 0.9 or overlapPercentageLR < 0.1:
                messagebox.showwarning("OverlapPercentageLR","Please Enter values between 0.1 - 0.9")
            else:
                break
    
    if ImgsRows > 1:
        while True:
            overlapPercentageTB = simpledialog.askfloat("Top-Bottom Overlap", "Enter the percentage of overlap (top-bottom):",
                                                        initialvalue=0.2,
                                                        minvalue=0.1,
                                                        maxvalue=0.9)
            if overlapPercentageTB is None:
                return False, [None]*outputListNum
            if overlapPercentageTB > 0.9 or overlapPercentageTB < 0.1:
                messagebox.showwarning("overlapPercentageTB","Please Enter values between 0.1 - 0.9")
            else:
                break
    
    # Check Y Alignment Percent
    if ImgsCols > 1:
        while True:
            checkYAlignmentPercent = simpledialog.askfloat("Check Y Alignment Percent", "Check Y Alignment Percent (For Left-Right Stitching) \n (Height of Image * x), x=?",
                                                        initialvalue=0.01,
                                                        minvalue=0.0,
                                                        maxvalue=1.0)
            if checkYAlignmentPercent is None:
                return False, [None]*outputListNum
            if checkYAlignmentPercent >= 1.0 or checkYAlignmentPercent <= 0.0:
                messagebox.showwarning("Check Alignment Percent","Please Enter values in between 0.0 - 1.0")
            else:
                break
            
    if ImgsRows > 1:
        while True:
            checkXAlignmentPercent = simpledialog.askfloat("Check X Alignment Percent", "Check X Alignment Percent (For Top-Bottom Stitching) \n (Width of Image * x), x=?",
                                                        initialvalue=0.01,
                                                        minvalue=0.0,
                                                        maxvalue=1.0)
            if checkXAlignmentPercent is None:
                return False, [None]*outputListNum
            if checkXAlignmentPercent >= 1.0 or checkXAlignmentPercent <= 0.0:
                messagebox.showwarning("Check Alignment Percent","Please Enter values in between 0.0 - 1.0")
            else:
                break

    selection = askSelection(
    "Cupy(GPU) or Numpy(CPU)",
    [
        "Cupy (GPU): Large image > [500,500]",
        "Numpy (CPU): Small image < [500,500]",
        "Cancel"
    ],
    "Cupy/Numpy",0)
    
    if selection == "Cancel" or selection is None:
        return False, [None]*outputListNum
    
    saveInfo = saveStitchedImage()
    
    return True, (inputPath, imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkXAlignmentPercent, checkYAlignmentPercent, selection, saveInfo)
# ===================================================================================================================================

def main():
    #try:
    status, (inputPath, imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkXAlignmentPercent, checkYAlignmentPercent, selection, saveInfo) = getInputInfo()
    
    if not status:
        return
    
    startTime = time.time()
    
    if selection.startswith('Numpy'):
            output = numpyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent)
    elif selection.startswith('Cupy'):
            output = cupyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent)
        
    endTime = time.time()
    processTime = endTime - startTime
    print("Process Time:", str(processTime))
    
    if saveInfo is not None:
        if output is None:
            messagebox.showerror("None","Output is None")
            return
        if selection.startswith('Cupy'):
            output = cp.asnumpy(output)
            output = output.astype('uint8')
        cv2.imwrite(saveInfo,output,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        saveStitchedImageJSON(inputPath, saveInfo, 
                                overlapPercentageLR, 
                                overlapPercentageTB, 
                                checkYAlignmentPercent, 
                                checkXAlignmentPercent, 
                                ImgsRows, ImgsCols, 
                                PoseDictList,
                                OverlapXIndexList,OverlapYIndexList,
                                ProcessTimeList, processTime)
    messagebox.showinfo("Complete","Process Completed.")
    #except Exception as e:
       # messagebox.showinfo("Exception",e)

# ===================================================================================================================================

minOverlap = 0.015
st = time.time()
PoseDictList = []
ProcessTimeList = []
OverlapXIndexList = []
OverlapYIndexList = []
processedRow = 0
processedCol = 0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))

"""
im1 = np.float32(im1)
im2 = np.float32(im2)
im1 /= 255.0
im2 /= 255.0

mask1 = np.full(im1.shape[:2], 255, dtype=np.uint8)
mask2 = np.full(im2.shape[:2], 255, dtype=np.uint8)

c1 = (0,0)
c2 = (0,4425)

finder = cv2.detail.SeamFinder.createDefault(cv2.detail.SEAM_FINDER_DP_SEAM)

seq = finder.find([im1, im2], [c1, c2], [mask1, mask2])

im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

m1 = cv2.UMat.get(seq[0])
m1 = m1.astype('float32')
m2 = cv2.UMat.get(seq[1])
m2 = m2.astype('float32')

w = im1.shape[1]
h = im1.shape[0]

output1 = np.full((h,w),0)
output1[m1==255] = im1[m1==255]*255.0

temp = np.full((4425,w),0)
output1 = np.vstack([output1,temp])
im1 = np.vstack([im1,temp])
im2 = np.vstack([temp,im2])
temp2 = np.full((4425,w),0)
m2 = np.vstack([temp2,m2])
m1 = np.vstack([m1,temp2])

output2 = np.full((h,w),0)
output2 = np.vstack([temp,output2])
output2[m2==255] = im2[m2==255]*255.0

height = 5120-4432
mask = m1[4432-height:4432+height,:]
mask = cv2.GaussianBlur(mask,(5,5),height,sigmaY=height)

cv2.imshow("mask1",imageResize(mask,height=500))
m1[4432-height:4432+height,:] = mask
cv2.imshow("m1",imageResize(m1,height=500))

'''
temp = np.full((h,4432,3),150,dtype=np.uint8)
print(output.shape, temp.shape)
output = np.hstack([output,temp])
im2 = np.hstack([temp,im2])
temp2 = np.full((h,4432),150,dtype=np.uint8)
m2 = np.hstack([temp2,m2])
output[m2==255] = im2[m2==255]*255.0'''
output1 = (output1*(m1)+(1-m1))
#output2 = output2*(1-mask)+(mask)
#output = (output1+output2)/255
cv2.imshow("output",imageResize(output1,height=900))
#"""

main()