import gc
import torch
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import cv2
import time
import os
from tkinter import filedialog, simpledialog,messagebox,Label,Tk,Button,IntVar,Radiobutton
import json

# ==================================================== Numpy Methods ================================================================

def numpyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent):
    global processedRow
    global processedCol
    processedRow = 0
    w=None
    h=None
    for row in range(ImgsRows):
        PoseDict = {
        "Up": 0,
        "Down": 0,
        "Left": 0,
        "Right": 0
        }
        output = cv2.imread(imgs[ImgsCols*row],cv2.IMREAD_GRAYSCALE)
        st = time.time()
        processedCol = 0
        
        for col in range(1, ImgsCols):
            img1 = cv2.imread(imgs[ImgsCols*row+col],cv2.IMREAD_GRAYSCALE)
            
            w = img1.shape[1]
            tmpUP = np.zeros((PoseDict["Up"],w))
            tmpDOWN = np.zeros((PoseDict["Down"],w))
            img1_processed = np.vstack([tmpDOWN,img1,tmpUP])
            h = img1_processed.shape[0]
            
            img1 = None
            tmpDOWN = None
            tmpUP = None
            
            out_h = output.shape[0]
            diff = abs(out_h-h)
            
            if h > out_h:
                temp = np.zeros((diff,output.shape[1]))
                output = np.vstack([output,temp])
            else:
                temp = np.zeros((diff,w))
                img1_processed = np.vstack([img1_processed,temp])
            h = img1_processed.shape[0]
            
            temp = None
            out_h = None
            diff = None
            
            img1_left_pixel = int(w*(overlapPercentageLR/5))
            value = int(w*overlapPercentageLR)
            img0_right_cnt = value if output.shape[1] - img1_left_pixel >= value else output.shape[1]
            img0_right_index = output.shape[1] - img0_right_cnt
            
            gc.collect()

            checkYAlignmentCnt = int(h*checkYAlignmentPercent)
            checkYAlignmentCnt = 1 if checkYAlignmentCnt == 0 else checkYAlignmentCnt
            x, align_img1, align_output, movementY = findHOverlapNotAlignIndex(output, img1_processed, img1_left_pixel, 
                                                                      img0_right_cnt, checkYAlignmentCnt,
                                                                      img0_right_index, PoseDict)
            checkYAlignmentCnt = None
            output = None
            img1_processed = None
            img1_left_pixel = None
            value = None
            img0_right_cnt = None
            img0_right_index = None   
            gc.collect()
            
            OverlapXIndexList.append(int(x))
            output = blendAlphaX(align_img1, align_output,x,w,movementY,PoseDict)
            processedCol+=1
            x, align_img1, align_output, movementY = None, None, None, None
            
            gc.collect()
        
        if row > 0:
            tmpLEFT = np.zeros((output.shape[0],PoseDictList[row-1]["Left"]))
            tmpRIGHT = np.zeros((output.shape[0],PoseDictList[row-1]["Right"]))
            output_processed = np.hstack([tmpRIGHT,output,tmpLEFT])
            
            output = None
            tmpLEFT = None
            tmpRIGHT = None
            
            PrevOutput_w = PrevOutput.shape[1]
            PrevOutput_h = PrevOutput.shape[0]
            output_w = output_processed.shape[1]
            output_h = output_processed.shape[0]
            
            diff = abs(output_w-PrevOutput_w)
            if PrevOutput_w > output_w:
                temp = np.zeros((output_h,diff))
                output_processed = np.hstack([output_processed,temp])
            else:
                temp = np.zeros((PrevOutput_h,diff))
                PrevOutput = np.hstack([PrevOutput,temp])
                
            temp = None
            
            img1_top_pixel = int(output_processed.shape[0]*overlapPercentageTB/5)
            value = int(output_processed.shape[0]*overlapPercentageTB)
            img0_bottom_cnt = value if PrevOutput.shape[0] - img1_top_pixel >= value else PrevOutput.shape[0]
            img0_bottom_index = PrevOutput.shape[0] - img0_bottom_cnt
            
            width = output_processed.shape[1] if w == None else w
            checkXAlignmentCnt = int(width*checkXAlignmentPercent)
            checkXAlignmentCnt = 1 if checkXAlignmentCnt == 0 else checkXAlignmentCnt
            y, align_img1, align_output, movementX = findVOverlapNotAlignIndexCu(PrevOutput, output_processed, 
                                                                      img1_top_pixel, img0_bottom_cnt, 
                                                                      checkXAlignmentCnt, 
                                                                      img0_bottom_index, PoseDict)
            checkXAlignmentCnt = None
            width = None
            img1_top_pixel = None
            value = None
            img0_bottom_cnt = None
            img0_bottom_index = None
            
            OverlapYIndexList.append(int(y))
            output = blendAlphaY(align_img1, align_output, y, int(output_processed.shape[0]), movementX, PoseDict)
            y, align_img1, align_output, movementX, output_processed = None, None, None, None, None
            gc.collect()
        
        PrevOutput = None
        PrevOutput = output
        output = None
        PoseDictList.append(PoseDict)
        et = time.time()
        ProcessTimeList.append(et-st)
        st = time.time()
        processedRow += 1
        gc.collect()
    return PrevOutput

def findHOverlapIndex(output_right, img1_left, img1_left_pixel, img0_right_cnt, min_overlap_count = 0):
    sqdif_arr = np.zeros(img0_right_cnt-img1_left_pixel-min_overlap_count, float)
    print("Finding X Overlap......")
    shape = img1_left.shape
    for x in range(min_overlap_count, img0_right_cnt - img1_left_pixel):
        img = output_right[:,x:x+img1_left_pixel]
        if shape[0] < 10000:
            step = 2
        else:
            step = shape[0] % 5000
        imgHalf = img[::step]
        img1Half = img1_left[::step]
        diff = imgHalf - img1Half
        sum_sqdif = np.sum(diff*diff)
        sqdif_arr[x-min_overlap_count] = sum_sqdif
    print()
    index = int(np.where(sqdif_arr == sqdif_arr.min())[0][0]) + min_overlap_count
    
    img = None
    step = None
    img1Half = None
    imgHalf = None
    diff = None
    sum_sqdif = None
    sqdif_arr = None
    gc.collect()
    return index

def findHOverlapNotAlignIndex(output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index, PoseDict):
    output_right = output[PoseDict["Down"]:output.shape[0]-PoseDict["Up"],img0_right_index:]
    img1_left = img1[PoseDict["Down"]:img1.shape[0]-PoseDict["Up"],:img1_left_pixel]
    print(str(processedRow)+"-"+str(processedCol),": Start Check Align")
    sqdif_arr = np.zeros((check_y_align_pixel_cnt*2, 2), float)
    '''
    [y diff value, x -> the best overlap x position when in this y value]
    '''
    h = img1_left.shape[0]
    for j in range(check_y_align_pixel_cnt):
        print(str(processedRow)+"-"+str(processedCol),": Checking Align")
        img1DOWN = img1_left[:h-j,:] # cut down part (add black row at top) -> move down
        img1UP = img1_left[j:,:] # cut up part (add black row at bottom) -> move up
        if j > 0:
            temp = np.zeros((j,img1_left_pixel))
            img1DOWN = np.vstack([temp,img1DOWN])
            img1UP = np.vstack([img1UP,temp])
       
        output_right_TOP = output_right[j:,:]
        img1DOWN_TOP = img1DOWN[j:,:]
        xDOWN = findHOverlapIndex(output_right_TOP, img1DOWN_TOP, img1_left_pixel, img0_right_cnt, int(img1.shape[1]*minOverlap))+img0_right_index
        
        output_right_BOTTOM = output_right[:h-j,:]
        img1UP_BOTTOM = img1UP[:h-j,:]
        xUP = findHOverlapIndex(output_right_BOTTOM, img1UP_BOTTOM, img1_left_pixel, img0_right_cnt, int(img1.shape[1]*minOverlap))+img0_right_index
        
        output_right_BOTTOM_diffDOWN = output_right_TOP[:,xDOWN-img0_right_index:xDOWN-img0_right_index+img1_left_pixel]
        output_right_BOTTOM_diffUP = output_right_BOTTOM[:,xUP-img0_right_index:xUP-img0_right_index+img1_left_pixel]
        
        diffDOWN = output_right_BOTTOM_diffDOWN - img1DOWN_TOP
        diffUP = output_right_BOTTOM_diffUP - img1UP_BOTTOM
        
        sum_sqdifDOWN = np.sum(diffDOWN*diffDOWN)
        sum_sqdifUP = np.sum(diffUP*diffUP)
        
        sqdif_arr[j,0] = sum_sqdifDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,0] = sum_sqdifUP
        
        sqdif_arr[j,1] = xDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,1] = xUP
    
    temp=None
    output_right_BOTTOM = None
    output_right_TOP = None
    output_right_BOTTOM_diffDOWN = None
    output_right_BOTTOM_diffUP = None
    img1DOWN = None
    img1UP = None
    img1DOWN_TOP = None
    img1UP_BOTTOM = None
    diffDOWN = None
    diffUP = None
    sum_sqdifDOWN = None
    sum_sqdifUP = None
    output_right = None
    img1_left = None
    h = None
    gc.collect()

    index = int(np.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    x = sqdif_arr[index,1]
    sqdif_arr = None
    movement = []
    if index >= check_y_align_pixel_cnt:
        v = index-check_y_align_pixel_cnt
        temp1 = np.zeros((v,img1.shape[1]))
        temp2 = np.zeros((v,output.shape[1]))
        align_img1 = np.vstack([img1,temp1])
        temp1 = None
        img1 = None
        align_output = np.vstack([temp2,output])
        temp2 = None
        output = None
        PoseDict["Up"] += v
        movement.append("Up")
        movement.append(v)
    else:
        temp1 = np.zeros((index,img1.shape[1]))
        temp2 = np.zeros((index,output.shape[1]))
        align_img1 = np.vstack([temp1,img1])
        temp1 = None
        img1 = None
        align_output = np.vstack([output,temp2])
        temp2 = None
        output = None
        PoseDict["Down"] += index
        movement.append("Down")
        movement.append(index)
    
    gc.collect()
    return x, align_img1, align_output, movement

def findVOverlapIndex(output_bottom, img1_top, img1_top_pixel, img0_bottom_cnt, min_overlap_count = 0):
    sqdif_arr = np.zeros(img0_bottom_cnt-img1_top_pixel)
    print("Finding Y Overlap......")
    shape = img1_top.shape
    for y in range(min_overlap_count, img0_bottom_cnt - img1_top_pixel):
        img = output_bottom[y:y+img1_top_pixel]
        if shape[1] < 10000:
            step = 2
        else:
            step = shape[1] % 5000
        imgHalf = img[:,::step]
        img1Half = img1_top[:,::step]
        diff = imgHalf - img1Half
        sum_sqdif = np.sum(diff*diff)
        sqdif_arr[y+min_overlap_count] = sum_sqdif
    print()
    index = int(np.where(sqdif_arr == sqdif_arr.min())[0][0])+min_overlap_count
    
    img = None
    imgHalf = None
    img1Half = None
    diff = None
    sum_sqdif = None
    sqdif_arr = None
    shape = None
    gc.collect()
    return index

def findVOverlapNotAlignIndex(output, img1, img1_top_pixel, img0_bottom_cnt, check_x_align_pixel_cnt, img0_bottom_index, PoseDict):
    output_bottom = equaliseNumpyImage(output[img0_bottom_index:,PoseDict["Right"]:output.shape[1]-PoseDict["Left"]])
    img1_top = equaliseNumpyImage(img1[:img1_top_pixel, PoseDict["Right"]:img1.shape[1]-PoseDict["Left"]])
    
    print(str(processedRow)+"-"+str(processedCol),": Start Check X Align")
    sqdif_arr = np.zeros((check_x_align_pixel_cnt*2, 2), float)
    '''
    [x diff value, y -> the best overlap y position when in this x value]
    '''
    w = img1_top.shape[1]
    for j in range(check_x_align_pixel_cnt):
        print(str(processedRow)+"-"+str(processedCol),": Checking X Align")
        img1RIGHT = img1_top[:,:w-j] # cut right part (add black row at left) -> move right
        img1LEFT = img1_top[:,j:] # cut left part (add black row at right) -> move left
        if j > 0:
            temp = np.zeros((img1_top_pixel,j))
            img1RIGHT = np.hstack([temp,img1RIGHT])
            img1LEFT = np.hstack([img1LEFT,temp])
            temp = None
            
        output_bottom_LEFT = output_bottom[:,j:]
        img1RIGHT_LEFT = img1RIGHT[:,j:]
        yRIGHT = findVOverlapIndex(output_bottom_LEFT, img1RIGHT_LEFT, img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        output_bottom_RIGHT = output_bottom[:,:w-j]
        img1LEFT_RIGHT = img1LEFT[:,:w-j]
        yLEFT = findVOverlapIndex(output_bottom_RIGHT, img1LEFT_RIGHT, img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        output_bottom_LEFT_diffRIGHT = output_bottom_LEFT[yRIGHT-img0_bottom_index:yRIGHT-img0_bottom_index+img1_top_pixel]
        output_bottom_RIGHT_diffLEFT = output_bottom_RIGHT[yLEFT-img0_bottom_index:yLEFT-img0_bottom_index+img1_top_pixel]
        output_bottom_LEFT = None
        output_bottom_RIGHT = None

        diffRIGHT = output_bottom_LEFT_diffRIGHT - img1RIGHT_LEFT
        diffLEFT =  output_bottom_RIGHT_diffLEFT - img1LEFT_RIGHT
        
        img1RIGHT = None
        img1LEFT = None
        img1RIGHT_LEFT = None
        img1LEFT_RIGHT = None
        output_bottom_LEFT_diffRIGHT = None
        output_bottom_LEFT_diffRIGHT = None
        
        sum_sqdifRIGHT = np.sum(diffRIGHT*diffRIGHT)
        sum_sqdifLEFT = np.sum(diffLEFT*diffLEFT)
        
        diffRIGHT = None
        diffLEFT = None
        
        sqdif_arr[j,0] = sum_sqdifRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,0] = sum_sqdifLEFT
        
        sqdif_arr[j,1] = yRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,1] = yLEFT
        
    sum_sqdifRIGHT = None
    sum_sqdifLEFT = None
        
    index = int(np.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    y = sqdif_arr[index,1]
    sqdif_arr = None
    movement = []
    # Align by adding black pixel at left/right
    if index >= check_x_align_pixel_cnt:
        v = index-check_x_align_pixel_cnt
        temp1 = np.zeros((img1.shape[0], v))
        temp2 = np.zeros((output.shape[0], v))
        align_img1 = np.hstack([img1,temp1])
        temp1 = None
        img1 = None
        align_output = np.hstack([temp2,output])
        temp2 = None
        output = None
        PoseDict["Left"] += v
        movement.append("LEFT")
        movement.append(v)
    else:
        temp1 = np.zeros((img1.shape[0],index))
        temp2 = np.zeros((output.shape[0],index))
        align_img1 = np.hstack([temp1,img1])
        temp1 = None
        img1 = None
        align_output = np.hstack([output,temp2])
        temp2 = None
        output = None
        PoseDict["Right"] += index
        movement.append("Right")
        movement.append(index)
    
    output_bottom = None
    img1_top = None
    w = None
    gc.collect()
    return y, align_img1, align_output, movement

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
def blendAlphaX(align_img1, align_output, x, w, movementY, PoseDict):
    width_to_blend = int(w*0.1)
    if movementY[0] == "UP":
        U = int(movementY[1] + PoseDict["Down"])
        D = int(align_output.shape[0] - PoseDict["UP"])
    else:
        U = int(PoseDict["Down"])
        D = int(align_output.shape[0] - movementY[1])

    if x-width_to_blend < 0:
        width_to_blend = int(x);
    if x+width_to_blend >= align_output.shape[1]:
        width_to_blend = int(align_output.shape[1]-x);
    
    src = align_output[U:D,int(x):int(x)+width_to_blend]
    src_shape = src.shape
    
    target = align_img1[U:D,:src_shape[1]]
    
    mask1 = np.linspace(1, 0, src_shape[1])
    mask2 = 1-mask1
    final = src * mask1 + target * mask2
    
    src = None
    target = None
    mask1 = None
    mask2 = None
    
    gc.collect()
    
    tempOut = align_output[:,:int(x)]
    output = np.hstack([tempOut,align_img1])
    
    tempOut = None
    align_img1 = None

    output[U:D,int(x):int(x)+src_shape[1]] = final
    
    final = None
    src_shape = None
    w = None
    U = None
    D = None
    width_to_blend = None
    gc.collect()
    return output

def blendAlphaYCu(align_img1, align_output, y, h, movementX, PoseDict):
    height_to_blend = int(h*0.05)
    ignoreCurr = max(PoseDict["Down"],PoseDict["Up"])
    if movementX[0] == "Left":
        L = movementX[1] + PoseDict["Right"]
        R = align_output.shape[1] - PoseDict["Left"]
    else:
        L = PoseDict["Right"]
        R = align_output.shape[1] - movementX[1]

    if h-height_to_blend < 0:
        height_to_blend = int(y);
    if y+height_to_blend > int(align_output.shape[0]):
        height_to_blend = int(align_output.shape[0]-y);
    
    src = align_output[int(y)-height_to_blend:int(y)+height_to_blend,L:R]
    src_shape = src.shape
    target = align_img1[:src_shape[0]-height_to_blend,L:R]
    
    tempMask = np.swapaxes(np.tile(np.linspace(1, 0, int(src_shape[0]-height_to_blend)), (src_shape[1], 1)),0,1)
    tempMask_rev = 1-tempMask
    mask1, mask2 = ignoreBlackPixelNumpy(target, tempMask, tempMask_rev, ignoreCurr)
    
    tempMask = None
    tempMask_rev = None

    temp = np.ones((height_to_blend,src_shape[1]))
    mask1 = np.vstack([temp, mask1])
    temp2 = 1-temp
    target = np.vstack([(temp2),target])
    mask2 = np.vstack([(temp2),mask2])
    temp = None
    temp2 = None
    # blend both image 
    final = src * mask1 + target * mask2
    src = None
    target = None

    tempOut = align_output[:y,:]
    output = np.vstack([tempOut, align_img1])
    
    tempOut = None
    align_img1 = None
    
    output[int(y)-height_to_blend:int(y)+height_to_blend,L:R] = final
    final = None
    h = None
    src_shape = None
    L = None
    R = None
    gc.collect()
    return output

def equaliseNumpyImage(image_gray):
    image_gray_int = image_gray.astype('uint8')
    image_gray_int = None
    image_gray_int_equalised = clahe.apply(image_gray_int)
    image_gray_int = None
    image_gray_float = image_gray_int_equalised.astype('float64')
    image_gray_int_equalised = None
    gc.collect()
    return image_gray_float

def ignoreBlackPixelNumpy(target, mask1, mask2, ignoreCurr):
    alpha = target[:ignoreCurr]>=1
    alpha_float = alpha.astype('float64')
    alpha = None
    alpha = cv2.GaussianBlur(alpha_float,(45,45),100)
    alpha_float = None
    mask2[:ignoreCurr] *= alpha/4
    alpha = None
    mask1 = 1 - mask2
    gc.collect()
    return mask1, mask2

# ==================================================== Cupy Methods =================================================================
def cupyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent):
    global processedRow
    global processedCol
    processedRow = 0
    w=None
    h=None
    for row in range(ImgsRows):
        PoseDict = {
        "Up": 0,
        "Down": 0,
        "Left": 0,
        "Right": 0
        }
        output_numpy = cv2.imread(imgs[ImgsCols*row],cv2.IMREAD_GRAYSCALE)
        output = cp.array(output_numpy)
        output_numpy = None
        st = time.time()
        processedCol = 0
        
        for col in range(1, ImgsCols):
            img1_numpy = cv2.imread(imgs[ImgsCols*row+col],cv2.IMREAD_GRAYSCALE)
            img1 = cp.array(img1_numpy)
            img1_numpy = None
            
            w = img1.shape[1]
            tmpUP = cp.zeros((PoseDict["Up"],w))
            tmpDOWN = cp.zeros((PoseDict["Down"],w))
            img1_processed = cp.vstack([tmpDOWN,img1,tmpUP])
            h = img1_processed.shape[0]
            
            img1 = None
            tmpDOWN = None
            tmpUP = None
            
            out_h = output.shape[0]
            diff = abs(out_h-h)
            
            if h > out_h:
                temp = cp.zeros((diff,output.shape[1]))
                output = cp.vstack([output,temp])
            else:
                temp = cp.zeros((diff,w))
                img1_processed = cp.vstack([img1_processed,temp])
            h = img1_processed.shape[0]
            
            temp = None
            out_h = None
            diff = None
            
            img1_left_pixel = int(w*(overlapPercentageLR/5))
            value = int(w*overlapPercentageLR)
            img0_right_cnt = value if output.shape[1] - img1_left_pixel >= value else output.shape[1]
            img0_right_index = output.shape[1] - img0_right_cnt
            
            mempool.free_all_blocks()
            pinned_mempool.n_free_blocks()

            checkYAlignmentCnt = int(h*checkYAlignmentPercent)
            checkYAlignmentCnt = 1 if checkYAlignmentCnt == 0 else checkYAlignmentCnt
            x, align_img1, align_output, movementY = findHOverlapNotAlignIndexCu(output, img1_processed, img1_left_pixel, 
                                                                      img0_right_cnt, checkYAlignmentCnt,
                                                                      img0_right_index, PoseDict)
            checkYAlignmentCnt = None
            output = None
            img1_processed = None
            img1_left_pixel = None
            value = None
            img0_right_cnt = None
            img0_right_index = None   
            gc.collect()
            mempool.free_all_blocks()
            pinned_mempool.n_free_blocks()
            
            OverlapXIndexList.append(int(x))
            '''
            tempOut = cp.hstack([align_output[:,:int(x)],align_img1])
            output = blendSeamlessCloneXCu(tempOut, align_output, int(x), w, movementY, PoseDict)'''
            output = blendAlphaXCu(align_img1, align_output,x,w,movementY,PoseDict)
            processedCol+=1
            x, align_img1, align_output, movementY = None, None, None, None
            
            mempool.free_all_blocks()
            pinned_mempool.n_free_blocks()
        
        if row > 0:
            tmpLEFT = cp.zeros((output.shape[0],PoseDictList[row-1]["Left"]))
            tmpRIGHT = cp.zeros((output.shape[0],PoseDictList[row-1]["Right"]))
            output_processed = cp.hstack([tmpRIGHT,output,tmpLEFT])
            
            output = None
            tmpLEFT = None
            tmpRIGHT = None
            
            PrevOutput_w = PrevOutput.shape[1]
            PrevOutput_h = PrevOutput.shape[0]
            output_w = output_processed.shape[1]
            output_h = output_processed.shape[0]
            
            diff = abs(output_w-PrevOutput_w)
            if PrevOutput_w > output_w:
                temp = cp.zeros((output_h,diff))
                output_processed = cp.hstack([output_processed,temp])
            else:
                temp = cp.zeros((PrevOutput_h,diff))
                PrevOutput = cp.hstack([PrevOutput,temp])
                
            temp = None
            
            img1_top_pixel = int(output_processed.shape[0]*overlapPercentageTB/5)
            value = int(output_processed.shape[0]*overlapPercentageTB)
            img0_bottom_cnt = value if PrevOutput.shape[0] - img1_top_pixel >= value else PrevOutput.shape[0]
            img0_bottom_index = PrevOutput.shape[0] - img0_bottom_cnt
            
            width = output_processed.shape[1] if w == None else w
            checkXAlignmentCnt = int(width*checkXAlignmentPercent)
            checkXAlignmentCnt = 1 if checkXAlignmentCnt == 0 else checkXAlignmentCnt
            y, align_img1, align_output, movementX = findVOverlapNotAlignIndexCu(PrevOutput, output_processed, 
                                                                      img1_top_pixel, img0_bottom_cnt, 
                                                                      checkXAlignmentCnt, 
                                                                      img0_bottom_index, PoseDict)
            checkXAlignmentCnt = None
            width = None
            img1_top_pixel = None
            value = None
            img0_bottom_cnt = None
            img0_bottom_index = None
            
            OverlapYIndexList.append(int(y))
            output = blendAlphaYCu(align_img1, align_output, y, int(output_processed.shape[0]), movementX, PoseDict)
            '''
            tempOut = cp.vstack([align_output[:y,:],align_img1])
            output = blendSeamlessCloneYCu(tempOut, align_output, y, h, movementX, PoseDict)'''
            y, align_img1, align_output, movementX, output_processed = None, None, None, None, None
            mempool.free_all_blocks()
        
        PrevOutput = None
        PrevOutput = output
        output = None
        PoseDictList.append(PoseDict)
        et = time.time()
        ProcessTimeList.append(et-st)
        st = time.time()
        processedRow += 1
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.n_free_blocks()
    return PrevOutput

def findHOverlapIndexCu(output_right, img1_left, img1_left_pixel, img0_right_cnt, min_overlap_count = 0):
    sqdif_arr = cp.zeros(img0_right_cnt-img1_left_pixel-min_overlap_count, float)
    print("Finding X Overlap using Cuda......")
    shape = img1_left.shape
    for x in range(min_overlap_count, img0_right_cnt - img1_left_pixel):
        img = output_right[:,x:x+img1_left_pixel]
        if shape[0] < 10000:
            step = 2
        else:
            step = shape[0] % 5000
        imgHalf = img[::step]
        img1Half = img1_left[::step]
        diff = imgHalf - img1Half
        sum_sqdif = cp.sum(diff*diff)
        sqdif_arr[x-min_overlap_count] = sum_sqdif
    print()
    index = int(cp.where(sqdif_arr == sqdif_arr.min())[0][0]) + min_overlap_count
    
    img = None
    step = None
    img1Half = None
    imgHalf = None
    diff = None
    sum_sqdif = None
    sqdif_arr = None
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return index

def findHOverlapNotAlignIndexCu(output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index, PoseDict):
    output_right = output[PoseDict["Down"]:output.shape[0]-PoseDict["Up"],img0_right_index:]
    img1_left = img1[PoseDict["Down"]:img1.shape[0]-PoseDict["Up"],:img1_left_pixel]
    print(str(processedRow)+"-"+str(processedCol),": Start Check Align using Cuda")
    sqdif_arr = cp.zeros((check_y_align_pixel_cnt*2, 2), float)
    '''
    [y diff value, x -> the best overlap x position when in this y value]
    '''
    h = img1_left.shape[0]
    for j in range(check_y_align_pixel_cnt):
        print(str(processedRow)+"-"+str(processedCol),": Checking Align using Cuda")
        img1DOWN = img1_left[:h-j,:] # cut down part (add black row at top) -> move down
        img1UP = img1_left[j:,:] # cut up part (add black row at bottom) -> move up
        if j > 0:
            temp = cp.zeros((j,img1_left_pixel))
            img1DOWN = cp.vstack([temp,img1DOWN])
            img1UP = cp.vstack([img1UP,temp])
       
        output_right_TOP = output_right[j:,:]
        img1DOWN_TOP = img1DOWN[j:,:]
        xDOWN = findHOverlapIndexCu(output_right_TOP, img1DOWN_TOP, img1_left_pixel, img0_right_cnt, int(img1.shape[1]*minOverlap))+img0_right_index
        
        output_right_BOTTOM = output_right[:h-j,:]
        img1UP_BOTTOM = img1UP[:h-j,:]
        xUP = findHOverlapIndexCu(output_right_BOTTOM, img1UP_BOTTOM, img1_left_pixel, img0_right_cnt, int(img1.shape[1]*minOverlap))+img0_right_index
        
        output_right_BOTTOM_diffDOWN = output_right_TOP[:,xDOWN-img0_right_index:xDOWN-img0_right_index+img1_left_pixel]
        output_right_BOTTOM_diffUP = output_right_BOTTOM[:,xUP-img0_right_index:xUP-img0_right_index+img1_left_pixel]
        
        diffDOWN = output_right_BOTTOM_diffDOWN - img1DOWN_TOP
        diffUP = output_right_BOTTOM_diffUP - img1UP_BOTTOM
        
        sum_sqdifDOWN = cp.sum(diffDOWN*diffDOWN)
        sum_sqdifUP = cp.sum(diffUP*diffUP)
        
        sqdif_arr[j,0] = sum_sqdifDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,0] = sum_sqdifUP
        
        sqdif_arr[j,1] = xDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,1] = xUP
    
    temp=None
    output_right_BOTTOM = None
    output_right_TOP = None
    output_right_BOTTOM_diffDOWN = None
    output_right_BOTTOM_diffUP = None
    img1DOWN = None
    img1UP = None
    img1DOWN_TOP = None
    img1UP_BOTTOM = None
    diffDOWN = None
    diffUP = None
    sum_sqdifDOWN = None
    sum_sqdifUP = None
    output_right = None
    img1_left = None
    h = None
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    gc.collect()

    index = int(cp.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    x = sqdif_arr[index,1]
    sqdif_arr = None
    movement = []
    if index >= check_y_align_pixel_cnt:
        v = index-check_y_align_pixel_cnt
        temp1 = cp.zeros((v,img1.shape[1]))
        temp2 = cp.zeros((v,output.shape[1]))
        align_img1 = cp.vstack([img1,temp1])
        temp1 = None
        img1 = None
        align_output = cp.vstack([temp2,output])
        temp2 = None
        output = None
        PoseDict["Up"] += v
        movement.append("Up")
        movement.append(v)
    else:
        temp1 = cp.zeros((index,img1.shape[1]))
        temp2 = cp.zeros((index,output.shape[1]))
        align_img1 = cp.vstack([temp1,img1])
        temp1 = None
        img1 = None
        align_output = cp.vstack([output,temp2])
        temp2 = None
        output = None
        PoseDict["Down"] += index
        movement.append("Down")
        movement.append(index)
    
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return x, align_img1, align_output, movement

def findVOverlapIndexCu(output_bottom, img1_top, img1_top_pixel, img0_bottom_cnt, min_overlap_count = 0):
    sqdif_arr = cp.zeros(img0_bottom_cnt-img1_top_pixel)
    print("Finding Y Overlap using Cuda......")
    shape = img1_top.shape
    for y in range(min_overlap_count, img0_bottom_cnt - img1_top_pixel):
        img = output_bottom[y:y+img1_top_pixel]
        if shape[1] < 10000:
            step = 2
        else:
            step = shape[1] % 5000
        imgHalf = img[:,::step]
        img1Half = img1_top[:,::step]
        diff = imgHalf - img1Half
        sum_sqdif = cp.sum(diff*diff)
        sqdif_arr[y+min_overlap_count] = sum_sqdif
    print()
    index = int(cp.where(sqdif_arr == sqdif_arr.min())[0][0])+min_overlap_count
    
    img = None
    imgHalf = None
    img1Half = None
    diff = None
    sum_sqdif = None
    sqdif_arr = None
    shape = None
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return index

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
            temp = None
            
        output_bottom_LEFT = output_bottom[:,j:]
        img1RIGHT_LEFT = img1RIGHT[:,j:]
        yRIGHT = findVOverlapIndexCu(output_bottom_LEFT, img1RIGHT_LEFT, img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        output_bottom_RIGHT = output_bottom[:,:w-j]
        img1LEFT_RIGHT = img1LEFT[:,:w-j]
        yLEFT = findVOverlapIndexCu(output_bottom_RIGHT, img1LEFT_RIGHT, img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        output_bottom_LEFT_diffRIGHT = output_bottom_LEFT[yRIGHT-img0_bottom_index:yRIGHT-img0_bottom_index+img1_top_pixel]
        output_bottom_RIGHT_diffLEFT = output_bottom_RIGHT[yLEFT-img0_bottom_index:yLEFT-img0_bottom_index+img1_top_pixel]
        output_bottom_LEFT = None
        output_bottom_RIGHT = None

        diffRIGHT = output_bottom_LEFT_diffRIGHT - img1RIGHT_LEFT
        diffLEFT =  output_bottom_RIGHT_diffLEFT - img1LEFT_RIGHT
        
        img1RIGHT = None
        img1LEFT = None
        img1RIGHT_LEFT = None
        img1LEFT_RIGHT = None
        output_bottom_LEFT_diffRIGHT = None
        output_bottom_LEFT_diffRIGHT = None
        
        sum_sqdifRIGHT = cp.sum(diffRIGHT*diffRIGHT)
        sum_sqdifLEFT = cp.sum(diffLEFT*diffLEFT)
        
        diffRIGHT = None
        diffLEFT = None
        
        sqdif_arr[j,0] = sum_sqdifRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,0] = sum_sqdifLEFT
        
        sqdif_arr[j,1] = yRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,1] = yLEFT
        
    sum_sqdifRIGHT = None
    sum_sqdifLEFT = None
        
    index = int(cp.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    y = sqdif_arr[index,1]
    sqdif_arr = None
    movement = []
    # Align by adding black pixel at left/right
    if index >= check_x_align_pixel_cnt:
        v = index-check_x_align_pixel_cnt
        temp1 = cp.zeros((img1.shape[0], v))
        temp2 = cp.zeros((output.shape[0], v))
        align_img1 = cp.hstack([img1,temp1])
        temp1 = None
        img1 = None
        align_output = cp.hstack([temp2,output])
        temp2 = None
        output = None
        PoseDict["Left"] += v
        movement.append("LEFT")
        movement.append(v)
    else:
        temp1 = cp.zeros((img1.shape[0],index))
        temp2 = cp.zeros((output.shape[0],index))
        align_img1 = cp.hstack([temp1,img1])
        temp1 = None
        img1 = None
        align_output = cp.hstack([output,temp2])
        temp2 = None
        output = None
        PoseDict["Right"] += index
        movement.append("Right")
        movement.append(index)
    
    output_bottom = None
    img1_top = None
    w = None
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return y, align_img1, align_output, movement

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
    width_to_blend = int(w*0.015)
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
    height_to_blend = int(h*0.015)
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
    width_to_blend = int(w*0.1)
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
    
    mask1 = cp.linspace(1, 0, src_shape[1])
    mask2 = 1-mask1
    final = src * mask1 + target * mask2
    
    src = None
    target = None
    mask1 = None
    mask2 = None
    
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    
    tempOut = align_output[:,:x]
    output = cp.hstack([tempOut,align_img1])
    
    tempOut = None
    align_img1 = None

    output[U:D,int(x):int(x)+src_shape[1]] = final
    
    final = None
    src_shape = None
    w = None
    U = None
    D = None
    width_to_blend = None
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return output

def blendAlphaYCu(align_img1, align_output, y, h, movementX, PoseDict):
    height_to_blend = int(h*0.05)
    ignoreCurr = max(PoseDict["Down"],PoseDict["Up"])
    if movementX[0] == "Left":
        L = movementX[1] + PoseDict["Right"]
        R = align_output.shape[1] - PoseDict["Left"]
    else:
        L = PoseDict["Right"]
        R = align_output.shape[1] - movementX[1]

    if h-height_to_blend < 0:
        height_to_blend = int(y);
    if y+height_to_blend > int(align_output.shape[0]):
        height_to_blend = int(align_output.shape[0]-y);
    
    src = align_output[int(y)-height_to_blend:int(y)+height_to_blend,L:R]
    src_shape = src.shape
    target = align_img1[:src_shape[0]-height_to_blend,L:R]
    
    tempMask = cp.swapaxes(cp.tile(cp.linspace(1, 0, int(src_shape[0]-height_to_blend)), (src_shape[1], 1)),0,1)
    tempMask_rev = 1-tempMask
    mask1, mask2 = ignoreBlackPixel(target, tempMask, tempMask_rev, ignoreCurr)
    
    tempMask = None
    tempMask_rev = None

    temp = cp.ones((height_to_blend,src_shape[1]))
    mask1 = cp.vstack([temp, mask1])
    temp2 = 1-temp
    target = cp.vstack([(temp2),target])
    mask2 = cp.vstack([(temp2),mask2])
    temp = None
    temp2 = None
    # blend both image 
    final = src * mask1 + target * mask2
    src = None
    target = None

    tempOut = align_output[:y,:]
    output = cp.vstack([tempOut, align_img1])
    
    tempOut = None
    align_img1 = None
    
    output[int(y)-height_to_blend:int(y)+height_to_blend,L:R] = final
    final = None
    h = None
    src_shape = None
    L = None
    R = None
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return output

def equaliseCupyImage(image_gray):
    image_gray_numpy = cp.asnumpy(image_gray)
    image_gray_numpy_int = image_gray_numpy.astype('uint8')
    image_gray_numpy = None
    image_gray_int_equalised = clahe.apply(image_gray_numpy_int)
    image_gray_numpy_int = None
    image_gray_float = image_gray_int_equalised.astype('float64')
    image_gray_int_equalised = None
    cupyImage = cp.asarray(image_gray_float)
    image_gray_float = None
    gc.collect()
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return cupyImage 

def ignoreBlackPixel(target, mask1, mask2, ignoreCurr):
    alpha = target[:ignoreCurr]>=1
    alpha_float = alpha.astype('float64')
    alpha = None
    alpha = ndimage.gaussian_filter(alpha_float,100)
    alpha_float = None
    mask2[:ignoreCurr] *= alpha/4
    alpha = None
    mask1 = 1 - mask2
    mempool.free_all_blocks()
    pinned_mempool.n_free_blocks()
    return mask1, mask2

#  ======================================================== Others ==================================================================
def prepareImages(path = None):
    if path is None:
        selection = askSelection("Folder or Images?",["Folder","Images"],"Select Input Types",0)
        if selection is None:
            return None
        elif selection == "Folder":
            path = filedialog.askdirectory()
            if path == "":
                return None
        else:
            imgs_list = filedialog.askopenfilenames()
            return imgs_list
           
    file_list = os.listdir(path)
    imgs_list = [
                path+"\\"+f
                for f in file_list
                if os.path.isfile(os.path.join(path, f))
                and f.lower().endswith((".png", ".gif",".jpg",".jpeg",".bmp"))
            ]
        
    return imgs_list

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
    outputListNum = 9
    inputPath = prepareImages()
    if inputPath is None or len(inputPath) == 0:
        return False, [None]*outputListNum
    
    inputLength = len(inputPath)

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
            
        if ImgsCols * ImgsRows > len(inputPath):
            messagebox.showerror("ERROR", str(ImgsCols)+" * "+str(ImgsRows)+" = "+str(ImgsCols * ImgsRows)+", but input images only have "+str(len(inputPath)))
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
                                                        initialvalue=0.0025,
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
                                                        initialvalue=0.0025,
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
    
    return True, (inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkXAlignmentPercent, checkYAlignmentPercent, selection, saveInfo)
# ===================================================================================================================================

def main():
    #try:
    status, (inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkXAlignmentPercent, checkYAlignmentPercent, selection, saveInfo) = getInputInfo()
    
    if not status:
        return
    
    startTime = time.time()
    
    if selection.startswith('Numpy'):
            output = numpyStitch(inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent)
    elif selection.startswith('Cupy'):
            output = cupyStitch(inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent)
        
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
      #  messagebox.showinfo("Exception",e)

# ===================================================================================================================================

minOverlap = 0.015
st = time.time()
PoseDictList = []
ProcessTimeList = []
OverlapXIndexList = []
OverlapYIndexList = []
processedRow = 0
processedCol = 0
clahe = cv2.createCLAHE(tileGridSize=(3,3))
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

main()