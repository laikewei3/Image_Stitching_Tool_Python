import numpy as np
import cupy as cp
import cv2
import time
import os
import math
import tkinter as tk
from tkinter import filedialog, simpledialog,messagebox,Label,Tk,Button,IntVar,Radiobutton
import json

PoseDict = {
    "Up": 0,
    "Down": 0,
    "Left": 0,
    "Right": 0
    }

# ==================================================== Numpy Methods ================================================================

def numpyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkAlignmentPercent):
    
    for row in range(ImgsRows):
        output = imgs[ImgsCols*row]
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
            x, align_img1, align_output = findHOverlapNotAlignIndex(output, img1, img1_left_pixel, img0_right_cnt,math.floor(h*checkAlignmentPercent),img0_right_index)#math.floor(h*0.0015)
           
            tempOut = cv2.hconcat([align_output[:,:int(x)],align_img1])
            output = blendSeamlessCloneX(tempOut, align_output, x, w)
            
    return output
        
def findHOverlapIndex(output_right, img1_left, img1_left_pixel, img0_right_cnt):
    sqdif_arr = np.zeros(img0_right_cnt-img1_left_pixel, int)
    shape = img1_left.shape
    print("Finding X Overlap......")
    for x in range(img0_right_cnt - img1_left_pixel):
        diff = output_right[:,x:x+img1_left_pixel] - img1_left
        sum_sqdif = np.sum(diff*diff, dtype=np.int64)
        sqdif_arr[x] = int(sum_sqdif / shape[0] / shape[1])
    print()
    return np.where(sqdif_arr == sqdif_arr.min())[-1][0]

def findHOverlapNotAlignIndex(output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index):
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
    sqdif_arr = np.zeros(img0_bottom_cnt-img1_top_pixel, dtype=np.int64)
    print("Finding Y Overlap......")
    for y in range(img0_bottom_cnt - img1_top_pixel):
        diff = output_bottom[y:y+img1_top_pixel,:] - img1_top
        sum_sqdif = np.sum(diff*diff, dtype=np.int64)
        sqdif_arr[y] = sum_sqdif
    print()
    return np.where(sqdif_arr == sqdif_arr.min())[-1][0]

def findVOverlapNotAlignIndex(output, img1, img1_top_pixel, img0_bottom_cnt, check_x_align_pixel_cnt, img0_bottom_index):
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

# ==================================================== Cupy Methods =================================================================

def cupyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkAlignmentPercent):
    for row in range(ImgsRows):
        output =  cp.array(imgs[ImgsCols*row])
        for col in range(1, ImgsCols):
            img1 = cp.array(imgs[ImgsCols*row+col])
            w = img1.shape[1]
            
            tmpUP = cp.zeros((PoseDict["Up"],w))
            tmpDOWN = cp.zeros((PoseDict["Down"],w))
            img1 = cp.vstack([tmpDOWN,img1,tmpUP])
            
            h = img1.shape[0]
            
            
            img0_right_index = output.shape[1] - math.floor(w*overlapPercentageLR)
            img1_left_pixel = math.floor(w*(overlapPercentageLR/5))
           
            img0_right_cnt = output.shape[1] - img0_right_index
            x, align_img1, align_output = findHOverlapNotAlignIndexCu(output, img1, img1_left_pixel, img0_right_cnt,math.floor(h*checkAlignmentPercent),img0_right_index)
            
            tempOut = cp.hstack([align_output[:,:int(x)],align_img1])
            output = blendSeamlessCloneXCu(tempOut, align_output, int(x), w)
            
    return output

def findHOverlapIndexCu(output_right, img1_left, img1_left_pixel, img0_right_cnt):
    sqdif_arr = cp.zeros(img0_right_cnt-img1_left_pixel, int)
    shape = img1_left.shape
    print("Finding Overlap using Cuda......")
    for x in range(img0_right_cnt - img1_left_pixel):
        diff = output_right[:,x:x+img1_left_pixel] - img1_left
        sum_sqdif = cp.sum(diff*diff)
        sqdif_arr[x] = int(sum_sqdif / shape[0] / shape[1])
    print()
    return cp.where(sqdif_arr == sqdif_arr.min())[-1][0]

def findHOverlapNotAlignIndexCu(output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index):
    output_right = output[PoseDict["Down"]:output.shape[0]-PoseDict["Up"],img0_right_index:]
    img1_left = img1[PoseDict["Down"]:img1.shape[0]-PoseDict["Up"],:img1_left_pixel]
    print("Start Check Align using Cuda")
    sqdif_arr = cp.zeros((check_y_align_pixel_cnt*2, 2), float)
    '''
    [y diff value, x -> the best overlap x position when in this y value]
    '''
    h = img1_left.shape[0]
    w = img1_left.shape[1]
    for j in range(check_y_align_pixel_cnt):
        print("Checking Align using Cuda")
        img1DOWN = img1_left[:h-j,:] # cut down part (add black row at top) -> move down
        img1UP = img1_left[j:,:] # cut up part (add black row at bottom) -> move up
        if j > 0:
            temp = cp.zeros((j,img1_left_pixel))
            img1DOWN = cp.vstack([temp,img1DOWN])
            img1UP = cp.vstack([img1UP,temp])
       
        xDOWN = findHOverlapIndexCu(output_right[j:,:], img1DOWN[j:,:], img1_left_pixel, img0_right_cnt)+img0_right_index
        xUP = findHOverlapIndexCu(output_right[:h-j,:], img1UP[:h-j,:], img1_left_pixel, img0_right_cnt)+img0_right_index
        
        diffDOWN = output_right[j:,xDOWN-img0_right_index:xDOWN-img0_right_index+img1_left_pixel] - img1DOWN[j:,:]
        diffUP = output_right[:h-j,xUP-img0_right_index:xUP-img0_right_index+img1_left_pixel] - img1UP[:h-j,:]
        
        sum_sqdifDOWN = cp.sum(diffDOWN*diffDOWN)
        sum_sqdifUP = cp.sum(diffUP*diffUP)
        
        sqdif_arr[j,0] = sum_sqdifDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,0] = sum_sqdifUP
        
        sqdif_arr[j,1] = xDOWN
        sqdif_arr[j+check_y_align_pixel_cnt,1] = xUP
    
    index = int(cp.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    
    if index >= check_y_align_pixel_cnt:
        temp1 = cp.zeros((index-check_y_align_pixel_cnt,img1.shape[1]))
        temp2 = cp.zeros((index-check_y_align_pixel_cnt,output.shape[1]))
        align_img1 = cp.vstack([img1,temp1])
        align_output = cp.vstack([temp2,output])
        PoseDict["Up"] += index-check_y_align_pixel_cnt
    else:
        temp1 = cp.zeros((index,img1.shape[1]))
        temp2 = cp.zeros((index,output.shape[1]))
        align_img1 = cp.vstack([temp1,img1])
        align_output = cp.vstack([output,temp2])
        PoseDict["Down"] += index
        
    return sqdif_arr[index,1], align_img1, align_output

def findVOverlapIndexCu(output_bottom, img1_top, img1_top_pixel, img0_bottom_cnt):
    sqdif_arr = cp.zeros(img0_bottom_cnt-img1_top_pixel)
    print("Finding Y Overlap......")
    for y in range(img0_bottom_cnt - img1_top_pixel):
        diff = output_bottom[y:y+img1_top_pixel,:] - img1_top
        sum_sqdif = cp.sum(diff*diff)
        sqdif_arr[y] = sum_sqdif
    print()
    return cp.where(sqdif_arr == sqdif_arr.min())[-1][0]

def findVOverlapNotAlignIndexCu(output, img1, img1_top_pixel, img0_bottom_cnt, check_x_align_pixel_cnt, img0_bottom_index):
    output_bottom = output[img0_bottom_index:,PoseDict["Right"]:output.shape[1]-PoseDict["Left"]]
    img1_top = img1[:img1_top_pixel, PoseDict["Right"]:img1.shape[1]-PoseDict["Left"]]
    print("Start Check X Align")
    sqdif_arr = cp.zeros((check_x_align_pixel_cnt*2, 2), float)
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
            temp = cp.zeros((img1_top_pixel,j))
            img1RIGHT = cp.hstack([temp,img1RIGHT])
            img1LEFT = cp.hstack([img1LEFT,temp])
       
        yRIGHT = findVOverlapIndex(output_bottom[:,j:], img1RIGHT[:,j:], img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        yLEFT = findVOverlapIndex(output_bottom[:,:w-j], img1LEFT[:,:w-j], img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
        difRIGHT = output_bottom[yRIGHT-img0_bottom_index:yRIGHT-img0_bottom_index+img1_top_pixel,j:] - img1RIGHT[:,j:]
        diffLEFT = output_bottom[yLEFT-img0_bottom_index:yLEFT-img0_bottom_index+img1_top_pixel,:w-j] - img1LEFT[:,:w-j]
        
        sum_sqdifRIGHT = cp.sum(difRIGHT*difRIGHT)
        sum_sqdifLEFT = cp.sum(diffLEFT*diffLEFT)
        
        sqdif_arr[j,0] = sum_sqdifRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,0] = sum_sqdifLEFT
        
        sqdif_arr[j,1] = yRIGHT
        sqdif_arr[j+check_x_align_pixel_cnt,1] = yLEFT
    
    index = int(cp.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
    
    if index >= check_x_align_pixel_cnt:
        temp1 = cp.zeros((img1.shape[0], index-check_x_align_pixel_cnt))
        temp2 = cp.zeros((output.shape[0], index-check_x_align_pixel_cnt))
        align_img1 = cp.hstack([img1,temp1])
        align_output = cp.hstack([temp2,output])
        PoseDict["Left"] += index-check_x_align_pixel_cnt
    else:
        temp1 = cp.zeros((img1.shape[0],index))
        temp2 = cp.zeros((output.shape[0],index))
        align_img1 = cp.hstack([temp1,img1])
        align_output = cp.hstack([output,temp2])
        PoseDict["Right"] += index
        
    return sqdif_arr[index,1], align_img1, align_output


def blendSeamlessCloneXCu(tempOut, align_output, x, w):
    width_to_blend = math.floor(w*0.02)
    if x-width_to_blend < 0:
        width_to_blend = x;
    if x+width_to_blend >= w:
        width_to_blend = w-x;
    src = align_output[:,int(x)-width_to_blend:int(x)+width_to_blend]
    mask = 255*np.ones(src.shape, np.uint8)
    tempOut = cp.asnumpy(tempOut)
    tempOut = tempOut.astype('uint8')
    src = cp.asnumpy(src)
    src = src.astype('uint8')
    tempOut = cv2.cvtColor(tempOut,cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    output_img = cv2.seamlessClone(src,tempOut,mask,(int(x),tempOut.shape[0]//2),cv2.NORMAL_CLONE) #cv2.MONOCHROME_TRANSFER also can
    output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
    output_img = output_img.astype('float64')
    output = cp.array(output_img)
    return output

#  ======================================================== Others ==================================================================

def prepareImages(path = None, gray = False):
    if path is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory()
        
    file_list = os.listdir(path)
    imgs_list = [
                path+"\\"+f
                for f in file_list
                if os.path.isfile(os.path.join(path, f))
                and f.lower().endswith((".png", ".gif",".jpg",".jpeg",".bmp"))
            ]
    imgs=[]

    for img in imgs_list:
        print("LOADING IMAGE......")
        if gray:
            imgs.append(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
        else:
            imgs.append(cv2.imread(img))
    
    return imgs

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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

def showCupyImage(name, img):
    imgNUMPY = cp.asnumpy(img)
    imgNUMPY = imgNUMPY.astype('uint8')
    cv2.imshow(name, image_resize(imgNUMPY,height=500))

def ask_multiple_choice_question(prompt, options,title=None):
    root = Tk()
    if title:
        root.title(title)
    if prompt:
        Label(root, text=prompt).pack()
    v = IntVar(value=2)
    for i, option in enumerate(options):
        Radiobutton(root, text=option, variable=v, value=i).pack(anchor="w")
    Button(text="Submit", command=root.destroy).pack()
    root.mainloop()
    return options[v.get()]

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
    
def saveStitchedImageJSON(imgPath, overlapPercentageLR, overlapPercentageTB, 
                          checkAlignmentPercent, ImgsRows, ImgsCols, 
                          PoseUp, PoseDown, processTime):
    # Data to be written
    data = {
        "Image Path": imgPath,
        "Left-Right Overlap %": overlapPercentageLR,
        "Top-Bottom Overlap %": overlapPercentageTB,
        "Check Alignment Percent": checkAlignmentPercent,
        "Input Images Rows": ImgsRows,
        "Input Images Cols": ImgsCols,
        "Number of Pixel Shift Up": PoseUp,
        "Number of Pixel Shift Down": PoseDown,
        "Process Time": processTime
    }
 
    # Serializing json
    json_object = json.dumps(data, indent=9)
 
    # Writing to sample.json
    with open(os.path.join(os.path.dirname(imgPath),os.path.splitext(os.path.basename(imgPath))[0]+"_JSON.json"), "w") as outfile:
        outfile.write(json_object)

def main():
    selection = ask_multiple_choice_question(
    "Cupy(GPU) or Numpy(CPU)",
    [
        "Cupy (GPU): Large image > [1000,1000]",
        "Numpy (CPU): Small image < [1000,1000]",
        "Cancel"
    ],
    "Cupy/Numpy")
    
    if selection == "Cancel":
        return
    
    imgs = prepareImages(gray = True)
    
    inputLength = len(imgs)
    # Set Col Num and Row Num
    while True:
        ImgsCols = simpledialog.askinteger("Column (Input Length: "+str(inputLength)+")", "Enter your column number:(Combine left-right)",
                                           initialvalue=2,
                                           minvalue=1)
        if ImgsCols <= 0:
            messagebox.showerror("Column Number cannot smaller than/equal to 0")
        ImgsRows = simpledialog.askinteger("Row (Input Length: "+str(inputLength)+"), Cols: "+str(ImgsCols), 
                                           "Enter your row number:(Combine top-bottom)",
                                           initialvalue=1,
                                           minvalue=1)
        if ImgsRows <= 0:
            messagebox.showerror("Row Number cannot smaller than/equal to 0")
        if ImgsCols * ImgsRows > len(imgs):
            messagebox.showerror(str(ImgsCols)+" * "+str(ImgsRows)+" = "+str(ImgsCols * ImgsRows)+", but input images only have "+len(imgs))
        else:
            break
    
    # Set Overlap Percentage
    while True:
        overlapPercentageLR = simpledialog.askfloat("Left-Right Overlap", "Enter the percentage of overlap (left-right):",
                                                    initialvalue=0.3,
                                                    minvalue=0.1,
                                                    maxvalue=0.9)
        if overlapPercentageLR > 0.9 or overlapPercentageLR < 0.1:
            messagebox.showwarning("OverlapPercentageLR","Please Enter values between 0.1 - 0.9")
        else:
            break
    while True:
        overlapPercentageTB = simpledialog.askfloat("Top-Bottom Overlap", "Enter the percentage of overlap (top-bottom):",
                                                    initialvalue=0.3,
                                                    minvalue=0.1,
                                                    maxvalue=0.9)
        if overlapPercentageTB > 0.9 or overlapPercentageTB < 0.1:
            messagebox.showwarning("overlapPercentageTB","Please Enter values between 0.1 - 0.9")
        else:
            break
    
    # Check Alignment Percent
    while True:
        checkAlignmentPercent = simpledialog.askfloat("Check Alignment Percent,x (Height of Image * x)", "x:",
                                                    initialvalue=0.01,
                                                    minvalue=0.0,
                                                    maxvalue=1.0)
        if checkAlignmentPercent >= 1.0 or checkAlignmentPercent <= 0.0:
            messagebox.showwarning("overlapPercentageTB","Please Enter values in between 0.0 - 1.0")
        else:
            break

    saveInfo = saveStitchedImage()
    startTime = time.time()
    
    if selection.startswith('Numpy'):
        output = numpyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkAlignmentPercent)
    elif selection.startswith('Cupy'):
        output = cupyStitch(imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkAlignmentPercent)
        
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
        cv2.imwrite(saveInfo,output)
        saveStitchedImageJSON(saveInfo, 
                              overlapPercentageLR, 
                              overlapPercentageTB, 
                              checkAlignmentPercent, 
                              ImgsRows, ImgsCols, 
                              PoseDict["Up"], 
                              PoseDict["Down"],
                              processTime)
    messagebox.showinfo("Complete","Process Completed.")

# ===================================================================================================================================

main()