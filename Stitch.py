import gc
import time
from tkinter import messagebox
import cv2
import cupy as cp
import numpy as np
from numpy import ndarray

class StitchBase(object):
    
    processedRow = 0
    processedCol = 0
    clahe = cv2.createCLAHE() # for alignment
    clahe_illuminance = cv2.createCLAHE(clipLimit=2) # for illuminance difference
    
    # Variable to be save in the JSON file
    PoseDictList = []
    ProcessTimeList = []
    OverlapXIndexList = []
    OverlapYIndexList = []
    
    def __init__(self, imgs, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent):
        self.imgs = imgs
        self.ImgsRows = ImgsRows
        self.ImgsCols = ImgsCols
        self.overlapPercentageLR = overlapPercentageLR
        self.overlapPercentageTB = overlapPercentageTB
        self.checkYAlignmentPercent = checkYAlignmentPercent
        self.checkXAlignmentPercent = checkXAlignmentPercent
        self.minOverlap = 0.015
        
    def imageResize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
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
        
        dim = None
        r = None
        gc.collect()
        # return the resized image
        return resized
    
    def showImages(self, img, title = "Imgs", height = None, width = None):
        # if image is cupy format, convert to numpy format first
        if not isinstance(img,ndarray): 
            img = cp.asnumpy(img)
            img = img.astype('uint8')
            
        cv2.imshow(title, self.imageResize(img,width,height))
    
    def equaliseInputImage(self, image_gray):
        image_gray_equalised = self.clahe_illuminance.apply(image_gray)
        image_gray = None
        gc.collect()
        return image_gray_equalised
    
    def Run(self):
        try:
            self.processedRow = 0
            w=h=None
            for row in range(self.ImgsRows):
                PoseDict = {
                "Up": 0,
                "Down": 0,
                "Left": 0,
                "Right": 0
                }
                # Equalise Histogram Adaptively to reduce illuminance difference
                output = self.equaliseInputImage(cv2.imread(self.imgs[self.ImgsCols*row],cv2.IMREAD_GRAYSCALE))
                
                st = time.time()
                self.processedCol = 0
                gc.collect()
        
                for col in range(1, self.ImgsCols):
                    # Equalise Histogram Adaptively to reduce illuminance difference
                    img1 = self.equaliseInputImage(cv2.imread(self.imgs[self.ImgsCols*row+col],cv2.IMREAD_GRAYSCALE))
                    w = img1.shape[1]
                    tmpUP = np.zeros((PoseDict["Up"],w))
                    tmpDOWN = np.zeros((PoseDict["Down"],w))
            
                    img1_processed = np.vstack([tmpDOWN,img1,tmpUP])
                    h = img1_processed.shape[0]
            
                    img1 = tmpDOWN = tmpUP = None
            
                    out_h = output.shape[0]
                    diff = abs(out_h-h)
            
                    if h > out_h:
                        temp = np.zeros((diff,output.shape[1]))
                        output = np.vstack([output,temp])
                    else:
                        temp = np.zeros((diff,w))
                        img1_processed = np.vstack([img1_processed,temp])
                    h = img1_processed.shape[0]
            
                    temp = out_h = diff = None
            
                    img1_left_pixel = int(w*(self.overlapPercentageLR/5))
                    value = int(w*self.overlapPercentageLR)
                    img0_right_cnt = value if output.shape[1] - img1_left_pixel >= value else output.shape[1]
                    img0_right_index = output.shape[1] - img0_right_cnt
            
                    checkYAlignmentCnt = int(h*self.checkYAlignmentPercent)
                    checkYAlignmentCnt = 1 if checkYAlignmentCnt == 0 else checkYAlignmentCnt
                    x, align_img1, align_output, movementY = self.findHOverlapNotAlignIndex(output, img1_processed, img1_left_pixel, 
                                                                                img0_right_cnt, checkYAlignmentCnt,
                                                                                img0_right_index, PoseDict)
                    
                    checkYAlignmentCnt = output = img1_processed = img1_left_pixel = value = img0_right_cnt = img0_right_index = None   
                    gc.collect()
            
                    self.OverlapXIndexList.append(int(x))
                    output = self.blendAlphaX(align_img1, align_output,x,w,movementY,PoseDict)
                    
                    self.processedCol+=1
                    x, align_img1, align_output, movementY = None, None, None, None
                    gc.collect()
        
                if row > 0:
                    tmpLEFT = np.zeros((output.shape[0],self.PoseDictList[row-1]["Left"]))
                    tmpRIGHT = np.zeros((output.shape[0],self.PoseDictList[row-1]["Right"]))
                    output_processed = np.hstack([tmpRIGHT,output,tmpLEFT])
            
                    output = tmpLEFT = tmpRIGHT = None
            
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
            
                    img1_top_pixel = int(output_processed.shape[0]*self.overlapPercentageTB/5)
                    value = int(output_processed.shape[0]*self.overlapPercentageTB)
                    img0_bottom_cnt = value if PrevOutput.shape[0] - img1_top_pixel >= value else PrevOutput.shape[0]
                    img0_bottom_index = PrevOutput.shape[0] - img0_bottom_cnt
            
                    width = output_processed.shape[1] if w == None else w
                    checkXAlignmentCnt = int(width*self.checkXAlignmentPercent)
                    checkXAlignmentCnt = 1 if checkXAlignmentCnt == 0 else checkXAlignmentCnt
                    y, align_img1, align_output, movementX = self.findVOverlapNotAlignIndex(PrevOutput, output_processed, 
                                                                                img1_top_pixel, img0_bottom_cnt, 
                                                                                checkXAlignmentCnt, 
                                                                                img0_bottom_index, PoseDict)
                    
                    checkXAlignmentCnt = width = img1_top_pixel = value = img0_bottom_cnt = img0_bottom_index = None
            
                    self.OverlapYIndexList.append(int(y))
                    output = self.blendAlphaY(align_img1, align_output, y, int(output_processed.shape[0]), movementX, PoseDict)
                
                    y, align_img1, align_output, movementX, output_processed = None, None, None, None, None
                    gc.collect()
        
                PrevOutput = None
                PrevOutput = output
                output = None
                self.PoseDictList.append(PoseDict)
                et = time.time()
                self.ProcessTimeList.append(et-st)
                st = time.time()
                self.processedRow += 1
                gc.collect()
        except Exception as e:
            messagebox.showinfo("Exception",e)
            return output
        return PrevOutput
    
    '''
    |          :             |
    |    img   :     img     |
    |     1    :      2      |
    |          :             |
     ------------------------
               ^
               |___ find this index
    '''
    def findHOverlapIndex(self, output_right, img1_left, img1_left_pixel, img0_right_cnt, min_overlap_count = 0):
        # Initialise an empty arr to record the sum of squared difference at each position
        sqdif_arr = np.zeros(img0_right_cnt-img1_left_pixel-min_overlap_count, float)
        
        print("Finding X Overlap......")
        
        shape = img1_left.shape
        for x in range(min_overlap_count, img0_right_cnt - img1_left_pixel):
            img = output_right[:,x:x+img1_left_pixel]
            
            # Use every pixel will slow down the process, just use part of it -> 1 image has around 5000 pixel per row, so i also choose <= 5000
            if shape[0] < 10000:
                step = 2
            else:
                step = shape[0] % 5000
            
            # Get sum of squared differences
            imgHalf = img[::step]
            img1Half = img1_left[::step]
            diff = imgHalf - img1Half
            sum_sqdif = np.sum(diff*diff)
            sqdif_arr[x-min_overlap_count] = sum_sqdif
        print()
        
        index = int(np.where(sqdif_arr == sqdif_arr.min())[0][0]) + min_overlap_count
    
        # Clear memory
        img = step = img1Half = imgHalf = diff = sum_sqdif = sqdif_arr = None
        return index
    
    '''
    |                       |
    |        img1           |
    |.......................| <-- find this index
    |    overlap area       |
     -----------------------
    |                       |
    |        img2           |
    |                       |
    |                       |
     -----------------------
    '''
    def findVOverlapIndex(self, output_bottom, img1_top, img1_top_pixel, img0_bottom_cnt, min_overlap_count = 0):
        # Initialise an empty arr to record the sum of squared difference at each position
        sqdif_arr = np.zeros(img0_bottom_cnt-img1_top_pixel)
        
        print("Finding Y Overlap......")
        shape = img1_top.shape
        for y in range(min_overlap_count, img0_bottom_cnt - img1_top_pixel):
            img = output_bottom[y:y+img1_top_pixel]
            
            # Use every pixel will slow down the process, just use part of it -> 1 image has around 5000 pixel per row, so i also choose <= 5000
            if shape[1] < 10000:
                step = 2
            else:
                step = shape[1] % 5000
             
            # Get sum of squared differences
            imgHalf = img[:,::step]
            img1Half = img1_top[:,::step]
            diff = imgHalf - img1Half
            sum_sqdif = np.sum(diff*diff)
            sqdif_arr[y+min_overlap_count] = sum_sqdif
        print()
        
        index = int(np.where(sqdif_arr == sqdif_arr.min())[0][0])+min_overlap_count
        
        # Clear memory
        img = imgHalf = img1Half = diff = sum_sqdif = sqdif_arr = shape = None
        return index

    # Before find the overlap on the x axis, there might be misalignment on y-axis, so need to move up and down before find overlap
    # move up/down -> get overlap index(min sum of squared diff)[x-axis] -> get most align position(min sum of squared diff)[y-axis]
    def findHOverlapNotAlignIndex(self, output, img1, img1_left_pixel, img0_right_cnt, check_y_align_pixel_cnt, img0_right_index, PoseDict):
        # Ignore part above PoseDict["Down"] & part below PoseDict["Up"] as they are area added for alignment purpose
        output_right = output[PoseDict["Down"]:output.shape[0]-PoseDict["Up"],img0_right_index:]
        img1_left = img1[PoseDict["Down"]:img1.shape[0]-PoseDict["Up"],:img1_left_pixel]
        
        print(str(self.processedRow)+"-"+str(self.processedCol),": Start Check Align")
        
        # Initialise an empty arr to record the sum of squared difference at each position
        # check_y_align_pixel_cnt*2 => at index [0,check_y_align_pixel_cnt) store the sum of squared difference when the img1 move DOWN (index) pixel
        # index 0 => img1 not moving
        # index 1 => img1 move down 1 pixel
        # index 2 => img2 move down 2 pixels

        # check_y_align_pixel_cnt*2 => at index [check_y_align_pixel_cnt,check_y_align_pixel_cnt*2) store the sum of squared difference when the img1 move UP (check_y_align_pixel_cnt*2-index) pixel
        # index check_y_align_pixel_cnt => img1 not moving
        # index check_y_align_pixel_cnt+1 => img1 move up 1 pixel
        # index check_y_align_pixel_cnt+2 => img2 move down 2 pixels
        sqdif_arr = np.zeros((check_y_align_pixel_cnt*2, 2), float)
        #[y diff value, x -> the best overlap x position when in this y value]
        
        h = img1_left.shape[0]
        for j in range(check_y_align_pixel_cnt):
            print(str(self.processedRow)+"-"+str(self.processedCol),": Checking Align")
            
            # Preprocess img1
            img1DOWN = img1_left[:h-j,:] # cut down part (add black row at top) -> move down
            img1UP = img1_left[j:,:] # cut up part (add black row at bottom) -> move up
            if j > 0:
                temp = np.zeros((j,img1_left_pixel))
                img1DOWN = np.vstack([temp,img1DOWN])
                img1UP = np.vstack([img1UP,temp])
       
            # Calculate the best X overlap index when img1 move DOWN
            output_right_TOP = output_right[j:,:]
            img1DOWN_TOP = img1DOWN[j:,:]
            xDOWN = self.findHOverlapIndex(output_right_TOP, img1DOWN_TOP, img1_left_pixel, img0_right_cnt, int(img1.shape[1]*self.minOverlap))+img0_right_index
        
            # Calculate the best X overlap index when img1 move UP
            output_right_BOTTOM = output_right[:h-j,:]
            img1UP_BOTTOM = img1UP[:h-j,:]
            xUP = self.findHOverlapIndex(output_right_BOTTOM, img1UP_BOTTOM, img1_left_pixel, img0_right_cnt, int(img1.shape[1]*self.minOverlap))+img0_right_index
        
            # Calculate the sum of squared difference
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
    
        # Clear the memory
        temp = output_right_BOTTOM = output_right_TOP = output_right_BOTTOM_diffDOWN = output_right_BOTTOM_diffUP = img1DOWN = img1UP = img1DOWN_TOP = None
        img1UP_BOTTOM = diffDOWN = diffUP = sum_sqdifDOWN = sum_sqdifUP = output_right = img1_left = h = None
        gc.collect()

        index = int(np.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
        x = sqdif_arr[index,1]
        
        # Align the output images and img1 by stack the images with rectangle
        sqdif_arr = None
        movement = []
        if index >= check_y_align_pixel_cnt:
            v = index-check_y_align_pixel_cnt
            temp1 = np.zeros((v,img1.shape[1]))
            temp2 = np.zeros((v,output.shape[1]))
            align_img1 = np.vstack([img1,temp1])
            align_output = np.vstack([temp2,output])
            PoseDict["Up"] += v
            movement.append("Up")
            movement.append(v)
        else:
            temp1 = np.zeros((index,img1.shape[1]))
            temp2 = np.zeros((index,output.shape[1]))
            align_img1 = np.vstack([temp1,img1])
            align_output = np.vstack([output,temp2])
            PoseDict["Down"] += index
            movement.append("Down")
            movement.append(index)
            
        temp1 = img1 = temp2 = output = None
        gc.collect()
        return x, align_img1, align_output, movement   

    # Before find the overlap on the y axis, there might be misalignment on x-axis, so need to move left and right before find overlap
    # move left/right -> get overlap index(min sum of squared diff)[y-axis] -> get most align position(min sum of squared diff)[x-axis]
    def findVOverlapNotAlignIndex(self, output, img1, img1_top_pixel, img0_bottom_cnt, check_x_align_pixel_cnt, img0_bottom_index, PoseDict):
        # Ignore part at the left PoseDict["Right"] & part at the right PoseDict["Left"] as they are area added for alignment purpose
        output_bottom = self.equaliseImageFindOverlap(output[img0_bottom_index:,PoseDict["Right"]:output.shape[1]-PoseDict["Left"]])
        img1_top = self.equaliseImageFindOverlap(img1[:img1_top_pixel, PoseDict["Right"]:img1.shape[1]-PoseDict["Left"]])
    
        print(str(self.processedRow)+"-"+str(self.processedCol),": Start Check X Align")
        
        # Initialise an empty arr to record the sum of squared difference at each position
        # check_x_align_pixel_cnt*2 => at index [0,check_x_align_pixel_cnt) store the sum of squared difference when the img1 move RIGHT (index) pixel
        # index 0 => img1 not moving
        # index 1 => img1 move RIGHT 1 pixel
        # index 2 => img2 move RIGHT 2 pixels

        # check_x_align_pixel_cnt*2 => at index [check_x_align_pixel_cnt,check_x_align_pixel_cnt*2) store the sum of squared difference when the img1 move LEFT (check_y_align_pixel_cnt*2-index) pixel
        # index check_x_align_pixel_cnt => img1 not moving
        # index check_x_align_pixel_cnt+1 => img1 move LEFT 1 pixel
        # index check_x_align_pixel_cnt+2 => img2 move LEFT 2 pixels
        sqdif_arr = np.zeros((check_x_align_pixel_cnt*2, 2), float)
        #[x diff value, y -> the best overlap y position when in this x value]
        
        w = img1_top.shape[1]
        for j in range(check_x_align_pixel_cnt):
            print(str(self.processedRow)+"-"+str(self.processedCol),": Checking X Align")
            
            #Preprocess Images
            img1RIGHT = img1_top[:,:w-j] # cut right part (add black row at left) -> move right
            img1LEFT = img1_top[:,j:] # cut left part (add black row at right) -> move left
            if j > 0:
                temp = np.zeros((img1_top_pixel,j))
                img1RIGHT = np.hstack([temp,img1RIGHT])
                img1LEFT = np.hstack([img1LEFT,temp])
                temp = None
                
            # Calculate the best Y overlap index when img1 move RIGHT
            output_bottom_LEFT = output_bottom[:,j:]
            img1RIGHT_LEFT = img1RIGHT[:,j:]
            yRIGHT = self.findVOverlapIndex(output_bottom_LEFT, img1RIGHT_LEFT, img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
            
            # Calculate the best Y overlap index when img1 move LEFT
            output_bottom_RIGHT = output_bottom[:,:w-j]
            img1LEFT_RIGHT = img1LEFT[:,:w-j]
            yLEFT = self.findVOverlapIndex(output_bottom_RIGHT, img1LEFT_RIGHT, img1_top_pixel, img0_bottom_cnt)+img0_bottom_index
        
            # Calculate the sum of squared difference
            output_bottom_LEFT_diffRIGHT = output_bottom_LEFT[yRIGHT-img0_bottom_index:yRIGHT-img0_bottom_index+img1_top_pixel]
            output_bottom_RIGHT_diffLEFT = output_bottom_RIGHT[yLEFT-img0_bottom_index:yLEFT-img0_bottom_index+img1_top_pixel]
            output_bottom_LEFT = output_bottom_RIGHT = None

            diffRIGHT = output_bottom_LEFT_diffRIGHT - img1RIGHT_LEFT
            diffLEFT =  output_bottom_RIGHT_diffLEFT - img1LEFT_RIGHT
        
            img1RIGHT = img1LEFT = img1RIGHT_LEFT = img1LEFT_RIGHT = output_bottom_LEFT_diffRIGHT = output_bottom_LEFT_diffRIGHT = None
        
            sum_sqdifRIGHT = np.sum(diffRIGHT*diffRIGHT)
            sum_sqdifLEFT = np.sum(diffLEFT*diffLEFT)
        
            diffRIGHT = diffLEFT = None
        
            sqdif_arr[j,0] = sum_sqdifRIGHT
            sqdif_arr[j+check_x_align_pixel_cnt,0] = sum_sqdifLEFT
        
            sqdif_arr[j,1] = yRIGHT
            sqdif_arr[j+check_x_align_pixel_cnt,1] = yLEFT
        
        sum_sqdifRIGHT = sum_sqdifLEFT = None
        
        index = int(np.where(sqdif_arr[:,0] == sqdif_arr[:,0].min())[0][0])
        y = sqdif_arr[index,1]
        sqdif_arr = None
        
        gc.collect()
        
        movement = []
        # Align by adding black pixel at left/right
        if index >= check_x_align_pixel_cnt:
            v = index-check_x_align_pixel_cnt
            temp1 = np.zeros((img1.shape[0], v))
            temp2 = np.zeros((output.shape[0], v))
            align_img1 = np.hstack([img1,temp1])
            align_output = np.hstack([temp2,output])
            PoseDict["Left"] += v
            movement.append("LEFT")
            movement.append(v)
        else:
            temp1 = np.zeros((img1.shape[0],index))
            temp2 = np.zeros((output.shape[0],index))
            align_img1 = np.hstack([temp1,img1])
            align_output = np.hstack([output,temp2])
            PoseDict["Right"] += index
            movement.append("Right")
            movement.append(index)
        
        # Clear memory
        temp1 = img1 = temp2 = output = output_bottom = img1_top = w = None
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
    def blendAlphaX(self, align_img1, align_output, x, w, movementY, PoseDict):
        width_to_blend = int(w*0.1)
        # Ignore the part added manually for alignment to prevent color errors caused by blending
        if movementY[0] == "UP":
            U = movementY[1] + PoseDict["Down"]
            D = align_output.shape[0] - PoseDict["UP"]
        else:
            U = PoseDict["Down"]
            D = align_output.shape[0] - movementY[1]
        
        # To makesure the width to blend the alpha will not exceed the image width
        if x+width_to_blend >= align_output.shape[1]:
            width_to_blend = align_output.shape[1]-x;
        
        src = align_output[U:D,int(x):int(x)+int(w*0.1)]
        src_shape = src.shape
        target = align_img1[U:D,:src_shape[1]]
        
        #Create the masks and blend images overlap's part
        mask1 = np.linspace(1, 0, src_shape[1])
        mask2 = 1-mask1
        final = src * mask1 + target * mask2
        
        # Clear memory
        src = target = mask1 = mask2 = None
        gc.collect()
    
        # Stack image together
        tempOut = align_output[:,:int(x)]
        output = np.hstack([tempOut,align_img1])
        output[U:D,int(x):int(x)+src_shape[1]] = final
    
        # Clear memory
        tempOut = align_img1 = None
        final = src_shape = w = U = D = None
        gc.collect()
        return output

    def blendAlphaY(self, align_img1, align_output, y, h, movementX, PoseDict):
        height_to_blend = int(h*0.1)
        ignoreCurr = max(PoseDict["Down"],PoseDict["Up"])
        # Ignore the part added manually for alignment to prevent color errors caused by blending
        if movementX[0] == "Left":
            L = movementX[1] + PoseDict["Right"]
            R = align_output.shape[1] - PoseDict["Left"]
        else:
            L = PoseDict["Right"]
            R = align_output.shape[1] - movementX[1]
        
        # To makesure the height to blend the alpha will not exceed the image height or too small
        if y+height_to_blend > int(align_output.shape[0]):
            height_to_blend = int(align_output.shape[0]-y);
    
        src = align_output[int(y)-height_to_blend:int(y)+height_to_blend,L:R]
        src_shape = src.shape
        target = align_img1[:src_shape[0]-height_to_blend,L:R]
    
        #Create the masks and blend images overlap's part
        temp = np.ones((height_to_blend+ignoreCurr,src_shape[1]))
        tempMask = np.swapaxes(np.tile(np.linspace(1, 0, int(src_shape[0]-height_to_blend-ignoreCurr)), (src_shape[1], 1)),0,1)
        mask1 = np.vstack([temp, tempMask])
        mask2 = 1-mask1
        tempMask = None
        target = np.vstack([temp[:height_to_blend],target])
        temp = None
        
        # blend both image 
        final = src * mask1 + target * mask2
        
        src = target = None

        tempOut = align_output[:int(y),:]
        output = np.vstack([tempOut, align_img1])
        output[int(y)-height_to_blend:int(y)+height_to_blend,L:R] = final
        
        # Clear memory
        tempOut = align_img1 = None
        final = h = src_shape = L = R = None
        gc.collect()
        return output

    def equaliseImageFindOverlap(self, image_gray):
        image_gray_int = image_gray.astype('uint8')
        image_gray_int_equalised = self.clahe.apply(image_gray_int)
        image_gray_int = None
        image_gray_float = image_gray_int_equalised.astype('float64')
        image_gray_int_equalised = None
        gc.collect()
        return image_gray_float
    