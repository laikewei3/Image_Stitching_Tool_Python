import gc
from re import S
import numpy as np
import cupy as cp
import cv2
import time
import os
from tkinter import filedialog, simpledialog,messagebox,Label,Tk,Button,IntVar,Radiobutton
import json
from cupyx.scipy import ndimage
from numpy.core.multiarray import ndarray
from Stitch import StitchBase
from cupyStitch import cupyStitch

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
            selection = None
            return imgs_list
           
    file_list = os.listdir(path)
    imgs_list = [
                path+"\\"+f
                for f in file_list
                if os.path.isfile(os.path.join(path, f))
                and f.lower().endswith((".png", ".gif",".jpg",".jpeg",".bmp"))
            ]
    selection = None
    file_list = None
    gc.collect()
    return imgs_list

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
        elif ImgsCols * ImgsRows > 42:
            answer = messagebox.askokcancel(
                title='Warning',
                message='Too many input images - might face Memory Issue (Memory Allocation Failure etc.)',
                icon=messagebox.WARNING)
            if answer:
                break
        else:
            break
    
    # Set Overlap Percentage
    if ImgsCols > 1:
        while True:
            overlapPercentageLR = simpledialog.askfloat("Left-Right Overlap", "Enter the percentage of overlap (left-right):",
                                                        initialvalue=0.3,
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
                                                        initialvalue=0.3,
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
                                                        initialvalue=0.005,
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
                                                        initialvalue=0.015,
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
    try:
        status, (inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkXAlignmentPercent, checkYAlignmentPercent, selection, saveInfo) = getInputInfo()
    
        if not status:
            return
    
        startTime = time.time()
    
        if selection.startswith('Numpy'):
                Stool = StitchBase(inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent)
        elif selection.startswith('Cupy'):
                Stool =  cupyStitch(inputPath, ImgsRows, ImgsCols, overlapPercentageLR, overlapPercentageTB, checkYAlignmentPercent, checkXAlignmentPercent)
        
        output = Stool.Run()
        
        endTime = time.time()
        processTime = endTime - startTime
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
                                    Stool.PoseDictList,
                                    Stool.OverlapXIndexList,Stool.OverlapYIndexList,
                                    Stool.ProcessTimeList, processTime)
        messagebox.showinfo("Complete","Process Completed.")
    except Exception as e:
        messagebox.showinfo("Exception",e)
    else:
        print("Process Time:", str(processTime))
    

# ===================================================================================================================================
if __name__ == '__main__':
    main()