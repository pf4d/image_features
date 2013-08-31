# ImageViewer.py
# CSCI 478 - Multimedia Data Processing
# Evan Cummings
# Program to start evaluating an image in python
#
# Show the image with:
# os.startfile(imageList[n].filename)

# Analysis:
#   A program which loads a directory of images and analyses the 
#   contents for pixel data. The Manhattan distance method 
#   is used to find the distance between histograms of pixel
#   data, then displays the results of the analysis method on
#   the "results" window.

# Design:
#   This program was designed in an iterative-prototyping manner,
#   And as such could be much improved.

# Usage:
#   If python is set in your environment variables, type "python 
#   ImageViewer.py" in the command window.


from Tkinter import *
import math, os
from PixInfo import PixInfo
import numpy as np


# Main app.
class ImageViewer(Frame):
    
    # Constructor.
    def __init__(self, master, pixInfo, resultWin):
                
        Frame.__init__(self, master)
        self.master    = master
        self.pixInfo   = pixInfo
        self.resultWin = resultWin
        self.colorCode = pixInfo.get_colorCode()
        self.intenCode = pixInfo.get_intenCode()
        self.energyList = pixInfo.get_energyList()
        self.entropyList = pixInfo.get_entropyList()
        self.contrastList = pixInfo.get_contrastList()
        self.normFeatMat = pixInfo.get_normFeatMat()
        # Full-sized images.
        self.imageList = pixInfo.get_imageList()
        # Thumbnail sized images.
        self.photoList = pixInfo.get_photoList()
        # Image size for formatting.
        self.xmax = pixInfo.get_xmax()
        self.ymax = pixInfo.get_ymax()
        # Relevance feedback list, initialized to 1.
        self.rfb = IntVar()
        self.relv = [0]*len(self.imageList)
        self.var = []
        for i in range(len(self.imageList)):
            self.var.append(IntVar())
        # Weighting vector:
        self.weight = [1 / float(len(self.normFeatMat))]*92
                
        
        # Create Main frame.
        mainFrame = Frame(master)
        mainFrame.pack()
        
        
        # Create Picture chooser frame.
        listFrame = Frame(mainFrame)
        listFrame.pack(side=LEFT)
        
        
        # Create Control frame.
        controlFrame = Frame(mainFrame)
        controlFrame.pack(side=RIGHT)
        
        
        # Create Preview frame.
        previewFrame = Frame(mainFrame, 
            width=self.xmax+45, height=self.ymax)
        previewFrame.pack_propagate(0)
        previewFrame.pack(side=RIGHT)
        
        
        # Create Results frame.
        resultsFrame = Frame(self.resultWin)
        resultsFrame.pack(side=BOTTOM)
        self.canvas = Canvas(resultsFrame)
        self.resultsScrollbar = Scrollbar(resultsFrame)
        self.resultsScrollbar.pack(side=RIGHT, fill=Y)
        
        
        # Layout Picture Listbox.
        self.listScrollbar = Scrollbar(listFrame)
        self.listScrollbar.pack(side=RIGHT, fill=Y)
        self.list = Listbox(listFrame, 
            yscrollcommand=self.listScrollbar.set, 
            selectmode=BROWSE, 
            height=10)
        for i in range(len(self.imageList)):
            self.list.insert(i, self.imageList[i].filename)
        self.list.pack(side=LEFT, fill=BOTH)
        self.list.activate(1)
        self.list.bind('<<ListboxSelect>>', self.update_preview)
        self.listScrollbar.config(command=self.list.yview)
        
        
        # Layout Controls.
        button = Button(controlFrame, text="Inspect Pic", 
            fg="red", padx = 10, width=10, 
            command=lambda: self.inspect_pic(
                self.list.get(ACTIVE)))
        button.grid(row=0, column=0, sticky=E)
        
        self.b1 = Button(controlFrame, text="Color-Code", 
            padx = 10, width=10, 
            command=lambda: self.find_color_distance(method='CC'))
        self.b1.grid(row=1, column=0, sticky=E)
        
        b2 = Button(controlFrame, text="Intensity", 
            padx = 10, width=10, 
            command=lambda: self.find_color_distance(method='inten'))
        b2.grid(row=2, column=0, sticky=E)

        b3 = Button(controlFrame, text="Energy", 
            padx = 10, width=10, 
            command=lambda: self.find_tex_distance(method='energy'))
        b3.grid(row=3, column=0, sticky=E)

        b4 = Button(controlFrame, text="Entropy", 
            padx = 10, width=10, 
            command=lambda: self.find_tex_distance(method='entropy'))
        b4.grid(row=4, column=0, sticky=E)

        b5 = Button(controlFrame, text="Contrast", 
            padx = 10, width=10, 
            command=lambda: self.find_tex_distance(method='contrast'))
        b5.grid(row=0, column=1, sticky=E)
        
        b6 = Button(controlFrame, text="CC and Texture", 
            padx = 10, width=10, 
            command=lambda: self.find_rel_distance(method='CCT'))
        b6.grid(row=1, column=1, sticky=E)
        
        b7 = Button(controlFrame, text="CC and Intensity", 
            padx = 10, width=10, 
            command=lambda: self.find_rel_distance(method='CCI'))
        b7.grid(row=2, column=1, sticky=E)
        
        b8 = Button(controlFrame, text="All", 
            padx = 10, width=10, 
            command=lambda: self.find_rel_distance(method='CCTI'))
        b8.grid(row=3, column=1, sticky=E)
        
        check = Checkbutton(controlFrame,
                            text='Relevance FB',
                            variable=self.rfb,
                            onvalue=1,
                            offvalue=0)
        check.grid(row=4, column=1, sticky=W)
        
        # Layout Preview.
        self.selectImg = Label(previewFrame, 
            image=self.photoList[0])
        self.selectImg.pack()


    # Event "listener" for listbox change.
    def update_preview(self, event):
    
        i = self.list.curselection()[0]
        self.selectImg.configure(
            image=self.photoList[int(i)])


    # Update weight method:
    def update_weight(self):
        
        # Calculate the feature weights:
        # Initialize the list of relevant features:
        relImgMat = []
        for k in range(len(self.relv)):
            if self.relv[k] == 1:
                relImgMat.append(self.normFeatMat[k])
        relImgMat = np.mat(relImgMat)
        relImgMatMask = np.ma.masked_values(relImgMat, 0)
        
        
        # Calculate the weight, Wi = 1/std(i); Wi = Wi/sum(Wi):
        # If mean and std of feature i are both, zero, make Wi
        # 0.  If std(i) is zero, but mean(i) is not, make 
        # std(i) = 0.5*min(non-zero features) in Wi = 1/std(i)
        if len(relImgMat) != 0:
            
            # Wi = 1/std(i):
            for i in range(relImgMat.shape[1]):
                std = np.std(relImgMat[:,i])
                mean = np.mean(relImgMat[:,i])
                if std == 0 and mean != 0:
                    std = 0.5*np.min(relImgMatMask[:,i])
                    self.weight[i] = (1 / std)
                elif std ==0 and mean == 0:
                    self.weight[i] = 0
                else:
                    self.weight[i] = (1 / std)
                
            
            # Wi/sum(Wi):
            weightSum = np.sum(self.weight)
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] / weightSum
        
        # Otherwise make weight = 1 / N:
        else:
            self.weight = [1 / float(len(self.normFeatMat))]*92
        
        print self.weight

    
    # Find the distance on features with relevance feedback:
    def find_rel_distance(self, method):
        
        self.update_weight()        
        
        # Filter out the features we don't need:
        if method == 'CCT':
            length = range(0,67)
        elif method == 'CCI':
            length = range(3,92)
        elif method == 'CCTI':
            length = range(0,92)
        
        # Find the distance:
        # imgi = query image
        # imgk = comparison image
        i = self.list.index(ACTIVE)
        imgi = self.normFeatMat[i]
        distanceTup = []
        # For each relevant image,
        for k in range(len(self.normFeatMat)):
            imgk = self.normFeatMat[k]
            q = 0
            for j in length:
                q += self.weight[j] * math.fabs(imgi[j] - imgk[j])
            distanceTup.append((k, q))
        
        # Give a sorted tuple by distance:    
        sortedTup = sorted(distanceTup, key=lambda tup: tup[1])
        self.update_results(sortedTup)
        


    # Find the texture feature distance:
    def find_tex_distance(self, method):
    
        texList = self.normFeatMat
    
        if method == 'energy':
            tex = 0
        elif method == 'entropy':
            tex = 1
        elif method == 'contrast':
            tex = 2
        
        # Find the distance:
        # imgi = query image
        # imgk = comparison image 
        i = self.list.index(ACTIVE)
        imgi = texList[i][tex]
        distanceTup = []
        # For each relevant image feature,
        for k in range(len(texList)):
            imgk = texList[k][tex]
            q = math.fabs(imgi - imgk)
            distanceTup.append((k, q))
        
        # Give a sorted tuple by distance:    
        sortedTup = sorted(distanceTup, key=lambda tup: tup[1])
        self.update_results(sortedTup)


    # Find the color feature distance:
    def find_color_distance(self, method):
        
        if method == 'inten':
            binList = self.intenCode
        elif method == 'CC':
            binList = self.colorCode
        
        i = self.list.index(ACTIVE)
        iX = self.imageList[i].size[0]
        iY = self.imageList[i].size[1]
        distanceTup = []
        
        # i = query image index
        # k = image index
        # j = bin index
        for k in range(len(self.imageList)):
            
            imgDist = []
            sum = 0
            kX = self.imageList[k].size[0]
            kY = self.imageList[k].size[1]
            
            for j in range(len(binList[k])):
                
                imgi = float(binList[i][j]) / (iX * iY)
                imgk = float(binList[k][j]) / (kX * kY)
                q = math.fabs(imgi - imgk)
                imgDist.append(q)
            
            for q in imgDist: sum += q
            distanceTup.append((k, sum))
        
        sortedTup = sorted(distanceTup, key=lambda tup: tup[1])
        self.update_results(sortedTup)


    # Update the results window with the sorted results.
    def update_results(self, sortedTup):
        
        # Disable the rfb if checkbox is not active.
        if self.rfb.get() == 0:
            self.relv = [0]*len(self.imageList)
        
        # Calculate dimensions:
        cols = int(math.ceil(math.sqrt(len(sortedTup))))
        fullsize = (0, 0, (self.xmax*cols), ((self.ymax+20)*cols))
        
        # Initialize the canvas with dimensions equal to the 
        # number of results.
        self.canvas.delete(ALL)
        self.canvas.config( 
            width=self.xmax*cols, 
            height=(self.ymax+20)*cols/2, 
            yscrollcommand=self.resultsScrollbar.set,
            scrollregion=fullsize)
        self.canvas.pack()
        self.resultsScrollbar.config(command=self.canvas.yview)
        
        # Append images to the list in order by distance
        photoRemain = []
        for tup in sortedTup:
        
            i = tup[0]
            photoRemain.append((i, self.imageList[i].filename, 
                self.photoList[i]))
        
        
        # Place images on buttons, then on the canvas in order
        # by distance.  Buttons invoke the inspect_pic method.
        rowPos = 0
        while photoRemain:
            
            photoRow = photoRemain[:cols]
            photoRemain = photoRemain[cols:]
            colPos = 0
            for (index, filename, img) in photoRow:
                
                handler = lambda f=filename: self.inspect_pic(f)
                relvHandler = lambda i=index: self.update_relv(i)
                link = Button(self.canvas, image=img, 
                              command=handler)
                relY = Radiobutton(self.canvas, 
                                   text='R',
                                   variable=self.var[index],
                                   command=relvHandler,
                                   highlightcolor='Red',
                                   value=1)
                relN = Radiobutton(self.canvas, 
                                   text='NR',
                                   variable=self.var[index],
                                   command=relvHandler,
                                   value=0)
                if self.relv[index] == 0:
                    relY.deselect()
                    relN.deselect()
                link.pack(side=LEFT, expand=YES)
                relY.pack()
                relN.pack()
                self.canvas.create_window(
                    colPos, 
                    rowPos, 
                    anchor=NW,
                    window=link, 
                    width=self.xmax, 
                    height=(self.ymax))
                self.canvas.create_window(
                    colPos,
                    rowPos+self.ymax,
                    anchor=NW,
                    window=relY,
                    width=(self.xmax / 2),
                    height=20)
                self.canvas.create_window(
                    colPos+(self.xmax/2),
                    rowPos+self.ymax,
                    anchor=NW,
                    window=relN,
                    width=(self.xmax / 2),
                    height=20)
                colPos += self.xmax
                
            rowPos += self.ymax+20
    
    
    # Open the picture with the default operating system image
    # viewer.
    def inspect_pic(self, filename):
        
        os.startfile(filename)


    # Update the relevance feedback:
    def update_relv(self, i):
    
        self.relv[i] = self.var[i].get()
        print self.relv, "\n"


# Executable section.
if __name__ == '__main__':

    root = Tk()
    root.title('Image Analysis Tool')

    resultWin = Toplevel(root)
    resultWin.title('Result Viewer')
    resultWin.protocol('WM_DELETE_WINDOW', lambda: None)

    pixInfo = PixInfo()

    imageViewer = ImageViewer(root, pixInfo, resultWin)

    root.mainloop()

