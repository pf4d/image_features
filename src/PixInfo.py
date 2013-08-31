# PixInfo.py
# CSCI 478 - Multimedia Data Processing
# Evan Cummings
# Program to start evaluating an image in python
# Ideally written in C for speed optimization.

from PIL import Image, ImageTk
import glob, os, math
import numpy as np


# Pixel Info class.
class PixInfo:
  
  # Constructor.
  def __init__(self):
    
    self.imageList = []
    self.photoList = []
    self.xmax = 0
    self.ymax = 0
    self.colorCode = []
    self.intenCode = []
    self.gsImgList = []
    self.coMatList = []
    self.normMatList = []
    self.energyList = []
    self.entropyList = []
    self.contrastList = []
    self.featureMat = []
    self.normFeatMat = []
    
    # Add each image (for evaluation) into a list, 
    # and a Photo from the image (for the GUI) in a list.
    for infile in glob.glob('images/*.jpg'):
      
      file, ext = os.path.splitext(infile)
      im = Image.open(infile)
      
      # Resize the image for thumbnails.
      imSize = im.size
      x = imSize[0]/4
      y = imSize[1]/4
      imResize = im.resize((x, y), Image.ANTIALIAS)
      photo = ImageTk.PhotoImage(imResize)
      
      # Find the max height and width of the set of pics.
      if x > self.xmax:
        self.xmax = x
      if y > self.ymax:
        self.ymax = y
      
      # Add the images to the lists.
      self.imageList.append(im)
      self.photoList.append(photo)

    # Create a list of pixel data for each image and add it
    # to a list.
    for im in self.imageList:
      CcBins, InBins = self.encode(im)
      GsImg = self.gs_encode(im)
      CoMat = self.coMat_encode(GsImg)
      normMat = self.norm_mat(CoMat)
      energy, entropy, contrast = self.calc_tex_feat(normMat)
      self.energyList.append(energy)
      self.entropyList.append(entropy)
      self.contrastList.append(contrast)
      self.colorCode.append(CcBins)
      self.intenCode.append(InBins)
      self.gsImgList.append(GsImg)
      self.coMatList.append(CoMat)
      self.normMatList.append(normMat)
          
    # Create a feature matrix from the texture features:
    for i in range(len(self.energyList)):
      self.normFeatMat.append([])
      self.normFeatMat[i].append(self.energyList[i])
      self.normFeatMat[i].append(self.entropyList[i])
      self.normFeatMat[i].append(self.contrastList[i])
      for j in range(len(self.colorCode[i])):
        self.normFeatMat[i].append(self.colorCode[i][j])
      for j in range(len(self.intenCode[i])):
        self.normFeatMat[i].append(self.intenCode[i][j])
    self.normFeatMat = np.mat(self.normFeatMat)  
    
    # Gaussian normalization on features within matrix:
    for j in range(self.normFeatMat.shape[1]):
      mean = np.mean(self.normFeatMat[:,j])
      std = np.std(self.normFeatMat[:,j])
      for i in range(self.normFeatMat.shape[0]):
        if std == 0:
          self.normFeatMat[i,j] = 0
        else:
          self.normFeatMat[i,j] = (self.normFeatMat[i,j] - 
                                   mean) / std
    self.normFeatMat = self.normFeatMat.tolist()
  

  # Bin function returns an array of bins for each 
  # image, both Intensity and Color-Code methods.
  def encode(self, im):
    
    # List of pixel Information [R, G, B]
    pixList = list(im.getdata())
    
    # 2D array initilazation for bins, initialized
    # to zero.
    CcBins = [0]*64
    InBins = [0]*25
    
    # Get the most-significant bits in one 6 bit number.
    for pix in pixList:
        
      # Color-Code method.
      red = pix[0] >> 6 << 4
      green = pix[1] >> 6 << 2
      blue = pix[2] >> 6
      pixCode = red | green | blue
      CcBins[pixCode] += 1
      
      # Intensity method.
      R = 0.299*pix[0]
      B = 0.587*pix[1]
      G = 0.114*pix[2]
      pixIntensity = R + B + G
      pixIntensity = int(round(pixIntensity, 0))
      
      if pixIntensity > 239:
        InBins[24] += 1
      else:
        intensityIndex = pixIntensity / 10
        InBins[intensityIndex] += 1
    
    # Return the list of binary digits, one digit for each
    # pixel.
    return CcBins, InBins
  
  
  # Gray-scale intensity function:
  def gs_encode(self, im):
    
    imSize = im.size
    x = imSize[0]
    y = imSize[1]
    
    # List of pixel Information [R, G, B]
    pixList = list(im.getdata())
    
    # Create the two-dimensional List of gray-scale 
    #   intensity values:
    GsImg = []
    k = 0
    for i in range(x):
      GsImg.append([])
      for j in range(y):
        pix = pixList[k]
        R = 0.299*pix[0]
        B = 0.587*pix[1]
        G = 0.114*pix[2]
        I = int(math.floor(R + G + B))
        GsImg[i].append(I)
        k += 1
    
    # Return the gray scale image in list form.
    return GsImg
  
  
  # Gray-scale co-occurrence matrix function:
  def coMat_encode(self, GsImg):
      
    # Initialize the set of possible values to 0:
    set = []
    for i in GsImg:
      for j in i:
        if set.count(j) == 0:
          set.append(j)
    set.sort()
    
    # Create co-occurance matrix:
    CoMat = []
    l = len(set)
    dr = dc = 1
    # Initialization:
    for i in range(l):
      CoMat.append([])
      for j in range(l):
        CoMat[i].append(0)
    # Create with rule:
    # C[i,j] = { [r,c] | I[r,c] = i and I[r+dr, c+dc] = j }
    x = len(GsImg)
    y = len(GsImg[0])
    for r in range(x):
      for c in range(y):
        i = GsImg[r][c]
        if r+dr < x and c+dc < y:
          j = GsImg[r+dr][c+dc]
          CoMat[set.index(i)][set.index(j)] += 1
        
    # Return the co-occurrance matrix:
    return CoMat
    
  # Normalize co-occurance matrix function:
  def norm_mat(self, CoMat):
    
    # Initialize the norm matrix and calculate the sum:
    normMat = []
    sum = 0
    x = len(CoMat)
    y = len(CoMat[0])
    for i in range(x):
      normMat.append([])
      for j in range(y):
        normMat[i].append(0)
        sum += CoMat[i][j]
    
    # Create the normalized matrix and texture features:
    for i in range(x):
      for j in range(y):
        normMat[i][j] = CoMat[i][j] / float(sum)
    
    # Return the normalized co-occurance matrix:
    return normMat
  
  
  # Calculate texture features function:
  def calc_tex_feat(self, normMat):
  
    # Calculate the features at once:
    energy = 0
    entropy = 0
    contrast = 0
    sum = 0
    x = len(normMat)
    y = len(normMat[0])
    for i in range(x):
      for j in range(y):
        n = normMat[i][j]
        if n > 0:
          energy += n**2
          entropy += n * math.log(n, 2)
          contrast += n * (i - j)**2
    
    # Return the features:
    return energy, entropy, contrast

  
  # Accessor functions:
  def get_imageList(self):
    return self.imageList
  
  def get_photoList(self):
    return self.photoList
  
  def get_xmax(self):
    return self.xmax
  
  def get_ymax(self):
    return self.ymax
  
  def get_colorCode(self):
    return self.colorCode
      
  def get_intenCode(self):
    return self.intenCode
  
  def get_gsImgList(self):
    return self.gsImgList
  
  def get_coMatList(self):
    return self.coMatList
  
  def get_energyList(self):
    return self.energyList
  
  def get_entropyList(self):
    return self.entropyList
  
  def get_contrastList(self):
    return self.contrastList

  def get_normFeatMat(self):
    return self.normFeatMat

  def get_normMatList(self):
    return self.normMatList


