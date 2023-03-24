

import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
import matplotlib
from PIL import Image,ImageTk
import os
import sys
######################################################
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
#import skimage
# from skimage import util
###################################################################

class image_process:

  #def __loadimg__(self,path):
      #self.img1 = cv.imread(path)
      #return self.img1


  #defult img


  def Defultimg(self,img):
      return img

   # Converting to grayscale

  def Grayimg(self,img):
      img = cv.imread(img)
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      return img



  # NOise


  def GaussianNoise(self,img):
      img = cv.imread(img)
      img = cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)
      return img

  def Salt_PepperNoise(self,img):
      img=cv.imread(img)
      self.rows ,self.columns,self.channal = img.shape
      self.p = 0.05
      self.output  = np.copy(img)
      for i in range(self.rows):
          for j in range(self.columns):
              self.r= random.random()
              if self.r < self.p/2:
                  #pepper noise
                  self.output[i][j]=[0,0,0]
              elif self.r<self.p:
                   # salt noise
                   self.output[i][j]= [255,255,255]
              else :
                    self.output[i][j]= img[i][j]
      img = self.output
      return img
      #######################################################
      ###another salt and noise pepper  important#############################################
      #img=cv.imread(img)
      # self.row,self. col, self.ch = img.shape
      #
      # self.s_vs_p = 0.3
      # self.amount = 0.4
      # self.out = np.copy(img)
      # # Salt mode
      # self.num_salt = np.ceil(self.amount * img.size * self.s_vs_p)
      # self.coords = [np.random.randint(0, i - 1, int(self.num_salt))
      #                   for i in img.shape]
      # self.out[self.coords] = 1
      #
      # # Pepper mode
      # self.num_pepper = np.ceil(self.amount * img.size * (1. - self.s_vs_p))
      # self.coords = [np.random.randint(0, i - 1, int(self.num_pepper))
      #                   for i in img.shape]
      # self.out[self.coords] = 0
      # return self.out
      # ###############################################################


  def PoissonNoise(self,img):
      img=self.GaussianNoise(img)
      self.row,self. col, self.ch = img.shape

      self.s_vs_p = 0.3
      self.amount = 0.4
      self.out = np.copy(img)
      # Salt mode
      self.num_salt = np.ceil(self.amount * img.size * self.s_vs_p)
      self.coords = [np.random.randint(0, i - 1, int(self.num_salt))
                        for i in img.shape]
      self.out[self.coords] = 1

      # Pepper mode
      self.num_pepper = np.ceil(self.amount * img.size * (1. - self.s_vs_p))
      self.coords = [np.random.randint(0, i - 1, int(self.num_pepper))
                        for i in img.shape]
      self.out[self.coords] = 0
      return self.out
      ##another solve
      #img=cv.imread(img)
      # img=Salt_PepperNoise(self.self.GaussianNoise())
      # return img
      ########another solve
      #img=cv.imread(img)
      # self.vals = len(np.unique(img))
      # self.vals = 2 ** np.ceil(np.log2(self.vals))
      # img = np.random.poisson(img * self.vals) / float(self.vals)
      # return img

  #point transforms

  def BritnessImg(self,img):
      img = cv.imread(img)
      self.img7=cv.convertScaleAbs(img,alpha=1 , beta=100)
      return self.img7

  def ContrastImg(self,img):
     img = cv.imread(img)
     self.img8 = cv.convertScaleAbs(img, alpha=2.5, beta=0)
     return self.img8

  def HistogramImg(self,img):
      img = cv.imread(img)
      plt.hist(img.ravel(),256,[0,256])
      plt.savefig('photo\hist.png' ,dpi=500)
      plt.close()
      return 0


  def HistogramEqualImg(self,img):
      img = cv.imread(img)
      self.l,self.a,self.b=cv.split(img)
      self.hist=cv.equalizeHist(self.l)
      plt.hist(self.hist.flat,bins=100,range=(0,255))
      plt.savefig('photo\hist.png', dpi=500)
      plt.close()
      return 0

  #local transformation

  def LowPassFilter(self,img):
      img = cv.imread(img)
      self.k=np.ones((3,3),np.float32)/9
      self.img9=cv.filter2D(img,-1,self.k)
      return  self.img9

  def AveragingFilter(self,img):
      img = cv.imread(img)
      self.kn=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
      self.img10 = cv.filter2D(img, -1, self.kn)
      return self.img10

  def MedianFilter(self,img):
      # img = cv.imread(img)
      self.img11 = cv.medianBlur(self.Grayimg(img),3)
      return self.img11


  #edge detection filters

  def HighPassFilter(self,img):
      img = cv.imread(img)
      self.kn = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
      self.img12 = cv.filter2D(img, -1, self.kn,borderType=cv.BORDER_CONSTANT)
      return self.img12

  def GaussianFilter(self,img):
      img = self.Grayimg(img)
      self.img13 = cv.GaussianBlur(img,(3,3),0)
      return self.img13

  def LaplacianFilter(self,img):
      img = cv.imread(img,cv.IMREAD_GRAYSCALE)
      self.kn = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
      self.img14=cv.filter2D(img, -1, self.kn,borderType=cv.BORDER_CONSTANT)
      return self.img14

  def Vert_SobelFilter(self,img):
      img = self.Grayimg(img)
      self.kn = np.array([[-1, -2, -1], [0,0,0], [1, 2, 1]])
      self.img15 = cv.filter2D(img, -1, self.kn,borderType=cv.BORDER_CONSTANT)
      return self.img15

  def Horiz_SobelFilter(self,img):
      img = self.Grayimg(img)
      self.kn = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
      self.img16 = cv.filter2D(img, -1, self.kn,borderType=cv.BORDER_CONSTANT)
      return self.img16

  def Vert_PrewittxFilter(self,img):
      img = self.Grayimg(img)
      self.kn = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
      self.img17 = cv.filter2D(img, -1, self.kn)
      return self.img17

  def Horiz_PrewittxFilter(self,img):
      img = self.Grayimg(img)
      self.kn = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
      self.img18 = cv.filter2D(img, -1, self.kn)
      return self.img18

  def CannyFilter(self,img):
      img = self.Grayimg(img)
      self.img19 = cv.Canny(img,100,200)
      return self.img19

  def Lap_GusFilter(self,img):
      # img = self.Grayimg(img)
      # self.blur = cv.GaussianBlur(img, (3, 3), 0)
      # self.laplacian = cv.Laplacian(self.blur, cv.CV_64F)
      # self.img20 = self.laplacian / self.laplacian.max()
      # return self.img20
      img=self.GaussianFilter(img)
      self.kn = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
      img = cv.filter2D(img, -1, self.kn, borderType=cv.BORDER_CONSTANT)
      return img

  def Zero_crossingFilter(self,img):
      img = self.Grayimg(img)
      # self.z_c_image = np.copy(img.shape)
      #
      # # For each pixel, count the number of positive
      # # and negative pixels in the neighborhood
      #
      # for i in range(1, img.shape[0] - 1):
      #     for j in range(1, img.shape[1] - 1):
      #         self.negative_count = 0
      #         self.positive_count = 0
      #         self.neighbour = [img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1], img[i, j - 1], img[i, j + 1],
      #                      img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1]]
      #         d = max(self.neighbour)
      #         e = min(self.neighbour)
      #         for h in self.neighbour:
      #             if h > 0:
      #                 self.positive_count += 1
      #             elif h < 0:
      #                 self.negative_count += 1
      #
      #         # If both negative and positive values exist in
      #         # the pixel neighborhood, then that pixel is a
      #         # potential zero crossing
      #
      #         z_c = ((self.negative_count > 0) and (self.positive_count > 0))
      #
      #         # Change the pixel value with the maximum neighborhood
      #         # difference with the pixel
      #
      #         if z_c:
      #             if img[i, j] > 0:
      #                 self.z_c_image[i, j] = img[i, j] + np.abs(e)
      #             elif img[i, j] < 0:
      #                 self.z_c_image[i, j] = np.abs(img[i, j]) + d
      #
      # # Normalize and change datatype to 'uint8' (optional)
      # self.z_c_norm = self.z_c_image / self.z_c_image.max() * 255
      # self.z_c_image = np.uint8(self.z_c_norm)
      # self.img21 =self.z_c_image
      # return self.img21
      self.LoG = cv.Laplacian(img, cv.CV_16S)
      self.minLoG = cv.morphologyEx(self.LoG, cv.MORPH_ERODE, np.ones((3, 3)))
      self.maxLoG = cv.morphologyEx(self.LoG, cv.MORPH_DILATE, np.ones((3, 3)))
      self.zeroCross = np.logical_or(np.logical_and(self.minLoG < 0, self.LoG > 0), np.logical_and(self.maxLoG > 0, self.LoG < 0))
      return self.zeroCross

  def ThickenFilter(self,img):
      img = self.Grayimg(img)
      #self.k = np.ones((5, 5), np.uint8)
      self.k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
      self.img22 = cv.dilate(img, self.k , iterations=1)
      return self.img22

  def ThinningFilter(self,img):
      img = self.Grayimg(img)
      #self.k = np.ones((5, 5), np.uint8)
      self.k = kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
      self.img23 = cv.erode(img, self.k , iterations=1)
      return self.img23

  def Skletonfilter(self,img):
      img = self.Grayimg(img)
      self.skel = img.copy()

      self.skel[:,:] = 0
      self.kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

      while True:
          self.eroded = cv.morphologyEx(img, cv.MORPH_ERODE, self.kernel)
          self.temp = cv.morphologyEx(self.eroded, cv.MORPH_DILATE, self.kernel)
          self.temp = cv.subtract(img, self.temp)
          self.skel = cv.bitwise_or(self.skel, self.temp)
          img[:, :] = self.eroded[:, :]
          if cv.countNonZero(img) == 0:
              break
      self.img24=self.skel
      return self.img24


  #global transform

  def LineDetection(self,img):
      img = self.Grayimg(img)
      self.lines = cv.HoughLines(img, 1, np.pi / 180, 200)
      for self.rho, self.theta in self.lines[0]:
          self.a = np.cos(self.theta)
          self.b = np.sin(self.theta)
          self.x0 = self.a * self.rho
          self.y0 = self.b * self.rho
          self.x1 = int(self.x0 + 1000 * (-self.b))
          self.y1 = int(self.y0 + 1000 * (self.a))
          self.x2 = int(self.x0 - 1000 * (-self.b))
          self.y2 = int(self.y0 - 1000 * (self.a))

          self.img25 = cv.line(img, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)
      return self.img25

  def CircleDetection(self,img):
      img = self.Grayimg(img)
      self.detected_circles = cv.HoughCircles(img,
                                          cv.HOUGH_GRADIENT, 1, 20, param1=50,
                                          param2=30, minRadius=1, maxRadius=40)

      # Draw circles that are detected.
      if self.detected_circles is not None:

          # Convert the circle parameters a, b and r to integers.
          self.detected_circles = np.uint16(np.around(self.detected_circles))

          for self.pt in self.detected_circles[0, :]:
              self.a, self.b, self.r = self.pt[0], self.pt[1], self.pt[2]

              # Draw the circumference of the circle.
              self.img26=cv.circle(img, (self.a, self.b), self.r, (0, 255, 0), 2)

              # Draw a small circle (of radius 1) to show the center.
              #cv2.circle(img, (self.a, self.b), 1, (0, 0, 255), 3)
      # self.pt=[7,7]
      # self.img26=cv.findCirclesGrid(img,self.pt)
      return self.img26


  #morphological transform

  def DilationFilter(self,img):
      img = self.Grayimg(img)
      self.k = np.ones((5, 5), np.uint8)
      self.img27 = cv.dilate(img, self.k , iterations=1)
      return self.img27

  def ErrotionFilter(self,img):
      img = self.Grayimg(img)
      self.k = np.ones((5, 5), np.uint8)
      self.img28 = cv.erode(img, self.k , iterations=1)
      return self.img28

  def OpeningFilter(self,img):
      img = self.Grayimg(img)
      self.k = np.ones((5, 5), np.uint8)
      self.img29 = cv.morphologyEx(img, cv.MORPH_OPEN, self.k )
      return self.img29

  def ClosingFilter(self,img):
      img = self.Grayimg(img)
      self.k = np.ones((5, 5), np.uint8)
      self.img30 = cv.morphologyEx(img, cv.MORPH_CLOSE, self.k )
      return self.img30


#####################################################################
window = Tk()

window.geometry('1100x1100')
window.title('image processing project')

#secound
###############################################################################
#variable
f=None
photo=None
img1=None
arrimg=None
resPhoto=None
p=image_process()
var =IntVar()
var.set(0)
var2 = IntVar()
var2.set(0)
var3 = IntVar()
var3.set(0)
################################################################################
#frame image out
img = Image.open("photo\\er.jpg")
img = img.resize((420, 200), Image.ANTIALIAS)
img0 = ImageTk.PhotoImage(img)
##########
frame9=ttk.LabelFrame(window, text="orgnal image" ,width=700 ,height=230)
frame9.place(x=660,y=18)
# frame9.config(padding =(15,15))
frame9.config(relief=RAISED , borderwidth=5)
l1 =ttk.Label(frame9, image=img0)
l1.pack()
##########################################
frame10=ttk.LabelFrame(window, text="gray and noise image",width=700 ,height=230)
frame10.place(x=660,y=250)
frame10.config(relief=RAISED , borderwidth=5)
l2 =ttk.Label(frame10, image=img0)
l2.pack()
##########################################
frame11=ttk.LabelFrame(window, text="final redult image" ,width=700 ,height=230)
frame11.place(x=660,y=480)
frame11.config(relief=RAISED , borderwidth=5)
l3 =ttk.Label(frame11, image=img0)
l3.pack()
############################################################################
#function
def open():
    global photo,f,img1,arrimg,l1,dim
    dim=(300,300)
    f = fd.askopenfilename(title='Select content image',
                                filetypes=[('image', '*.jpg'),
                                           ('All Files', '*')] )
    l1.destroy()
    img1 = Image.open(f)
    arrimg = cv.imread(f)
    # arrimg = cv.resize(arrimg,dim,interpolation = cv.INTER_AREA)
    img1 = img1.resize((420, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img1)
    l1 =Label(frame9,image=photo)
    l1.pack()




def img():
   global resImg, resPhoto, funimg, var,l2

   if var.get() ==2:
      l2.destroy()
      funimg= p.Grayimg(f)
      resImg=Image.fromarray(funimg)
      resImg = resImg.resize((420, 200), Image.ANTIALIAS)
      resPhoto =ImageTk.PhotoImage(resImg)
      l2 = Label(frame10, image=resPhoto)
      l2.pack()
      return funimg
   elif var.get() ==1:
       l2.destroy()
       resPhoto = p.Defultimg(photo)
       l2 = Label(frame10, image=resPhoto)
       l2.pack()
       funimg = cv.imread(f)
       return funimg
   else:
       return arrimg


def noise():
    global var2 ,rimg,l3,l2,resImg,resPhoto
    if var2.get()==1:
        l2.destroy()
        rimg=p.Salt_PepperNoise(f)
        resImg = Image.fromarray(rimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l2 = Label(frame10, image=resPhoto)
        l2.pack()
        return rimg
    elif var2.get()==2:
        l2.destroy()
        rimg=p.GaussianNoise(f)
        resImg = Image.fromarray(rimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l2 = Label(frame10, image=resPhoto)
        l2.pack()
        return rimg
    elif var2.get()==3:
        l2.destroy()
        rimg=p.PoissonNoise(f)
        resImg = Image.fromarray(rimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l2 = Label(frame10, image=resPhoto)
        l2.pack()
        return rimg
    else:
        return arrimg

def britness():
    global l3,britimg,resImg,resPhoto
    britimg=p.BritnessImg(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def contrast():
    global l3,britimg,resImg,resPhoto
    britimg=p.ContrastImg(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def hist():
    global l3, britimg,img0
    p.HistogramImg(f)
    l3.destroy()
    img = Image.open("photo\\hist.png")
    img = img.resize((420, 200), Image.ANTIALIAS)
    img0 = ImageTk.PhotoImage(img)
    l3 = Label(frame11, image=img0)
    l3.pack()


def histeq():
    global l3, britimg,img0,histimg,resImg,resPhoto
    p.HistogramEqualImg(f)
    l3.destroy()
    img = Image.open("photo\\hist.png")
    img = img.resize((420, 200), Image.ANTIALIAS)
    img0 = ImageTk.PhotoImage(img)
    l3 = Label(frame11, image=img0)
    l3.pack()

def lowpass():
    global l3,britimg,resImg,resPhoto
    britimg=p.LowPassFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def highpass():
    global l3,britimg,resImg,resPhoto
    britimg=p.HighPassFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def medpass():
    global l3,britimg,resImg,resPhoto
    britimg=p.MedianFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def averpass():
    global l3,britimg,resImg,resPhoto
    britimg=p.AveragingFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg
#################
#filters

def filter():
    global l3, britimg, resImg, resPhoto
    if var3.get()==1:
        britimg = p.LaplacianFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==2:
        britimg = p.GaussianFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==3:
        britimg = p.Vert_SobelFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==4:
        britimg = p.Horiz_SobelFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==5:
        britimg = p.Vert_PrewittxFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==6:
        britimg = p.Horiz_PrewittxFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==7:
        britimg = p.Lap_GusFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==8:
        britimg = p.CannyFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==9:
        britimg = p.Zero_crossingFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==10:
        britimg = p.ThickenFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==11:
        britimg = p.Skletonfilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg
    elif var3.get()==12:
        britimg = p.ThinningFilter(f)
        l3.destroy()
        resImg = Image.fromarray(britimg)
        resImg = resImg.resize((420, 200), Image.ANTIALIAS)
        resPhoto = ImageTk.PhotoImage(resImg)
        l3 = Label(frame11, image=resPhoto)
        l3.pack()
        return rimg

#################
def save():
    global fl,resPhoto
    fl = fd.asksaveasfilename(initialdir=os.getcwd(),title="Save Image",filetypes=(("JPGFile","*.jpg"),("PNG File","*.png"),("PDF File",".pdf")))
    cv.imwrite(fl,resPhoto)


###################################

def linedetect():
    global l3,britimg,resImg,resPhoto
    britimg=p.LineDetection(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def circledetect():
    global l3,britimg,resImg,resPhoto
    britimg=p.CircleDetection(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def close():
    global l3,britimg,resImg,resPhoto
    britimg=p.ClosingFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def dilate():
    global l3,britimg,resImg,resPhoto
    britimg=p.DilationFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg
def errod():
    global l3,britimg,resImg,resPhoto
    britimg=p.ErrotionFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg

def opendetect():
    global l3,britimg,resImg,resPhoto
    britimg=p.OpeningFilter(f)
    l3.destroy()
    resImg = Image.fromarray(britimg)
    resImg = resImg.resize((420, 200), Image.ANTIALIAS)
    resPhoto = ImageTk.PhotoImage(resImg)
    l3 = Label(frame11, image=resPhoto)
    l3.pack()
    return rimg
######################################################################################

#gui

#######################

frame1= ttk.Labelframe(window,text='Load Image' ,width=50,height=5)
frame1.place(x=10,y=18)
#frame1.config(height=990,width=950)
frame1.config(relief=RAISED , borderwidth=5)
#frame1.config(padding =(5,5))
b1=Button(frame1,text='Open Image', relief=RAISED ,cursor='hand2', width=40,height=2 ,activebackgroun='green',activeforegroun='red' ,command = open).pack(padx=10,pady=10)


#############################################################################################

frame2=ttk.LabelFrame(window, text='Convert' ,width=50,height=5)
frame2.place(x=346,y=18)
frame2.config(relief=RAISED , borderwidth=5)
radio1 = ttk.Radiobutton(frame2, text='Default Color', variable=var , value=1 ,command=img)
radio1.pack(side=TOP ,padx=5,pady=5)
radio2 = ttk.Radiobutton(frame2, text='Gray Color     ', variable=var , value=2,command=img)
radio2.pack(side=BOTTOM,padx=5,pady=5)
############################################################################

frame3=ttk.LabelFrame(window, text='Add Noise' ,width=30,height=5)
frame3.place(x=477,y=18)
frame3.config(relief=RAISED , borderwidth=5)

radio1 = ttk.Radiobutton(frame3, text='Salt & Pepper Noise',variable=var2 , value=1 ,command=noise)
radio1.pack(side=TOP ,padx=10 )
radio2 = ttk.Radiobutton(frame3, text='Gaussian Noise        ', variable=var2 , value=2 ,command=noise)
radio2.pack(padx=10)
radio3 = ttk.Radiobutton(frame3, text='Poisson Noise          ', variable=var2 , value=3 ,command=noise)
radio3.pack(side=BOTTOM ,padx=10 )

##########################################################################################

frame4=ttk.LabelFrame(window, text="Point Transform Op's" ,width=626 ,height=150)
frame4.place(x=10,y=110)
frame4.config(relief=RAISED , borderwidth=5)
b2=Button(frame4,text='Brightness adjustment', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=britness ).place(x=0,y=0)
b3=Button(frame4,text='Contrast adjustment', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=contrast ).place(x=150,y=30)
b4=Button(frame4,text='Histogram adjustment', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red' ,command=hist).place(x=300,y=60)
b5=Button(frame4,text='HistogramEqual adjustment', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=histeq ).place(x=450,y=90)
############################################################################################################################
frame5=ttk.LabelFrame(window, text="Local Transform Op's" ,width=626 ,height=150)
frame5.place(x=10,y=263)

frame5.config(relief=RAISED , borderwidth=5)
b6=Button(frame5,text='High Pass Filter', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=highpass ).place(x=0,y=0)
b7=Button(frame5,text='Low Pass Filter', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=lowpass ).place(x=0,y=30)
b8=Button(frame5,text='Averaging Filter', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=averpass ).place(x=0,y=60)
b9=Button(frame5,text='Median Filter', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red' ,command=medpass).place(x=0,y=90)

frame6=ttk.LabelFrame(frame5, text="Edge detection filters" ,width=450 ,height=110)
frame6.place(x=160,y=0)
frame6.config(relief=RAISED , borderwidth=5)

radio4 = ttk.Radiobutton(frame6, text='Laplacian filiter       ', variable=var3 , value=1,command=filter)
radio4.place(x=0,y=0)
radio4 = ttk.Radiobutton(frame6, text='Gaussian filiter        ', variable=var3, value=2,command=filter)
radio4.place(x=0,y=30)
radio4 = ttk.Radiobutton(frame6, text='Vert. Sobel             ', variable=var3, value=3,command=filter)
radio4.place(x=0,y=60)
radio4 = ttk.Radiobutton(frame6, text='Horiz. Sobel            ', variable=var3, value=4,command=filter)
radio4.place(x=100,y=0)
radio4 = ttk.Radiobutton(frame6, text='Vert. Prewitt           ', variable=var3, value=5,command=filter)
radio4.place(x=100,y=30)
radio4 = ttk.Radiobutton(frame6, text='Horiz. Prewitt          ', variable=var3, value=6,command=filter)
radio4.place(x=100,y=60)
radio4 = ttk.Radiobutton(frame6, text='lap of Gau(log)         ', variable=var3, value=7,command=filter)
radio4.place(x=200,y=0)
radio4 = ttk.Radiobutton(frame6, text='Canny method            ', variable=var3, value=8,command=filter)
radio4.place(x=200,y=30)
radio4 = ttk.Radiobutton(frame6, text='Zero Cross              ', variable=var3, value=9,command=filter)
radio4.place(x=200,y=60)
radio4 = ttk.Radiobutton(frame6, text='Thicken                 ', variable=var3, value=10,command=filter)
radio4.place(x=300,y=0)
radio4 = ttk.Radiobutton(frame6, text='Skeleton                ', variable=var3, value=11,command=filter)
radio4.place(x=300,y=30)
radio4 = ttk.Radiobutton(frame6, text='Thinning                ', variable=var3, value=12,command=filter)
radio4.place(x=300,y=60)

########################################################################################################################
frame7=ttk.LabelFrame(window, text="Global Transform Op's" ,width=100 ,height=100)
frame7.place(x=10,y=425)

frame7.config(relief=RAISED , borderwidth=5)
b1=Button(frame7,text='Line detection using H.T', relief=RAISED ,cursor='hand2', width=40,height=2 ,activebackgroun='green',activeforegroun='red',command=linedetect ).pack(padx=10,pady=10)
b1=Button(frame7,text='Circles detection using H.T', relief=RAISED ,cursor='hand2', width=40,height=2 ,activebackgroun='green',activeforegroun='red' ,command=circledetect).pack(padx=10,pady=10)

########################################################################################################
frame8=ttk.LabelFrame(window, text="Morphological Op's" ,width=300 ,height=146)
frame8.place(x=335,y=425)

frame8.config(relief=RAISED , borderwidth=5)
b6=Button(frame8,text='Dilation', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=dilate ).place(x=0,y=0)
b7=Button(frame8,text='Erosion', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=errod ).place(x=0,y=30)
b8=Button(frame8,text='Close', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=close ).place(x=0,y=60)
b9=Button(frame8,text='Open', relief=RAISED ,cursor='hand2', width=20,height=1 ,activebackgroun='green',activeforegroun='red',command=opendetect ).place(x=0,y=90)

co=ttk.Combobox(frame8 , width=15)
co.place(x=160,y=45)
co.config(values=('arbitrary','rectangle','periodicline'))
co.set('arbitrary')
########################################################################################################################
b10=Button(window,text='Save Result image ', relief=RAISED ,cursor='hand2', width=30,height=2 ,activebackgroun='green',activeforegroun='red' ,command=save).place(x=40,y=625)
b11=Button(window,text='Exit', relief=RAISED ,cursor='hand2', width=30,height=2 ,activebackgroun='green',activeforegroun='red',command=window.destroy ).place(x=385,y=625)


########################################################################################################################
window.mainloop()

##################################################################################################################################





