import cv2
import numpy as np
import random
import math

#read the image 
image = cv2.imread("download.jpg",0)
cv2.namedWindow("Watermarking",cv2.WINDOW_AUTOSIZE)

#divide the image by 255.0 
floatImg = np.float64(image)/255.0

#calculate dct
dctImg = cv2.dct(floatImg)

#display the image
cv2.imshow("DCT IMAGE 1",dctImg)

#copy the dct image
copyDct = np.copy(dctImg)

#list to store each pixel in a list
pixels = []

for i in xrange(0,156):
    for j in xrange(0,240):
        cpixel = copyDct[i][j]
        pixels.append(cpixel)

pixels.sort()

#set the strength of watermark
alpha = 0.01

#generate random sequence using a particular key
rankey = 2
random.seed(rankey)
sequence = []

#create two dimensional location matrix
loc = np.zeros([500, 2])

for i in xrange(0, 500):
    p = random.random()
    sequence.append(p)
    for j in xrange(156):
        chk =0
        for k in xrange(240):
            if copyDct[j][k] == pixels[i]:
                copyDct[j][k] = copyDct[j][k] + alpha*p*copyDct[j][k]
                loc[i][0] = j
                loc[i][1] = k
                chk =1
                break
        if(chk):
            break
#print(np.matrix(loc[0:20,:]))
print sequence[0:10]


invDct = cv2.idct(copyDct)
cv2.imshow("Inverse",invDct)
cv2.imshow("DCT Watermarked",copyDct)

#Detection
#print((invDct))
#print np.max(invDct)
#print(np.sum(invDct-floatImg))
#print(copyDct==dctImg)

#add noise to the image
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.float64)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output
print np.sum(sequence)
invDct = sp_noise(invDct,0.01)
cv2.imshow("After noise",invDct)
newImg = cv2.dct(invDct)
#sequence
wtSequence=[]
for i in xrange(0,500):
    img = newImg[loc[i,0]][loc[i,1]]
    g = (img-dctImg[loc[i,0],loc[i,1]])/(alpha*dctImg[loc[i,0],loc[i,1]])
    wtSequence.append(g)

print wtSequence[0:10]
print np.sum(wtSequence)
#wait for input
k = cv2.waitKey(0)
