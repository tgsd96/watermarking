import cv2
import numpy as np
import random
import math
#from scipy import spatial
np.set_printoptions(threshold="nan")
img = cv2.imread("C:\\Python27\\download.jpg",0)
image=img
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
p=img.shape
print p
imf=np.float32(img)/255.0
img=cv2.dct(imf)
cv2.imshow("image",img)
img_dct=img
all_pixels=[]
w=[]
for i in range(0,156,1):
    for j in range(0,240,1):
        cpixel = img[i][j]
        all_pixels.append(cpixel)

all_pixels.sort(reverse=True)
alpha=0.01
k, h = 2, 500

random.seed(1)
Matrix = [[0 for x in range(k)] for y in range(h)]
for i in range(0,500,1):
    p = random.random()
    w.append(p)
    for j in range(0, 156, 1):
        for k in range(0,240,1):
            if all_pixels[i]==img[j][k]:
                img[j][k]=img[j][k]+ alpha * p*img[j][k]
                Matrix[i][0]=j
                Matrix[i][1]=k

print Matrix
print len(Matrix)
print w
imf=np.float32(img)
dst=cv2.idct(imf)
cv2.imshow("image",dst)

print dst
#cv2.imshow("image",dst[:,10:100])
#-------------detection--------------
reverse_list=[]
water_mark=[]
imf=np.float32(dst)
dst=cv2.dct(imf)
cv2.imshow("image",dst)
for i in range(0,500,1):
    q=dst[Matrix[i][0]][Matrix[i][1]]
    g=(q-img_dct[Matrix[i][0]][Matrix[i][1]])/(alpha*img_dct[Matrix[i][0]][Matrix[i][1]])
    print q
    reverse_list.append(g)

print reverse_list

# for i in range(0,500,1):
#     reverse_list.append((dst[Matrix[i][0]][Matrix[i][1]]))
# print 'reverse list'
# print reverse_list
# sum1=0.0
# for i in range(0,500,1):
#     c1=img_dct[Matrix[i][0]][Matrix[i][1]]
#     c2=reverse_list[i]
#     c=((c2/c1-1)*1)/alpha
#     sum1=sum1+c
#    # print sum1
#    # water_mark.append(c)
#
# print 'sum1'
# print sum1
#
# v=sum1/math.sqrt(sum1)
# print 1-v

k = cv2.waitKey()
if k == ord('s'):
    cv2.imwrite('C:\\Python27\\write3.jpg', img)
    cv2.destroyAllWindows()