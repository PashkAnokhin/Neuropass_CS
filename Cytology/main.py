import cv2
import numpy as np
import imutils
import os

from matplotlib import pyplot as pl

path = '/home/gans/Документы/Github/Neuropass_CS/Cytology'
temp = path + '/temp/'
temp_images = os.listdir(temp)
i = 1

for temp_image in temp_images:
    os.remove(temp + temp_image)

def export_temp_img(image, name, num):
    cv2.imwrite(path + '/temp/' + num + '_' + name, image)
   

img1 = cv2.imread(path  + '/img/1.jpg')
img2 = cv2.imread(path  + '/img/2.jpg')
img3 = cv2.imread(path  + '/img/3.jpg')
#img3 = img3[img3.shape[1] // 3 : img3.shape[1], 0 : img3.shape[0] // 2]
img4 = cv2.imread(path  + '/img/4.jpg')

img = img3
export_temp_img(img, 'current.jpg', str(i))
i = i + 1

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)
target = cv2.bitwise_and(img,img, mask=mask1)

cv2.imshow('target', target)
cv2.waitKey(0)

#в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
export_temp_img(gray, 'gray.jpg', str(i))
i = i + 1


#cнижение шумов
img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
export_temp_img(img_filter, 'filter.jpg', str(i))
i = i + 1

"""
cv2.bilateralFilter(src, d, sigma_color, sigma_space)
    src — исходное изображение для фильтрации; 4
    d — диаметр окрестности каждого пикселя, используемой во время фильтрации; 4
    sigma_color — фильтр сигма в цветовом пространстве. Большее значение сохраняет более сильные края; 4
    sigma_space — фильтр сигма в координатном пространстве. Большее значение рассматривает более далёкие пиксели внутри окрестности.
"""


#определение контуров
edges = cv2.Canny(img_filter, 5, 150)
export_temp_img(edges, 'edges.jpg', str(i))
i = i + 1

"""
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
    image — входной/исходный массив изображения; 1
    threshold1 — первый порог для процедуры гистерезиса; 1
    threshold2 — второй порог для процедуры гистерезиса; 1
    edges (опционально) — количество каналов в выходном изображении; 1
    apertureSize (опционально) — размер диафрагмы оператора; 1
    L2gradient (опционально) — флаг, указывающий, следует ли использовать более точное значение нормы L2 при вычислении градиента.
"""




"""
#вывод на экран этапов обработки изображений
temp_images = os.listdir(temp)
temp_images.sort()
print(temp_images)

for temp_image in temp_images:
    i = cv2.imread(temp + temp_image)
    cv2.imshow(temp_image, i)
    cv2.waitKey(0)


"""