import cv2
import numpy as np
import imutils

from matplotlib import pyplot as pl

#path  = '/home/pashka/Documents/GitHub/Stepik_opencv/nobrain'
path = '/home/gans/Документы/Github/Neuropass_CS/Cytology'

img1 = cv2.imread(path  + '/img/1.jpg')
#img1 = cv2.resize(img1, (img1.shape[1] // 2 ,img1.shape[0] // 2))

img2 = cv2.imread(path  + '/img/2.jpg')
img3 = cv2.imread(path  + '/img/3.jpg')
#img3 = img3[img3.shape[1] // 3 : img3.shape[1], 0 : img3.shape[0] // 2]
img4 = cv2.imread(path  + '/img/4.jpg')

img = img3

#в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cнижение шумов
img_filter = cv2.bilateralFilter(gray, 11, 15, 15)

"""
cv2.bilateralFilter(src, d, sigma_color, sigma_space)
    src — исходное изображение для фильтрации; 4
    d — диаметр окрестности каждого пикселя, используемой во время фильтрации; 4
    sigma_color — фильтр сигма в цветовом пространстве. Большее значение сохраняет более сильные края; 4
    sigma_space — фильтр сигма в координатном пространстве. Большее значение рассматривает более далёкие пиксели внутри окрестности.
"""


#определение контуров
edges = cv2.Canny(img_filter, 5, 150)

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


cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]

pos = None
for i in cont:
    approx = cv2.approxPolyDP(i, 10, True)
    if len(approx) == 4:
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask==255)
(x1, y1) = np.min(x), np.min(y)
(x2, y2) = np.max(x), np.max(y)
crop = gray[x1:x2, y1:y2]


#pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#pl.show()
"""

cv2.imshow('res', img)
cv2.waitKey(0)
cv2.imshow('res', gray)
cv2.waitKey(0)
cv2.imshow('res', img_filter)
cv2.waitKey(0)
cv2.imshow('res', edges)
cv2.waitKey(0)
