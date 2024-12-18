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

lower_red = np.array([139,0,209])
upper_red = np.array([255,255,255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)

lower_blue = np.array([68,12,0])
upper_blue = np.array([100,255,255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)



"""
cv2.inRange — это функция в OpenCV, которая позволяет наложить на кадр цветовой фильтр в заданном диапазоне. 3
Функция принимает три параметра: изображение, нижнее и верхнее значения диапазона. Она возвращает бинарную маску (ndarray из единиц и нулей) размером
с оригинальное изображение. Единицы обозначают значения в пределах диапазона, а нули — вне его. 2
Пример использования: маска красного цвета mask1 = cv2.inRange(img, (0, 0, 50), (50, 50, 255))

"""

target_red = cv2.bitwise_and(img,img, mask=mask_red)
target_blue = cv2.bitwise_and(img,img, mask=mask_blue)

export_temp_img(target_red, 'red.jpg', str(i))
i = i + 1

export_temp_img(target_blue, 'blue.jpg', str(i))
i = i + 1

target = target_blue

#в оттенки серого
gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
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



cont = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


"""
Функция cv2.findContours в OpenCV используется для идентификации и извлечения контуров из бинарных или монохромных изображений. 
Контуры — это границы объектов или фигур на изображении. 

У функции три аргумента: 4

Исходное изображение. 4 Это должно быть 8-битное изображение, для работы используется монохромное изображение, 
где все пиксели с ненулевым цветом будут интерпретироваться как 1, а все нулевые — как 0. 1
Режим группировки. Один из четырёх режимов группировки найденных контуров: 1
CV_RETR_LIST — выдаёт все контуры без группировки; 1
CV_RETR_EXTERNAL — выдаёт только крайние внешние контуры; 1
CV_RETR_CCOMP — группирует контуры в двухуровневую иерархию; 1
CV_RETR_TREE — группирует контуры в многоуровневую иерархию. 1

Метод упаковки. Один из трёх методов упаковки контуров: 1
CV_CHAIN_APPROX_NONE — упаковка отсутствует и все контуры хранятся в виде отрезков, состоящих из двух пикселей; 1
CV_CHAIN_APPROX_SIMPLE — склеивает все горизонтальные, вертикальные и диагональные контуры. 1
Функция возвращает список всех найденных контуров, представленных в виде векторов, иерархию контуров и сдвиг. 1

Для отображения контуров можно использовать функцию drawContours.
"""


cont = imutils.grab_contours(cont)

"""
imutils.grab_contours — это функция из библиотеки imutils, которая извлекает реальные контуры из результата cv.findContours
"""
#cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]

pos = None
for i in cont:
    approx = cv2.approxPolyDP(pos, 0.01 * cv2.arcLength(pos, True), True)

    """
    cv2.approxPolyDP() в OpenCV используется для аппроксимации многоугольной кривой, уменьшая количество вершин при сохранении общей формы кривой. 2

Функция принимает четыре параметра: 1

curve — входной вектор двухмерных точек, сохранённый в std::vector или Mat. 1
approxCurve — результат аппроксимации, тип которого должен соответствовать типу входной кривой. 1
epsilon — параметр, характеризующий точность аппроксимации. 3 Меньшие значения эпсилона приведут к кривым, 
которые будут близко соответствовать исходным, а большие — к более упрощённым кривым. 2
closed — параметр, указывающий, является ли кривая аппроксимации закрытой. 14
Для аппроксимации используется алгоритм Дугласа-Пеукера.
    """


    if len(approx) == 4:
        pos = approx
        break

mask_cont = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask_cont, [pos], 0, 255, -1)
cv2.imshow('res', new_img)

bitwise_img = cv2.bitwise_and(img, img, mask=mask_cont)

(x, y) = np.where(mask_cont==255)
(x1, y1) = np.min(x), np.min(y)
(x2, y2) = np.max(x), np.max(y)
crop = gray[x1:x2, y1:y2]


pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
pl.show()





#вывод на экран этапов обработки изображений
temp_images = os.listdir(temp)
temp_images.sort()
print(temp_images)

for temp_image in temp_images:
    i = cv2.imread(temp + temp_image)
    cv2.imshow('res', i)
    cv2.waitKey(0)


