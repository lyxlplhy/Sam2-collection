import matplotlib.pyplot as plt
import cv2
a = "E:\era5\LabPicsV1\Simple\Test\Instance\Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot (1).png"
img = cv2.imread(a)
plt.imshow(img*255)
plt.show()