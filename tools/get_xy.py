import os
import cv2
import random
import argparse

clicked = False
g_rectangle = [0, 0, 0, 0]
g_startPoint = [0, 0]


def onMouse(event, x, y, flags, param):
    global clicked
    global g_rectangle
    global g_startPoint
    if event == cv2.EVENT_MOUSEMOVE:
        if clicked == True:
            g_rectangle[0] = min(g_startPoint[0], x)
            g_rectangle[1] = min(g_startPoint[1], y)
            g_rectangle[2] = max(g_startPoint[0], x)
            g_rectangle[3] = max(g_startPoint[1], y)

    # 左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        g_startPoint[0] = x
        g_startPoint[1] = y
        clicked = True
    # 左键弹起事件
    if event == cv2.EVENT_LBUTTONUP:
        print("=======================选中框的坐标：=======================")
        # print("矩形框左上角坐标：")
        # print(g_rectangle[0], g_rectangle[1])
        # print("矩形框右下角坐标：")
        # print(g_rectangle[2], g_rectangle[3])

        rect = (g_rectangle[0], g_rectangle[1], g_rectangle[2], g_rectangle[3])
        print(rect)

        rect_center_x = int((g_rectangle[2] + g_rectangle[0])/2)
        rect_center_y = int((g_rectangle[1] + g_rectangle[3])/2)
        rect_width = int(g_rectangle[2] - g_rectangle[0])
        rect_height = int(g_rectangle[3] - g_rectangle[1])

        # 获取文件名  
        filename = os.path.basename(img_path)  
        
        # 去除后缀  
        filename_without_ext = os.path.splitext(filename)[0] 

        # create labels
        # with open(f'labels/{filename_without_ext}.txt', 'w') as file:  
        #     file.write(f'0 {round(rect_center_x/width, 6)} {round(rect_center_y/height, 6)} {round(rect_width/width, 6)} {round(rect_height/height, 6)}')
        #     print('label已写入')

        clicked = False


def startRoi(path):
    cv2.namedWindow("MyWindow", 0)
    cv2.resizeWindow("MyWindow", 1280, 720)  # 设置长宽
    cv2.setMouseCallback("MyWindow", onMouse)

    print(f"正在框选图像：{path}")
    while cv2.waitKey(30) != 27:
        global img_path, frame, height, width
        img_path = path
        frame = cv2.imread(path)
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (g_rectangle[0], g_rectangle[1]), (g_rectangle[2], g_rectangle[3]), (0, 255, 0), 3)
        cv2.imshow("MyWindow", frame)
        point=[((g_rectangle[2]+g_rectangle[0]))/2,((g_rectangle[3]+g_rectangle[1]))/2]
    cv2.destroyWindow("MyWindow")
    return g_rectangle,point


if __name__ == '__main__':
    # folder_path = '/home/zyh/MyProjects/内托图片/val/6.jpg'

    # extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # image_paths = []

    # # 遍历  
    # for root, dirs, files in os.walk(folder_path):  
    #     for file in files:  
    #         # 获取扩展名
    #         _, ext = os.path.splitext(file)  
    #         ext = ext.lower()   
    #         if ext in extensions:  
    #             image_path = os.path.join(root, file)  # 获取绝对路径
    #             image_paths.append(image_path)

    # for path in image_paths:            
    #     startRoi(path)

    startRoi('E:\LYX_date\cover_date/132_28518.138672.jpg')
