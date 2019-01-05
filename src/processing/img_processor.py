import cv2

def resize_img(img,newshape):
    img_resized = cv2.resize(img, dsize=newshape)
    return img_resized