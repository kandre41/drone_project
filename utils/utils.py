import cv2
import numpy as np
def center_crop(img,size): #center crops and downsamples to lower, square resolution that yolo expects

    h,w=img.shape[:2] #y,x

    min_dim=min(h,w)
    max_dim=max(h,w)
    square_dim_half=(max_dim-min_dim)//2

    if min_dim==h: #landscape
        x_min=square_dim_half
        x_max=w-square_dim_half
        return cv2.resize(img[:,x_min:x_max],size,interpolation=cv2.INTER_AREA)
    else:
        y_min=square_dim_half
        y_max=h-square_dim_half
        return cv2.resize(img[y_min:y_max,:],size,interpolation=cv2.INTER_AREA)
def euclidean_distance(x1,x2,y1,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def interpolater(segments: list, arr: np.array) -> np.array:
    for start_i, start_v, end_v, end_i in segments:
        values_to_add = np.linspace(start_v, end_v, end_i-start_i+1)
        arr[start_i-1:end_i] = values_to_add
    return arr
if __name__ == '__main__':
    pass

    