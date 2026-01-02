import cv2
def center_crop(img,size):

    h,w=img.shape[:2]

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



if __name__ == '__main__':
    pass

    