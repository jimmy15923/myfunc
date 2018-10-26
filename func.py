#My functions for ML/DL/CV

def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.3):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow
    alpha: float [0, 1]

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    out = cv2.addWeighted(img_layer, alpha, out, 1-alpha, 0, out)
    return out
  
  
def IOU(y_true, y_pred, is_onehot=False, threshold=0.5):
    '''
    y_true: Groud truth array, shape=[x_size, y_size, channels]
    y_pred: predicted array, shape=[x_size, y_size, class]
    is_onehot: Boolean, is label been onehot encoding
    threshold: for binarizing prediction
    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''  
    if is_onehot:
        y_true = y_true.argmax(axis=-1)
        y_pred = y_pred.argmax(axis=-1)
    else:
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred != 1] = 0
    
    intersection = np.sum(y_true * y_pred)
    union = np.sum((y_true+y_pred) - (y_true*y_pred))
    
    iou = intersection / union
    return iou


def contrast_stretch_3d(data_arr):
    '''Stretch the contract on 3D image
    data_arr: 3D numpy array, shape should be (N, H, W, C)
    
    returns:
    3D np array with contrast stretched and scaled to [0, 1]
    '''
    from skimage import exposure
    arr = np.zeros(data_arr.shape)
    for n in range(len(data_arr)):
        for i in range(len(data_arr[0,0,0])):
            img = (data_arr[n,:,:,i]).astype("uint8")
            p2, p98 = np.percentile(img, (2, 98))
            img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
            arr[n,:,:,i] = img_rescale / (255.0)
    return arr
