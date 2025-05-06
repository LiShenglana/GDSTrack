import cv2
import numpy as np
from copy import deepcopy

def crop_image_visualize(template_img_color, template_img_ir, _t_box, search1_img_color, search1_img_ir, _s1_box, search2_img_color, search2_img_ir, _s2_box, search3_img_color, search3_img_ir, _s3_box):
    t_box = deepcopy(_t_box)
    s1_box = deepcopy(_s1_box)
    s2_box = deepcopy(_s2_box)
    s3_box = deepcopy(_s3_box)
    alpha = 1
    beta = 0.2
    input_size_temp = template_img_color.shape
    h1, w1 = input_size_temp[0], input_size_temp[1]
    input_size_search = search1_img_color.shape
    h2, w2 = input_size_search[0], input_size_search[1]
    template_img_color = cv2.cvtColor(template_img_color, cv2.COLOR_BGR2RGB)
    template_img_ir = cv2.cvtColor(template_img_ir, cv2.COLOR_BGR2RGB)
    search1_img_color = cv2.cvtColor(search1_img_color, cv2.COLOR_BGR2RGB)
    search1_img_ir = cv2.cvtColor(search1_img_ir, cv2.COLOR_BGR2RGB)
    search2_img_color = cv2.cvtColor(search2_img_color, cv2.COLOR_BGR2RGB)
    search2_img_ir = cv2.cvtColor(search2_img_ir, cv2.COLOR_BGR2RGB)
    search3_img_color = cv2.cvtColor(search3_img_color, cv2.COLOR_BGR2RGB)
    search3_img_ir = cv2.cvtColor(search3_img_ir, cv2.COLOR_BGR2RGB)
    # mask_temp = np.zeros([128, 128], dtype=np.uint8)

    t_box[2:4] = t_box[0:2] + t_box[2:4]
    s1_box[2:4] = s1_box[0:2] + s1_box[2:4]
    s2_box[2:4] = s2_box[0:2] + s2_box[2:4]
    s3_box[2:4] = s3_box[0:2] + s3_box[2:4]

    mask_temp = np.full([h1, w1], 0, dtype=np.uint8)
    mask_temp[t_box[1]:t_box[3]+1, t_box[0]:t_box[2]+1] = 1
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask1_1 = mask_temp * template_img_color
    mask1_2 = mask_temp * template_img_ir
    #
    mask_temp = np.full([h1, w1], 1, dtype=np.uint8)
    mask_temp[t_box[1]:t_box[3] + 1, t_box[0]:t_box[2] + 1] = 0
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask2_1 = mask_temp * template_img_color
    mask2_2 = mask_temp * template_img_ir

    template_img_color = cv2.addWeighted(mask1_1, alpha, mask2_1, beta, 1)
    template_img_ir = cv2.addWeighted(mask1_2, alpha, mask2_2, beta, 1)

    mask_temp = np.full([h2, w2], 0, dtype=np.uint8)
    mask_temp[s1_box[1]:s1_box[3]+1, s1_box[0]:s1_box[2]+1] = 1
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask1_1 = mask_temp * search1_img_color
    mask1_2 = mask_temp * search1_img_ir
    #
    mask_temp = np.full([h2, w2], 1, dtype=np.uint8)
    mask_temp[s1_box[1]:s1_box[3] + 1, s1_box[0]:s1_box[2] + 1] = 0
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask2_1 = mask_temp * search1_img_color
    mask2_2 = mask_temp * search1_img_ir

    search1_img_color = cv2.addWeighted(mask1_1, alpha, mask2_1, beta, 1)
    search1_img_ir = cv2.addWeighted(mask1_2, alpha, mask2_2, beta, 1)

    mask_temp = np.full([h2, w2], 0, dtype=np.uint8)
    mask_temp[s2_box[1]:s2_box[3]+1, s2_box[0]:s2_box[2]+1] = 1
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask1_1 = mask_temp * search2_img_color
    mask1_2 = mask_temp * search2_img_ir
    #
    mask_temp = np.full([h2, w2], 1, dtype=np.uint8)
    mask_temp[s2_box[1]:s2_box[3] + 1, s2_box[0]:s2_box[2] + 1] = 0
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask2_1 = mask_temp * search2_img_color
    mask2_2 = mask_temp * search2_img_ir

    search2_img_color = cv2.addWeighted(mask1_1, alpha, mask2_1, beta, 1)
    search2_img_ir = cv2.addWeighted(mask1_2, alpha, mask2_2, beta, 1)

    mask_temp = np.full([h2, w2], 0, dtype=np.uint8)
    mask_temp[s3_box[1]:s3_box[3]+1, s3_box[0]:s3_box[2]+1] = 1
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask1_1 = mask_temp * search3_img_color
    mask1_2 = mask_temp * search3_img_ir
    #
    mask_temp = np.full([h2, w2], 1, dtype=np.uint8)
    mask_temp[s3_box[1]:s3_box[3] + 1, s3_box[0]:s3_box[2] + 1] = 0
    mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)
    mask2_1 = mask_temp * search3_img_color
    mask2_2 = mask_temp * search3_img_ir

    search3_img_color = cv2.addWeighted(mask1_1, alpha, mask2_1, beta, 1)
    search3_img_ir = cv2.addWeighted(mask1_2, alpha, mask2_2, beta, 1)




    cv2.imshow('template_img_color', template_img_color)
    cv2.imshow('search1_img_color', search1_img_color)
    cv2.imshow('search2_img_color', search2_img_color)
    cv2.imshow('search3_img_color', search3_img_color)
    cv2.imshow('template_img_ir', template_img_ir)
    cv2.imshow('search1_img_ir', search1_img_ir)
    cv2.imshow('search2_img_ir', search2_img_ir)
    cv2.imshow('search3_img_ir', search3_img_ir)

    cv2.waitKey(0)