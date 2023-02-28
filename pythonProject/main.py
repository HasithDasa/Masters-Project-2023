import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import string
from plantcv import plantcv as pcv
from plantcv.plantcv import dilate
from plantcv.plantcv import image_subtract
from plantcv.plantcv import color_palette
import sys, traceback

def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (width, height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    shape = shape[1], shape[0]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    #print(img)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)





def find_tips(img2):
        """Find tips in skeletonized image.
        The endpoints algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
        Inputs:
        skel_img    = Skeletonized image
        mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
        label        = optional label parameter, modifies the variable name of observations recorded
        Returns:
        tip_img   = Image with just tips, rest 0
        :param skel_img: numpy.ndarray
        :param mask: numpy.ndarray
        :param label: str
        :return tip_img: numpy.ndarray
        """
        # In a kernel: 1 values line up with 255s, -1s line up with 0s, and 0s correspond to dont care
        endpoint1 = np.array([[-1, -1, -1],
                              [-1, 1, -1],
                              [0, 1, 0]])
        endpoint2 = np.array([[-1, -1, -1],
                              [-1, 1, 0],
                              [-1, 0, 1]])

        endpoint3 = np.rot90(endpoint1)
        endpoint4 = np.rot90(endpoint2)
        endpoint5 = np.rot90(endpoint3)
        endpoint6 = np.rot90(endpoint4)
        endpoint7 = np.rot90(endpoint5)
        endpoint8 = np.rot90(endpoint6)

        endpoints = [endpoint1, endpoint2, endpoint3, endpoint4, endpoint5, endpoint6, endpoint7, endpoint8]
        tip_img = np.zeros(img2.shape[:2], dtype=int)
        for endpoint in endpoints:
            tip_img = np.logical_or(cv2.morphologyEx(img2, op=cv2.MORPH_HITMISS, kernel=endpoint,
                                                     borderType=cv2.BORDER_CONSTANT, borderValue=0), tip_img)
        tip_img = tip_img.astype(np.uint8) * 255

        tip_objects, _ = find_objects(tip_img)
        print(tip_objects)

        dilated_skel = dilate(img2, 5, 1) # linetickness 5
        tip_plot = cv2.cvtColor(dilated_skel, cv2.COLOR_GRAY2RGB)

        # Initialize list of tip data points
        tip_list = []
        tip_labels = []
        for i4, tip in enumerate(tip_objects):
            x, y = tip.ravel()[:2]
            coord = (int(x), int(y))
            # print(coord)
            tip_list.append(coord)
            tip_labels.append(i4)
            # cv2.circle(tip_plot, (x, y), 5, (0, 255, 0), -1)
        # print(tip_objects)
        # print(tip_list)

        return tip_list


# finding branch points (https://github.com/danforthcenter/plantcv/blob/master/plantcv/plantcv/morphology/find_branch_pts.py)

def find_branch_pts(skel_img):
        """Find branch points in a skeletonized image.
        The branching algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699

        Inputs:
        skel_img    = Skeletonized image
        mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
        label        = optional label parameter, modifies the variable name of observations recorded

        Returns:
        branch_pts_img = Image with just branch points, rest 0

        :param skel_img: numpy.ndarray
        :param mask: np.ndarray
        :param label: str
        :return branch_pts_img: numpy.ndarray
        """
        # In a kernel: 1 values line up with 255s, -1s line up with 0s, and 0s correspond to don't care
        # T like branch points
        t1 = np.array([[-1, 1, -1],
                       [1, 1, 1],
                       [-1, -1, -1]])
        t2 = np.array([[1, -1, 1],
                       [-1, 1, -1],
                       [1, -1, -1]])
        t3 = np.rot90(t1)
        t4 = np.rot90(t2)
        t5 = np.rot90(t3)
        t6 = np.rot90(t4)
        t7 = np.rot90(t5)
        t8 = np.rot90(t6)

        # Y like branch points
        y1 = np.array([[1, -1, 1],
                       [0, 1, 0],
                       [0, 1, 0]])
        y2 = np.array([[-1, 1, -1],
                       [1, 1, 0],
                       [-1, 0, 1]])
        y3 = np.rot90(y1)
        y4 = np.rot90(y2)
        y5 = np.rot90(y3)
        y6 = np.rot90(y4)
        y7 = np.rot90(y5)
        y8 = np.rot90(y6)
        kernels = [t1, t2, t3, t4, t5, t6, t7, t8, y1, y2, y3, y4, y5, y6, y7, y8]

        branch_pts_img = np.zeros(skel_img.shape[:2], dtype=int)

        # Store branch points
        for kernel in kernels:
            branch_pts_img = np.logical_or(cv2.morphologyEx(skel_img, op=cv2.MORPH_HITMISS, kernel=kernel,
                                                            borderType=cv2.BORDER_CONSTANT, borderValue=0),
                                           branch_pts_img)

        # Switch type to uint8 rather than bool
        branch_pts_img = branch_pts_img.astype(np.uint8) * 255

        # Make debugging image

        dilated_skel = dilate(skel_img, 5, 1) # line tickness 5
        branch_plot = cv2.cvtColor(dilated_skel, cv2.COLOR_GRAY2RGB)

        branch_objects, _ = find_objects(branch_pts_img)

        # Initialize list of tip data points
        branch_list = []
        branch_list_curve = []
        branch_labels = []

        for i, branch in enumerate(branch_objects):
            x, y = branch.ravel()[:2]
            coord = (int(y), int(x))
            branch_list.append(coord)
            branch_labels.append(i)
            # cv2.circle(branch_pts_img, (x, y), 100, (0, 0, 255), -1)  # line tickness 5

            coord_curve = [int(x), int(y)]
            branch_list_curve.append(coord_curve)

        # print(branch_list_curve)
        pts_1 = np.array(branch_list_curve, np.int32)
        # cv2.polylines(branch_pts_img, [pts_1], False, (255, 0, 0), 10)

        return branch_list



with open('training_review.json') as f:
    data = json.load(f)
    data1 = data['images'][35]  # selecting the image
    img_name = data1['image_name']
    img_width = data1['width']
    img_height = data1['height']
    img_temp_shape = (img_height, img_width)
    img_temp_origi = np.zeros(img_temp_shape, dtype=np.uint8)
    data2 = data1['labels']  # selecting the label key
    print(img_name)

    list_of_keys = []
    dic_for_selection_ori = {}
    dic_for_selection_tip = {}
    dic_for_selection_branch = {}
    tip_list_curve = []
    branch_list_curve = []

    for i in range(len(data2)):
        data3 = data2[i]  # selecting the class label
        mask_rle = data3['mask']
        bbox = data3['bbox']
        class_name = data3['class_name']
        # print(class_name)

        if class_name == 'First Section Cutting' or class_name == 'Redundant Top End' or class_name == 'Redundant Bottom End' or class_name == 'Tip Cutting' or class_name == 'Non-Viable Part' or class_name == 'Second Section Cutting' or class_name == 'Third Section Cutting' or class_name == 'Fourth Section Cutting':

            # making mask for computer vision, converting to 8bit image format that opencv can read
            mask_for_ori = rle_decode(mask_rle, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
            mask_ori_for_cv = mask_for_ori.astype(np.uint8) * 255

            pcv.params.line_thickness = 5
            skeleton = pcv.morphology.skeletonize(mask=mask_ori_for_cv)
            pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=10)
            dilated_skel = dilate(pruned_skeleton, 5, 1)


            list_tip_cordi = find_tips(mask_ori_for_cv)
            # mask_for_tip_coord = np.zeros_like(mask_ori_for_cv)
            mask_for_tip_coord = np.zeros(mask_ori_for_cv.shape[:2], dtype="uint8")
            # print(mask_ori_for_cv.shape[:2])

            list_branch_cordi = find_branch_pts(dilated_skel)
            mask_for_branch_coord = np.zeros(mask_ori_for_cv.shape[:2], dtype="uint8")

            for cord_tip in list_tip_cordi:
                mask_for_tip_coord[cord_tip] = 255
                cv2.circle(mask_for_tip_coord, cord_tip, 3, (255, 255, 255), -1)
                tip_list_curve.append(cord_tip)

            pts_tip = np.array(tip_list_curve, np.int32)
            # cv2.polylines(mask_for_tip_coord, [pts_tip], False, (255, 255, 255), 1)

            for cord_bran in list_branch_cordi:
                mask_for_branch_coord[cord_bran] = 255
                cv2.circle(mask_for_branch_coord, cord_bran, 3, (255, 255, 255), -1)
                branch_list_curve.append(cord_bran)

            pts_bran = np.array(branch_list_curve, np.int32)
            # cv2.polylines(mask_for_branch_coord, [pts_bran], False, (255, 255, 255), 1)

            # where exactly the mask situated in main image
            row_start = bbox[1]
            row_end = bbox[3]

            col_start = bbox[0]
            col_end = bbox[2]

            # including mask in the main image, need to make it zero for everytime because of mask coincidence problem
            img_temp_origi = np.zeros(img_temp_shape, dtype=np.uint8)
            img_temp_origi[row_start:row_end, col_start:col_end] = mask_ori_for_cv

            img_temp_tip = np.zeros(img_temp_shape, dtype=np.uint8)
            img_temp_tip[row_start:row_end, col_start:col_end] = mask_for_tip_coord

            img_temp_branch = np.zeros(img_temp_shape, dtype=np.uint8)
            img_temp_branch[row_start:row_end, col_start:col_end] = mask_for_branch_coord

            variable = str(i)
            key = "lable" + variable
            list_of_keys.append(key)
            dic_for_selection_ori[key] = img_temp_origi
            dic_for_selection_tip[key] = img_temp_tip
            dic_for_selection_branch[key] = img_temp_branch

    for j in range(len(list_of_keys)):
        class_name_from_list = list_of_keys[j]

        selected_list_ori = dic_for_selection_ori[class_name_from_list]
        result_with_one_ori = np.where(selected_list_ori == 255)
        list_of_coordinates_ori = list(zip(result_with_one_ori[0], result_with_one_ori[1]))
        # iterate over the list of coordinates
        for cord_ori in list_of_coordinates_ori:
            img_temp_origi[cord_ori] = 255

        selected_list_tip = dic_for_selection_tip[class_name_from_list]
        result_with_one_tip = np.where(selected_list_tip == 255)
        list_of_coordinates_tip = list(zip(result_with_one_tip[0], result_with_one_tip[1]))
        for coord_tipp in list_of_coordinates_tip:
            img_temp_tip[coord_tipp] = 255

        selected_list_branch = dic_for_selection_branch[class_name_from_list]
        result_with_one_branch = np.where(selected_list_branch == 255)
        list_of_coordinates_branch = list(zip(result_with_one_branch[0], result_with_one_branch[1]))
        for coord_branchh in list_of_coordinates_branch:
            img_temp_branch[coord_branchh] = 255


    # plt.imshow(img_temp, 'gray')
    # plt.show()
    # plt.imsave('test_img.png', img_temp)

    # skeliterization (https://plantcv.readthedocs.io/en/v3.4.1/morphology_tutorial/#morphology-script)

    img_for_cv_ori = img_temp_origi
    img_for_cv_tip = img_temp_tip
    img_for_cv_branch = img_temp_branch

    # pcv.params.line_thickness = 5
    # skeleton_1 = pcv.morphology.skeletonize(mask=img_for_cv_ori)
    # pruned_skeleton_1, segmented_img_1, segment_objects_1 = pcv.morphology.prune(skel_img=skeleton_1, size=10)

    # tip_img_mask = find_tips(pruned_skeleton)
    # branch_pts_mask_aft = find_branch_pts(pruned_skeleton_1)

    # img_for_cv = img_temp*255

    # leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=pruned_skeleton_1, objects=segment_objects)
    # leaf_segmented_img, leaf_labeled_img = pcv.morphology.segment_id(skel_img=pruned_skeleton_1, objects=leaf_obj)


# Scaling down Parameters
    scale_percent = 50  # percent of original size
    width = int(img_width * scale_percent / 100)
    height = int(img_height * scale_percent / 100)
    dim = (width, height)

# Resized images for display
    resized_image_ori = cv2.resize(img_for_cv_ori, dim, interpolation=cv2.INTER_AREA)
    resized_branch_pts_mask = cv2.resize(img_for_cv_branch, dim, interpolation=cv2.INTER_AREA)
    resized_tip_plot = cv2.resize(img_for_cv_tip, dim, interpolation=cv2.INTER_AREA)
    # resized_image_skel = cv2.resize(skeleton, dim, interpolation=cv2.INTER_AREA)
    # resized_image_skel_pr = cv2.resize(segmented_img, dim, interpolation=cv2.INTER_AREA)
    # resized_leaf_labeled_img = cv2.resize(leaf_labeled_img, dim, interpolation=cv2.INTER_AREA)
    # resized_segmented_img_aft = cv2.resize(segmented_img_aft, dim, interpolation=cv2.INTER_AREA)


#converting to 3 channel BGR image for display (https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html)
    # zeros_for_merge = np.zeros(resized_image_ori.shape[:2], dtype="uint8")
    # resized_image_ori_rgb = cv2.merge([resized_image_ori, resized_image_ori, resized_image_ori]) # BGR Channels

    # mixed_img_ori_skel_leaf = cv2.addWeighted(resized_image_ori_rgb, 0.5, resized_leaf_labeled_img, 0.5, 0.0)

    # resized_branch_pts_mask_rgb = cv2.merge([zeros_for_merge, resized_branch_pts_mask, zeros_for_merge])
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_tip_plot, cv2.COLOR_GRAY2RGB)
    # resized_image_ori_rgb = cv2.cvtColor(resized_image_ori, cv2.COLOR_GRAY2RGB)
    mixed_img_ori_skel_branch_pts = cv2.addWeighted(resized_image_ori, 0.5, resized_tip_plot, 0.5, 0.0)
    # mixed_img_ori_skel_tip_branch_pts = cv2.addWeighted(mixed_img_ori_skel_branch_pts, 0.5, resized_tip_plot, 0.5, 0.0)

    # cv2.imwrite('C:/Users/Hasith Dasanayake/Desktop/image_shehan_sir.jpg', resized_image)

    cv2.imshow('original', resized_image_ori)
    # cv2.imshow('skeleton', resized_image_skel)
    # cv2.imshow('Pruned_skeleton', resized_image_skel_pr)
    cv2.imshow('branch_pts_mask', resized_branch_pts_mask)
    # cv2.imshow('leaf_labeled_img', resized_leaf_labeled_img)
    # cv2.imshow('mixed_img_ori_skel', mixed_img_ori_skel_leaf)
    cv2.imshow('mixed_img_ori_skel_branch_pts', mixed_img_ori_skel_branch_pts)
    cv2.imshow('resized_tip_plot', resized_tip_plot)
    # cv2.imshow('mixed_img_ori_skel_tip_branch_pts', mixed_img_ori_skel_tip_branch_pts)

    # pcv.print_results(filename='test_workflow_results.txt')


    cv2.waitKey(0)
    cv2.destroyWindow('i')






    # for i in range(len(data2)):
    #     data3 = data2[i]  # selecting the class label
    #     mask_rle = data3['mask']
    #     bbox = data3['bbox']
    #
    #     # for k in range(len(bbox)):
    #     # for k, item in enumerate(bbox):
    #     if bbox[1] == 0:
    #         print("ok")
    #         print(i)
    #         mask = rle_decode(mask_rle, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
    #         print(type(mask))
    #             # plt.imshow(mask)
    #             # plt.colorbar()
    #             # plt.show()

    # print(np.shape(img_temp))
    # print(np.shape(mask))
    # print(row_start)
    # print(row_end)
    # print(col_start)
    # print(col_end)
    # print(row_end - row_start)
    # print(col_end - col_start)
    # print(img_temp[row_start:row_end, col_start:col_end])




# resized_image = cv2.resize(img_temp, (100, 50))
    #
    # # resize image
    # imagem = cv2.bitwise_not(resized_image)
    # # _, bw_image = cv2.threshold(imagem, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow('GrayImage', imagem)
    # cv2.waitKey()

    # plt.imshow(img_temp)
    # plt.colorbar()
    # plt.show()


    # gen = np.array(img_temp, dtype=np.uint8)
    # cv2.imshow('i', gen)
    # cv2.waitKey(0)
    # cv2.destroyWindow('i')

