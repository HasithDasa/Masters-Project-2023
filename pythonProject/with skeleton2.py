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


with open('training_review.json') as f:
    data = json.load(f)
    data1 = data['images'][10]  # selecting the image
    img_name = data1['image_name']
    img_width = data1['width']
    img_height = data1['height']
    img_temp_shape = (img_height, img_width)
    img_temp = np.zeros(img_temp_shape, dtype=np.uint8)
    data2 = data1['labels']  # selecting the label key
    print(img_name)

    list_of_keys = []
    dic_for_selection = {}
    cropped_image = []
    for i in range(len(data2)):
        data3 = data2[i]  # selecting the class label
        mask_rle = data3['mask']
        bbox = data3['bbox']
        class_name = data3['class_name']
        # print(class_name)

        if class_name == 'First Section Cutting' or class_name == 'Redundant Top End' or class_name == 'Redundant Bottom End' or class_name == 'Tip Cutting' or class_name == 'Non-Viable Part' or class_name == 'Second Section Cutting' or class_name == 'Third Section Cutting' or class_name == 'Fourth Section Cutting':

            # making mask
            mask = rle_decode(mask_rle, (bbox[2] - bbox[0], bbox[3] - bbox[1]))

            # where exactly the mask situated in main image
            row_start = bbox[1]
            row_end = bbox[3]

            col_start = bbox[0]
            col_end = bbox[2]

            # including mask in the main image, need to make it zero for everytime because of mask coincidence problem
            img_temp = np.zeros(img_temp_shape, dtype=np.uint8)
            img_temp[row_start:row_end, col_start:col_end] = mask

            variable = str(i)
            key = "lable" + variable
            list_of_keys.append(key)
            dic_for_selection[key] = img_temp

    for j in range(len(list_of_keys)):
        # print(list_of_keys)
        class_name_from_list = list_of_keys[j]
        selected_list = dic_for_selection[class_name_from_list]

        result_with_one = np.where(selected_list == 1)
        listOfCoordinates = list(zip(result_with_one[0], result_with_one[1]))
        # print(listOfCoordinates)
        # iterate over the list of coordinates
        for cord in listOfCoordinates:
            img_temp[cord] = 1

    # plt.imshow(img_temp, 'gray')
    # plt.show()
    # plt.imsave('test_img.png', img_temp)

# converting to 8bit image format that opencv can read
#     img_for_cv = img_temp*255
    img_for_cv = img_temp.astype(np.uint8) * 255
    img_for_BB = img_temp.astype(np.uint8) * 255

    # thresh_image_bbox = cv2.threshold(img_for_cv, 0, 254, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for i_2 in contours:
        x_BB, y_BB, w_BB, h_BB = cv2.boundingRect(i_2)
        x_BB_left = x_BB - 10
        y_BB_left = y_BB - 10
        x_BB_right = x_BB + w_BB + 10
        y_BB_right = y_BB + h_BB + 10

        cv2.rectangle(img_for_BB, (x_BB_left, y_BB_left), (x_BB_right, y_BB_right), (255, 0, 0), 2)
        cropped_image.append(img_for_cv[y_BB_left:y_BB_right, x_BB_left:x_BB_right])


# skeliterization (https://plantcv.readthedocs.io/en/v3.4.1/morphology_tutorial/#morphology-script)

    pcv.params.line_thickness = 3
    skeleton = pcv.morphology.skeletonize(mask=img_for_cv)
    skeleton_BB = pcv.morphology.skeletonize(mask=cropped_image[1])
    pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=10)

# Hough Transform
#
#     edges_canny = cv2.Canny(cropped_image[5], 50, 150, apertureSize=3)
#     # lines = cv2.HoughLines(edges_canny, 1, np.pi / 180, 60)
#     lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=2)
#
#
#
#
#
    # for line in lines:
    #     # rho, theta = line[0]
    #     # a = np.cos(theta)
    #     # b = np.sin(theta)
    #     # x0 = a * rho
    #     # y0 = b * rho
    #     # # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    #     # x1 = int(x0 + 1000 * (-b))
    #     # # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
    #     # y1 = int(y0 + 1000 * (a))
    #     # # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
    #     # x2 = int(x0 - 1000 * (-b))
    #     # # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
    #     # y2 = int(y0 - 1000 * (a))
    #     edges_canny_rgb = cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(edges_canny_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # cv2.imshow('edges_canny_rgb', edges_canny_rgb)

    # Create default parametrization LSD
#Line Segment Detection Method

    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines in the image
    lines = lsd.detect(cropped_image[4])[0]  # Position 0 of the returned tuple are the detected lines
    print(lines)

    copy_cropped_img = cropped_image[4] * 0

    # Draw detected lines in the image
    drawn_img = lsd.drawSegments(copy_cropped_img, lines)
    drawn_img_gray = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(drawn_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                              cv2.THRESH_BINARY, 15, -2)

    cv2.imshow('bw', bw)


    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [init]
    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_CROSS, (horizontal_size, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    cv2.imshow('horizontal', horizontal)
    cv2.imshow('vertical', vertical)













# #hough 2nd try
#     edges_canny = cv2.Canny(drawn_img_gray, 50, 150, apertureSize=5)
#     lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, 50, minLineLength=1, maxLineGap=50)
#
#     for line in lines:
#         edges_canny_rgb = cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)
#         x1, y1, x2, y2 = line[0]
#         cv2.line(edges_canny_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)
#     cv2.imshow('edges_canny_rgb', edges_canny_rgb)




# finding branch points
    # branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skeleton, mask=img_for_cv)

# finding objects (https://github.com/danforthcenter/plantcv/blob/d1456d0fabb2634cdfa5d79b681fde03ba54f22a/plantcv/plantcv/find_objects.py#L10)

    def find_objects(img1):
        """
        Find all objects and color them blue.
        Inputs:
        img       = RGB or grayscale image data for plotting
        mask      = Binary mask used for contour detection
        Returns:
        objects   = list of contours
        hierarchy = contour hierarchy list
        :param img: numpy.ndarray
        :param mask: numpy.ndarray
        :return objects: list
        :return hierarchy: numpy.ndarray
        """
        # mask1 = np.copy(mask)
        ori_img = img1
        # If the reference image is grayscale convert it to color
        if len(np.shape(img1)) == 2:
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
        objects, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        # Cast tuple objects as a list
        objects = list(objects)
        for ii, cnt in enumerate(objects):
            cv2.drawContours(ori_img, objects, ii, (255, 102, 255), -1, lineType=8, hierarchy=hierarchy)

        return objects, hierarchy

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

        dilated_skel = dilate(img2, 5, 1) # linetickness 5
        tip_plot = cv2.cvtColor(dilated_skel, cv2.COLOR_GRAY2RGB)

        # Initialize list of tip data points
        tip_list = []
        tip_labels = []
        for i, tip in enumerate(tip_objects):
            x, y = tip.ravel()[:2]
            coord = (int(x), int(y))
            tip_list.append(coord)
            tip_labels.append(i)
            cv2.circle(tip_plot, (x, y), 5, (0, 255, 0), -1)

        return tip_plot


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
            coord = (int(x), int(y))
            branch_list.append(coord)
            branch_labels.append(i)
            cv2.circle(branch_plot, (x, y), 5, (255, 0, 0), -1)  # line tickness 5

            # coord_curve = [int(x), int(y)]
            # branch_list_curve.append(coord_curve)

        # print(branch_list_curve)
        # pts = np.array(branch_list_curve, np.int32)
        # cv2.polylines(branch_plot, [pts], True, (255, 0, 0), 50)

        return branch_plot


    tip_img_mask = find_tips(pruned_skeleton)
    branch_pts_mask_aft = find_branch_pts(pruned_skeleton)

    leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=pruned_skeleton, objects=segment_objects)
    leaf_segmented_img, leaf_labeled_img = pcv.morphology.segment_id(skel_img=pruned_skeleton, objects=stem_obj)


# Scaling down Parameters
    scale_percent = 50  # percent of original size
    width = int(img_width * scale_percent / 100)
    height = int(img_height * scale_percent / 100)
    width_BB = int(w_BB * scale_percent / 100)
    height_BB = int(h_BB * scale_percent / 100)
    dim = (width, height)
    dim_BB = (width_BB, height_BB)

# Resized images for display
    resized_image_ori = cv2.resize(img_for_cv, dim, interpolation=cv2.INTER_AREA)
    resized_image_ori_BB = cv2.resize(img_for_BB, dim, interpolation=cv2.INTER_AREA)
    resized_image_cropped = skeleton_BB
    resized_image_skel_pr = cv2.resize(segmented_img, dim, interpolation=cv2.INTER_AREA)
    resized_branch_pts_mask = cv2.resize(branch_pts_mask_aft, dim, interpolation=cv2.INTER_AREA)
    resized_leaf_labeled_img = cv2.resize(leaf_labeled_img, dim, interpolation=cv2.INTER_AREA)
    resized_tip_plot = cv2.resize(tip_img_mask, dim, interpolation=cv2.INTER_AREA)
    resized_drawn_img = cv2.resize(drawn_img_gray, dim, interpolation=cv2.INTER_AREA)
    resized_horizontal = cv2.resize(horizontal, dim, interpolation=cv2.INTER_AREA)
    # resized_segmented_img_aft = cv2.resize(segmented_img_aft, dim, interpolation=cv2.INTER_AREA)


#converting to 3 channel BGR image for display (https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html)
    # zeros_for_merge = np.zeros(resized_image_ori.shape[:2], dtype="uint8")
    # resized_image_ori_rgb = cv2.merge([resized_image_ori, resized_image_ori, resized_image_ori]) # BGR Channels
    resized_image_ori_rgb = cv2.cvtColor(resized_image_ori, cv2.COLOR_GRAY2RGB)
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)

    mixed_img_ori_skel_leaf = cv2.addWeighted(resized_image_ori_rgb, 0.5, resized_leaf_labeled_img, 0.5, 0.0)

    # resized_branch_pts_mask_rgb = cv2.merge([zeros_for_merge, resized_branch_pts_mask, zeros_for_merge])
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)
    mixed_img_ori_skel_branch_pts = cv2.addWeighted(resized_image_ori_rgb, 0.5, resized_branch_pts_mask, 0.5, 0.0)
    mixed_img_ori_skel_tip_branch_pts = cv2.addWeighted(mixed_img_ori_skel_branch_pts, 0.5, resized_tip_plot, 0.5, 0.0)

    # cv2.imwrite('C:/Users/Hasith Dasanayake/Desktop/image_shehan_sir.jpg', resized_image)

    cv2.imshow('original', resized_image_ori)
    cv2.imshow('resized_image_ori_BB', resized_image_ori_BB)
    cv2.imshow('skeleton', resized_image_cropped)
    # cv2.imshow('resized_drawn_img', resized_drawn_img)
    # cv2.imshow('resized_horizontal', resized_horizontal)
    # cv2.imshow('Pruned_skeleton', resized_image_skel_pr)
    # cv2.imshow('branch_pts_mask', resized_branch_pts_mask)
    # cv2.imshow('leaf_labeled_img', resized_leaf_labeled_img)
    # cv2.imshow('mixed_img_ori_skel', mixed_img_ori_skel_leaf)
    # cv2.imshow('mixed_img_ori_skel_branch_pts', mixed_img_ori_skel_branch_pts)
    # cv2.imshow('resized_tip_plot', resized_tip_plot)
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

