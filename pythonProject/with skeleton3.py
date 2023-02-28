import json
import numpy as np
import cv2
import math
import statistics

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
    # print(img)
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
    eucli_dis = []
    slope_angles = []
    # slope_angles_for_classification= []

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
    # Line Segment Detection Method

    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines in the image
    lines = lsd.detect(cropped_image[4])[0]  # Position 0 of the returned tuple are the detected lines
    # print(lines)

    copy_cropped_img = cropped_image[4] * 0

    # Draw detected lines in the image
    drawn_img = lsd.drawSegments(copy_cropped_img, lines)
    drawn_img_gray = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)

    binary_image_line = cv2.adaptiveThreshold(drawn_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                              cv2.THRESH_BINARY, 15, -2)

    cv2.imshow('binary_image_line', binary_image_line)

    # # unit by unit line segmenting
    # for jj in range(len(list_of_keys)):
    #     # print(list_of_keys)
    #     class_name_from_list_units = list_of_keys[jj]
    #     selected_list_units = dic_for_selection[class_name_from_list_units]

    # unit by unit line segmenting
    class_name_from_list_units = list_of_keys[11] # changing type of branch or plant unit
    selected_list_units = dic_for_selection[class_name_from_list_units]

    selected_list_units_for_cv = selected_list_units.astype(np.uint8) * 255
    lsd_unit = cv2.createLineSegmentDetector(0)
    lines_unit = lsd_unit.detect(selected_list_units_for_cv)[0]

    # lines_unit_test = lsd_unit.detect(selected_list_units_for_cv)
    # print(lines_unit)

    copy_selected_list_units = selected_list_units_for_cv * 0

    drawn_img_unit = lsd_unit.drawSegments(copy_selected_list_units, lines_unit)
    drawn_img_unit_gray = cv2.cvtColor(drawn_img_unit, cv2.COLOR_BGR2GRAY)

    binary_image_line_unit = cv2.adaptiveThreshold(drawn_img_unit_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                   cv2.THRESH_BINARY, 15, -2)

    binary_image_line_unit_RGB = cv2.cvtColor(binary_image_line_unit, cv2.COLOR_GRAY2BGR)

    for length in lines_unit:
        x0 = length[0][0]
        y0 = length[0][1]
        x1 = length[0][2]
        y1 = length[0][3]

        point1 = np.array((x0, y0))
        point2 = np.array((x1, y1))

        # calculate Euclidean distance for min length line
        dist = np.linalg.norm(point1 - point2)
        eucli_dis.append(dist)

        angle = (y1 - y0) / (x1 - x0)
        angle_degrees = int(math.degrees(math.atan(angle)))


        slope_angles.append(angle_degrees)

    slope_angles_for_classification = slope_angles.copy()

    count_0 = 0
    count_25 = 0
    count_45 = 0
    count_65 = 0
    count_90 = 0
    count_135 = 0
    count_180 = 0
    count_m0 = 0
    count_m25 = 0
    count_m45 = 0
    count_m65 = 0
    count_m90 = 0
    count_m135 = 0
    count_m180 = 0

    for jj in range(len(slope_angles)):

        if 0 < slope_angles[jj] < 25:
            slope_angles_for_classification[jj] = 0
            count_0 = count_0 + 1
        elif 25 < slope_angles[jj] < 45:
            slope_angles_for_classification[jj] = 45
            count_45 = count_45 + 1
        elif 45 < slope_angles[jj] < 65:
            slope_angles_for_classification[jj] = 65
            count_65 = count_65 + 1
        elif 65 < slope_angles[jj] < 90:
            slope_angles_for_classification[jj] = 90
            count_90 = count_90 + 1
        elif 90 < slope_angles[jj] < 135:
            slope_angles_for_classification[jj] = 135
            count_135 = count_135 + 1
        elif 135 < slope_angles[jj] < 180:
            slope_angles_for_classification[jj] = 180
            count_180 = count_180 + 1

        elif -25 < slope_angles[jj] < 0:
            slope_angles_for_classification[jj] = 0
            count_m0 = count_m0 + 1
        elif -45 < slope_angles[jj] < -25:
            slope_angles_for_classification[jj] = -45
            count_m45 = count_m45 + 1
        elif -65 < slope_angles[jj] < -45:
            slope_angles_for_classification[jj] = -65
            count_m65 = count_m65 + 1
        elif -90 < slope_angles[jj] < -65:
            slope_angles_for_classification[jj] = -90
            count_m90 = count_m90 + 1
        elif -135 < slope_angles[jj] < -90:
            slope_angles_for_classification[jj] = -135
            count_m135 = count_m135 + 1
        else:
            slope_angles_for_classification[jj] = -180
            count_m180 = count_m180 + 1



        # elif -180 < slope_angles[jj] < -135:
        #     slope_angles_for_classification[jj] = -180
        #     count_m180 = count_m180 + 1



    max_usage_angle = [count_0, count_45, count_65, count_90, count_135, count_180, count_m0, count_m45, count_m65, count_m90, count_m135, count_m180]
    east_count = count_m0 + count_m45 + count_m65 + count_m90 + count_m135 + count_m180
    west_count = count_0 + count_45 + count_65 + count_90 + count_135 + count_180

    if east_count > west_count:
        direction = "East"

    else:
        direction = "West"


    print(slope_angles)
    print(slope_angles_for_classification)
    # print("mode:", statistics.mean(slope_angles))
    print("max usage index:", max_usage_angle.index(max(max_usage_angle)))
    print("East count:", east_count, "  West count:", west_count)
    print("Direction:", direction)

    min_line_index = eucli_dis.index(min(eucli_dis))
    # print(min_line_index)

    min_line_coordinates = lines_unit[18][0]  # put min line index variable here
    x0_min = int(min_line_coordinates[0])
    y0_min = int(min_line_coordinates[1])
    x1_min = int(min_line_coordinates[2])
    y1_min = int(min_line_coordinates[3])

    cv2.line(binary_image_line_unit_RGB, (x0_min, y0_min), (x1_min, y1_min), (0, 255, 0), 1, cv2.LINE_AA)

    # Using structuring element
    # horizontal = np.copy(bw)
    # vertical = np.copy(bw)
    # # [init]
    # # [horiz]
    # # Specify size on horizontal axis
    # cols = horizontal.shape[1]
    # horizontal_size = cols // 30
    # # Create structure element for extracting horizontal lines through morphology operations
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # # Apply morphology operations
    # horizontal = cv2.erode(horizontal, horizontalStructure)
    # horizontal = cv2.dilate(horizontal, horizontalStructure)
    #
    # rows = vertical.shape[0]
    # verticalsize = rows // 30
    # # Create structure element for extracting vertical lines through morphology operations
    # verticalStructure = cv2.getStructuringElement(cv2.MORPH_CROSS, (horizontal_size, verticalsize))
    # # Apply morphology operations
    # vertical = cv2.erode(vertical, verticalStructure)
    # vertical = cv2.dilate(vertical, verticalStructure)
    #
    # cv2.imshow('horizontal', horizontal)
    # cv2.imshow('vertical', vertical)

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
    resized_drawn_img = cv2.resize(drawn_img_gray, dim, interpolation=cv2.INTER_AREA)
    resized_binary_image_line_unit = cv2.resize(binary_image_line_unit, dim, interpolation=cv2.INTER_AREA)
    resized_binary_image_line_unit_RGB = cv2.resize(binary_image_line_unit_RGB, dim, interpolation=cv2.INTER_AREA)

    # resized_horizontal = cv2.resize(horizontal, dim, interpolation=cv2.INTER_AREA)
    # resized_segmented_img_aft = cv2.resize(segmented_img_aft, dim, interpolation=cv2.INTER_AREA)

    # converting to 3 channel BGR image for display (https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html)
    # zeros_for_merge = np.zeros(resized_image_ori.shape[:2], dtype="uint8")
    # resized_image_ori_rgb = cv2.merge([resized_image_ori, resized_image_ori, resized_image_ori]) # BGR Channels
    # resized_image_ori_rgb = cv2.cvtColor(resized_image_ori, cv2.COLOR_GRAY2RGB)
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)

    # resized_branch_pts_mask_rgb = cv2.merge([zeros_for_merge, resized_branch_pts_mask, zeros_for_merge])
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)

    # cv2.imwrite('C:/Users/Hasith Dasanayake/Desktop/image_shehan_sir.jpg', resized_image)

    cv2.imshow('original', resized_image_ori)
    # cv2.imshow('resized_image_ori_BB', resized_image_ori_BB)
    # cv2.imshow('resized_binary_image_line_unit', resized_binary_image_line_unit)
    cv2.imshow('resized_binary_image_line_unit_RGB', resized_binary_image_line_unit_RGB)

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
