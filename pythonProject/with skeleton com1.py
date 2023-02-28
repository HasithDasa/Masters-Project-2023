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
    data1 = data['images'][11]  # selecting the image
    img_name = data1['image_name']
    img_width = data1['width']
    img_height = data1['height']
    img_temp_shape = (img_height, img_width)
    img_temp = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_5 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_4 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_last_unit = np.zeros(img_temp_shape, dtype=np.uint8)
    data2 = data1['labels']  # selecting the label key
    print(img_name)

    list_of_keys = []
    dic_for_selection = {}
    cropped_image = []
    eucli_dis = []
    slope_angles = []
    bbox_list = []
    co_with_ones = []
    bbox_ones_mask_list = []
    mask_list = []
    intersected_comm_ones_list = []
    save_comm_list_1 = []
    save_comm_list_2 = []

    # slope_angles_for_classification= []

    for i in range(len(data2)):
        data3 = data2[i]  # selecting the class label
        mask_rle = data3['mask']
        bbox = data3['bbox']
        class_name = data3['class_name']


        if class_name == 'First Section Cutting' or class_name == 'Redundant Top End' or class_name == 'Redundant Bottom End' or class_name == 'Tip Cutting' or class_name == 'Non-Viable Part' or class_name == 'Second Section Cutting' or class_name == 'Third Section Cutting' or class_name == 'Fourth Section Cutting':
            # making mask
            # print("class name", class_name, i)
            mask = rle_decode(mask_rle, (bbox[2] - bbox[0], bbox[3] - bbox[1]))

            # where exactly the mask situated in main image
            row_start = bbox[1]
            row_end = bbox[3]

            col_start = bbox[0]
            col_end = bbox[2]

            #collecting bbox information
            bbox_list.append([col_start, col_end, row_start, row_end])


            # including mask in the main image, need to make it zero for everytime because of mask coincidence problem
            img_temp = np.zeros(img_temp_shape, dtype=np.uint8)
            img_temp[row_start:row_end, col_start:col_end] = mask


            variable = str(i)
            key = "lable" + variable
            list_of_keys.append(key)
            dic_for_selection[key] = img_temp






# considering as whole image

    for j in range(len(list_of_keys)-1):
        # print(list_of_keys)

        class_name_from_list = list_of_keys[j]
        selected_list = dic_for_selection[class_name_from_list]
        mask_list.append(selected_list)

        result_with_one = np.where(selected_list == 1)
        listOfCoordinates = list(zip(result_with_one[0], result_with_one[1]))

        for cord in listOfCoordinates:
            img_temp[cord] = 1

    # for j_3 in mask_list:
    #
    #     with open("all the masks", "w") as output4:
    #         output4.write(str(j_3))
    #
    #     last_index_bbox = len(bbox_list) - 1
    #     last_variables = bbox_list[last_index_bbox]
    #
    #     row_start_last = last_variables[2]
    #     row_end_last = last_variables[3]
    #
    #     col_start_last = last_variables[0]
    #     col_end_last = last_variables[1]
    #
    #     j_3[row_start_last:row_end_last, col_start_last:col_end_last] = 0
    #
    #     result_with_one_3 = np.where(j_3 == 1)
    #     listOfCoordinates_3 = list(zip(result_with_one_3[0], result_with_one_3[1]))
    #
    #     co_with_ones.append(listOfCoordinates_3)
    #
    #     for cord_3 in listOfCoordinates_3:
    #         img_temp_3[cord_3] = 1

    for j_3 in range(len(list_of_keys)):
        class_name_from_list_3 = list_of_keys[j_3]
        selected_list_3 = dic_for_selection[class_name_from_list_3]

        last_index_bbox = len(bbox_list) - 1
        last_variables = bbox_list[last_index_bbox]

        row_start_last = last_variables[2]
        row_end_last = last_variables[3]

        col_start_last = last_variables[0]
        col_end_last = last_variables[1]

        # remove automatically coming last bounding box element from all the other bounding box elements
        if j_3 <= len(list_of_keys) - 2:
            selected_list_3[row_start_last:row_end_last, col_start_last:col_end_last] = 0 #making last one zero

            result_with_one_3 = np.where(selected_list_3 == 1)
            listOfCoordinates_3 = list(zip(result_with_one_3[0], result_with_one_3[1]))

            co_with_ones.append(listOfCoordinates_3)

            for cord_3 in listOfCoordinates_3:
                img_temp_3[cord_3] = 1

        #special operation on the last element
        else:
            # print("fail")
            img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)
            last_unit = selected_list_3[row_start_last:row_end_last, col_start_last:col_end_last]
            img_temp_last_unit[row_start_last:row_end_last, col_start_last:col_end_last] = last_unit

            result_with_one_4 = np.where(img_temp_last_unit == 1)
            listOfCoordinates_4 = list(zip(result_with_one_4[0], result_with_one_4[1]))

            co_with_ones.append(listOfCoordinates_4)

            for cord_4 in listOfCoordinates_4:
                img_temp_3[cord_4] = 1



    # plt.imshow(img_temp, 'gray')
    # plt.show()
    # plt.imsave('test_img.png', img_temp)

    # converting to 8bit image format that opencv can read

    img_for_cv = img_temp.astype(np.uint8) * 255
    img_for_BB = img_temp.astype(np.uint8) * 255
    img_for_cv_temp = img_temp_3.astype(np.uint8) * 255
    # img_for_cv_temp = img_temp_last_unit.astype(np.uint8) * 255

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
    lines = lsd.detect(cropped_image[1])[0]  # 2nd Position 0 of the returned tuple are the detected lines, then 1st one is the seperate image
    # print(lines)

    copy_cropped_img = cropped_image[0] * 0

    # Draw detected lines in the image
    drawn_img = lsd.drawSegments(copy_cropped_img, lines)
    drawn_img_gray = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)

    binary_image_line = cv2.adaptiveThreshold(drawn_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                              cv2.THRESH_BINARY, 15, -2)

    #cv2.imshow('binary_image_line', binary_image_line)

    # # unit by unit line segmenting
    # for jj in range(len(list_of_keys)):
    #     # print(list_of_keys)
    #     class_name_from_list_units = list_of_keys[jj]
    #     selected_list_units = dic_for_selection[class_name_from_list_units]

# Considering unit by unit line segmenting, contour matching etc

    class_name_from_list_units = list_of_keys[1] # changing type of branch or plant unit
    selected_list_units = dic_for_selection[class_name_from_list_units]

    selected_list_units_for_cv = selected_list_units.astype(np.uint8) * 255
    lsd_unit = cv2.createLineSegmentDetector(0)
    lines_unit = lsd_unit.detect(selected_list_units_for_cv)[0]

    copy_selected_list_units = selected_list_units_for_cv * 0

    drawn_img_unit = lsd_unit.drawSegments(copy_selected_list_units, lines_unit)
    drawn_img_unit_gray = cv2.cvtColor(drawn_img_unit, cv2.COLOR_BGR2GRAY)

    binary_image_line_unit = cv2.adaptiveThreshold(drawn_img_unit_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                   cv2.THRESH_BINARY, 15, -2)

    binary_image_line_unit_RGB = cv2.cvtColor(binary_image_line_unit, cv2.COLOR_GRAY2BGR)

    # for contour operation
    contours_units = cv2.findContours(selected_list_units, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_units = contours_units[0] if len(contours_units) == 2 else contours_units[1]
    # cv2.drawContours(binary_image_line_unit_RGB, contours_units, -1, (0, 0, 255), 3)
    # print("contour units:", contours_units[0])
    # c_X = contours_units[0][0][0][0]
    # c_Y = contours_units[0][0][0][1]
    # cv2.circle(binary_image_line_unit_RGB, (c_X, c_Y), 5, (255, 0, 0), -1)

#finding centre
    # for c in contours_units:
    #     # calculate moments for each contour
    #     M = cv2.moments(c)
    #     # calculate x,y coordinate of center
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     cv2.circle(binary_image_line_unit_RGB, (cX, cY), 5, (255, 0, 0), -1)

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
    # print("bbox list col", bbox_list)



    min_line_index = eucli_dis.index(min(eucli_dis))
    # print(min_line_index)

    min_line_coordinates = lines_unit[0][0]  # put min line index variable here
    x0_min = int(min_line_coordinates[0])
    y0_min = int(min_line_coordinates[1])
    x1_min = int(min_line_coordinates[2])
    y1_min = int(min_line_coordinates[3])

    cv2.line(binary_image_line_unit_RGB, (x0_min, y0_min), (x1_min, y1_min), (0, 255, 0), 1, cv2.LINE_AA)

# checking which mask belongs to which Bbox

    with open("co_with_ones.txt", "w") as output2:
        output2.write(str(co_with_ones))

    for bbox_index in range(len(bbox_list)):
        cv2.rectangle(img_for_cv_temp, (bbox_list[bbox_index][0], bbox_list[bbox_index][2]), (bbox_list[bbox_index][1], bbox_list[bbox_index][3]),
                  (255, 0, 0), 2)
        cv2.rectangle(img_for_cv, (bbox_list[bbox_index][0], bbox_list[bbox_index][2]),
                      (bbox_list[bbox_index][1], bbox_list[bbox_index][3]),
                      (255, 0, 0), 2)
        cv2.putText(img=img_for_cv_temp, text=str(bbox_index), org=(bbox_list[bbox_index][0], bbox_list[bbox_index][2]), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=3)
        cv2.putText(img=img_for_cv, text=str(bbox_index), org=(bbox_list[bbox_index][0], bbox_list[bbox_index][2]),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=3)

    # cv2.circle(binary_image_line_unit_RGB, (1427, 986), 5, (255, 0, 0), -1)

    for mask_index in range(len(bbox_list)-1):
        # print("mask index", mask_index)

        for mask_index_checked in range(len(co_with_ones)):

            for ones_index in range(len(co_with_ones[mask_index_checked])):
                # print("mask index", co_with_ones[mask_index][ones_index])
                if (bbox_list[mask_index][0] <= co_with_ones[mask_index_checked][ones_index][1] < bbox_list[mask_index][1]) and (bbox_list[mask_index][2] <= co_with_ones[mask_index_checked][ones_index][0] < bbox_list[mask_index][3]):

                    if mask_index != mask_index_checked:
                        bbox_ones_mask_list.append([mask_index, mask_index_checked])
                        # print("mask_index_checked", mask_index_checked)

                        intersected_comm_ones_list.append([(co_with_ones[mask_index_checked][ones_index][0], co_with_ones[mask_index_checked][ones_index][1]), mask_index, mask_index_checked])

                        # bbox_ones_mask_list1 = np.sort(np.array(bbox_ones_mask_list))

    intersected_masks_list = np.sort(np.array(bbox_ones_mask_list))
    intersected_masks_list = np.unique(intersected_masks_list, axis=0)
    # bbox_ones_mask_list1 = set(bbox_ones_mask_list)
    with open("intersected_masks_list.txt", "w") as output1:
        output1.write(str(intersected_masks_list))

    with open("intersected_comm_ones_list.txt", "w") as output1:
        output1.write(str(intersected_comm_ones_list))

    for save_comm in intersected_comm_ones_list:
        # print("save_comm", save_comm[1], save_comm[2])

        if save_comm[1] == 8 and save_comm[2] == 1: # common parts in 2nd image (in 1st image)
            save_comm_list_1.append(save_comm[0])

        if save_comm[1] == 1 and save_comm[2] == 8: # common parts in 2nd image (in 1st image)
            save_comm_list_2.append(save_comm[0])

    for coord_comm_1 in save_comm_list_1:
        img_temp_5[coord_comm_1] = 1

    for coord_comm_2 in save_comm_list_2:
        img_temp_4[coord_comm_2] = 1

    img_for_cv_temp_5 = img_temp_5.astype(np.uint8) * 255
    img_for_cv_temp_4 = img_temp_4.astype(np.uint8) * 255


## canny try
    edges_canny_temp_5 = cv2.Canny(img_for_cv_temp_5, 1, 250)
    edges_canny_temp_4 = cv2.Canny(img_for_cv_temp_4, 1, 250)

    and_edges_t4_t5 = cv2.bitwise_and(edges_canny_temp_5, edges_canny_temp_4)

    row_indexes, col_indexes = np.nonzero(and_edges_t4_t5)

    print("row_indexes",row_indexes,"col_indexes",col_indexes)


# # Line segment try
#     lsd_unit_2 = cv2.createLineSegmentDetector(0)
#     lines_unit_2 = lsd_unit_2.detect(img_for_cv_temp_5)[0]
#
#
#     copy_selected_list_units_2 = img_for_cv_temp_5 * 0
#
#     drawn_img_unit_2 = lsd_unit_2.drawSegments(copy_selected_list_units_2, lines_unit_2)
#     drawn_img_unit_gray_2 = cv2.cvtColor(drawn_img_unit_2, cv2.COLOR_BGR2GRAY)
#
#     binary_image_line_unit_2 = cv2.adaptiveThreshold(drawn_img_unit_gray_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                                                    cv2.THRESH_BINARY, 15, -2)
#
#     binary_image_line_unit_RGB_2 = cv2.cvtColor(binary_image_line_unit_2, cv2.COLOR_GRAY2BGR)




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
    resized_image_ori_3 = cv2.resize(img_for_cv_temp, dim, interpolation=cv2.INTER_AREA)
    resized_image_comm_5 = cv2.resize(and_edges_t4_t5, dim, interpolation=cv2.INTER_AREA)

    resized_image_ori_BB = cv2.resize(img_for_BB, dim, interpolation=cv2.INTER_AREA)
    resized_drawn_img = cv2.resize(drawn_img_gray, dim, interpolation=cv2.INTER_AREA)
    resized_binary_image_line_unit = cv2.resize(binary_image_line_unit, dim, interpolation=cv2.INTER_AREA)
    resized_binary_image_line_unit_RGB = cv2.resize(binary_image_line_unit_RGB, dim, interpolation=cv2.INTER_AREA)
    resized_image_ori_rgb = cv2.cvtColor(resized_image_ori, cv2.COLOR_GRAY2RGB)

    resized_image_ori_rgb_3 = cv2.cvtColor(resized_image_ori_3, cv2.COLOR_GRAY2RGB)

    # resized_horizontal = cv2.resize(horizontal, dim, interpolation=cv2.INTER_AREA)
    # resized_segmented_img_aft = cv2.resize(segmented_img_aft, dim, interpolation=cv2.INTER_AREA)

    # converting to 3 channel BGR image for display (https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html)
    # zeros_for_merge = np.zeros(resized_image_ori.shape[:2], dtype="uint8")
    # resized_image_ori_rgb = cv2.merge([resized_image_ori, resized_image_ori, resized_image_ori]) # BGR Channels

    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)

    # resized_branch_pts_mask_rgb = cv2.merge([zeros_for_merge, resized_branch_pts_mask, zeros_for_merge])
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)
    # resized_branch_pts_mask_rgb = cv2.cvtColor(resized_branch_pts_mask, cv2.COLOR_GRAY2RGB)

    # cv2.imwrite('C:/Users/Hasith Dasanayake/Desktop/image_shehan_sir.jpg', resized_image)

    cv2.imshow('original', resized_image_ori_rgb)
    # cv2.imshow('resized_image_ori_rgb_3', resized_image_ori_rgb_3)
    cv2.imshow('resized_image_comm_5', resized_image_comm_5)

    # cv2.imshow('resized_image_ori_BB', resized_image_ori_BB)
    # cv2.imshow('resized_binary_image_line_unit', resized_binary_image_line_unit)
    # cv2.imshow('resized_binary_image_line_unit_RGB', resized_binary_image_line_unit_RGB)

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
