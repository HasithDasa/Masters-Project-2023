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
    data1 = data['images'][13]  # selecting the image
    img_name = data1['image_name']
    img_width = data1['width']
    img_height = data1['height']
    img_temp_shape = (img_height, img_width)
    img_temp = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_5 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_4 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_6 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_8 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_last_unit = np.zeros(img_temp_shape, dtype=np.uint8)
    data2 = data1['labels']  # selecting the label key
    print(img_name)

    list_of_keys = []
    list_of_keys_t4 = []
    list_of_keys_t5 = []
    dic_for_selection = {}
    dic_for_selection_t4 = {}
    dic_for_selection_t5 = {}

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
    with_all_masks_list = []

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

            img_temp[row_start:row_end, col_start:col_end] = mask

            variable = str(i)
            key = "lable" + variable
            list_of_keys.append(key)
            dic_for_selection[key] = img_temp

            # including mask in the main image, need to make it zero for everytime because of mask coincidence problem
            img_temp = np.zeros(img_temp_shape, dtype=np.uint8)


    #one by one display temperary

    class_name_from_list_6 = list_of_keys[3] #change unit
    selected_list_6 = dic_for_selection[class_name_from_list_6]

    result_with_one_6 = np.where(selected_list_6 == 1)
    listOfCoordinates_6 = list(zip(result_with_one_6[0], result_with_one_6[1]))

    for cord_6 in listOfCoordinates_6:
        img_temp_6[cord_6] = 1

    img_for_cv_temp_6 = img_temp_6.astype(np.uint8) * 255

    # line segmentation trial
    # img_for_cv_temp_7 = img_temp_6.astype(np.uint8) * 255
    #
    # lsd_unit_1 = cv2.createLineSegmentDetector(refine = 1, scale = 0.5, sigma_scale = 0.6, quant = 0.5, ang_th = 5.5, log_eps = 0, density_th = 0.1)
    #
    # lines_unit_1 = lsd_unit_1.detect(img_for_cv_temp_7)[0]
    #
    # copy_selected_list_units_7 = img_for_cv_temp_7 * 0
    #
    # drawn_img_unit_7 = lsd_unit_1.drawSegments(copy_selected_list_units_7, lines_unit_1)
    #
    # # Create the kernel
    # kernel = np.ones((7, 7), np.uint8)
    #
    # # Perform dilation
    # drawn_img_unit_7_dil = cv2.dilate(drawn_img_unit_7, kernel, iterations=2)
    # drawn_img_unit_7_dil = cv2.erode(drawn_img_unit_7_dil, kernel, iterations=2)
    #
    # drawn_img_unit_gray_7 = cv2.cvtColor(drawn_img_unit_7, cv2.COLOR_BGR2GRAY)
    #
    # # img_for_cv_temp_7_RGB = cv2.cvtColor(img_for_cv_temp_7, cv2.COLOR_GRAY2BGR)
    #
    # binary_image_line_unit_7 = cv2.adaptiveThreshold(drawn_img_unit_gray_7, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                                                cv2.THRESH_BINARY, 15, -2)
    #
    # substracted_img = drawn_img_unit_7_dil - drawn_img_unit_7
    #
    # substracted_img = cv2.medianBlur(substracted_img, 5)
    #
    # substracted_img = cv2.cvtColor(substracted_img, cv2.COLOR_BGR2GRAY)
    #
    # # Find contours
    # contours, hierarchy = cv2.findContours(substracted_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # if len(contours) > 1:
    #     # Filter for contours with larger area
    #     min_area = 200
    #     large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    #
    # else:
    #     large_contours = contours
    #
    # single_contour = large_contours
    # # single_contour = [np.vstack(large_contours)]
    #
    # # Create a black image with the same size as the original image
    # large_contours_img = np.zeros_like(substracted_img)
    #
    # # Draw large contours on the image
    # cv2.drawContours(large_contours_img, single_contour, -1, (255, 0, 0), -1)
    #
    # # Save the image to a file
    # cv2.imwrite("leaves_plant.png", large_contours_img)



# considering as whole image

    for j in range(len(list_of_keys)):
        # print(list_of_keys)

        class_name_from_list = list_of_keys[j]
        selected_list = dic_for_selection[class_name_from_list]
        mask_list.append(selected_list)
        # co_with_ones.append(selected_list)

        result_with_one = np.where(selected_list == 1)
        listOfCoordinates = list(zip(result_with_one[0], result_with_one[1]))

        co_with_ones.append(listOfCoordinates)

        for cord in listOfCoordinates:
            img_temp_3[cord] = 1

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

    # for j_3 in range(len(list_of_keys)):
    #
    #     class_name_from_list_3 = list_of_keys[j_3]
    #     selected_list_3 = dic_for_selection[class_name_from_list_3]
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
    #     # remove automatically coming last bounding box element from all the other bounding box elements
    #     if j_3 <= len(list_of_keys) - 2:
    #         selected_list_3[row_start_last:row_end_last, col_start_last:col_end_last] = 0 #making last one zero
    #
    #         result_with_one_3 = np.where(selected_list_3 == 1)
    #         listOfCoordinates_3 = list(zip(result_with_one_3[0], result_with_one_3[1]))
    #
    #         # co_with_ones.append(listOfCoordinates_3)
    #
    #         for cord_3 in listOfCoordinates_3:
    #             img_temp_3[cord_3] = 1
    #
    #         with_all_masks_list.append(img_temp_3)
    #
    #     #special operation on the last element
    #     # else:
    #     #     # print("fail")
    #     #     # img_temp_3 = 0
    #     #     img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)
    #     #
    #     #     last_unit = selected_list_3[row_start_last:row_end_last, col_start_last:col_end_last]
    #     #     img_temp_last_unit[row_start_last:row_end_last, col_start_last:col_end_last] = last_unit
    #     #
    #     #     result_with_one_4 = np.where(img_temp_last_unit == 1)
    #     #     listOfCoordinates_4 = list(zip(result_with_one_4[0], result_with_one_4[1]))
    #     #
    #     #     co_with_ones.append(listOfCoordinates_4)
    #     #
    #     #     for cord_4 in listOfCoordinates_4:
    #     #         img_temp_3[cord_4] = 1
    #
    #     else:
    #         # last_element_list = [i_7 for i_7 in with_all_masks_list if i_7 not in co_with_ones]
    #
    #         for ind_5, val_5 in enumerate(with_all_masks_list):
    #             img_temp_3 = img_temp - val_5


    # img_temp_6 = img_temp_3



    # plt.imshow(img_temp, 'gray')
    # plt.show()
    # plt.imsave('test_img.png', img_temp)

    # converting to 8bit image format that opencv can read




    img_for_cv = img_temp_3.astype(np.uint8) * 255
    img_for_BB = img_temp_3.astype(np.uint8) * 255
    # img_for_cv_temp = img_temp_3.astype(np.uint8) * 255
    # img_for_cv_temp = img_temp_last_unit.astype(np.uint8) * 255

    # thresh_image_bbox = cv2.threshold(img_for_cv, 0, 254, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(img_temp_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    class_name_from_list_units = list_of_keys[0] # changing type of branch or plant unit
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
        cv2.rectangle(img_for_cv, (bbox_list[bbox_index][0], bbox_list[bbox_index][2]),
                      (bbox_list[bbox_index][1], bbox_list[bbox_index][3]),
                      (255, 0, 0), 2)
        cv2.putText(img=img_for_cv, text=str(bbox_index), org=(bbox_list[bbox_index][0], bbox_list[bbox_index][2]),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=3)

    # cv2.circle(binary_image_line_unit_RGB, (1427, 986), 5, (255, 0, 0), -1)

    for mask_index in range(len(bbox_list)):
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

    # intersected_masks_list_python = intersected_masks_list.tolist()
    # print('intersected_masks_list_python', intersected_masks_list_python)

    # bbox_ones_mask_list1 = set(bbox_ones_mask_list)
    with open("intersected_masks_list.txt", "w") as output1:
        output1.write(str(intersected_masks_list))

    # with open("intersected_comm_ones_list.txt", "w") as output1:
    #    output1.write(str(intersected_comm_ones_list))

    #going on one by one on intersected masks

    for ele_ind, ele_int_list in enumerate(intersected_masks_list):

        print("ele_int_list", ele_int_list)

        for save_comm in intersected_comm_ones_list:
            # print("save_comm", save_comm[1], save_comm[2])
            if save_comm[1] == ele_int_list[0] and save_comm[2] == ele_int_list[1]: # common parts in 2nd image (in 1st image)
                save_comm_list_1.append(save_comm[0])

            if save_comm[1] == ele_int_list[1] and save_comm[2] == ele_int_list[0]: # common parts in 2nd image (in 1st image)
                save_comm_list_2.append(save_comm[0])

        for coord_comm_1 in save_comm_list_1:
            img_temp_5[coord_comm_1] = 1

        for coord_comm_2 in save_comm_list_2:
            img_temp_4[coord_comm_2] = 1

        variable_t5 = str(ele_ind)
        key_t5 = "t5" + variable_t5
        list_of_keys_t5.append(key_t5)

        dic_for_selection_t5[key_t5] = img_temp_5
        dic_for_selection_t4[key_t5] = img_temp_4

        # making image masks zero after iteration
        img_temp_5 = np.zeros(img_temp_shape, dtype=np.uint8)
        img_temp_4 = np.zeros(img_temp_shape, dtype=np.uint8)

    final_completed_image_list = []
    and_list = []



    for list_of_keys_t5_ind in range(len(list_of_keys_t5)):
        class_name_from_list_of_keys_t5 = list_of_keys_t5[list_of_keys_t5_ind]
        selected_list_t5 = dic_for_selection_t5[class_name_from_list_of_keys_t5]
        selected_list_t4 = dic_for_selection_t4[class_name_from_list_of_keys_t5]

        # for stem detection
        selected_list_t7 = dic_for_selection_t5[class_name_from_list_of_keys_t5].astype(np.uint8) * 255
        selected_list_t6 = dic_for_selection_t4[class_name_from_list_of_keys_t5].astype(np.uint8) * 255


#  Canny Try
        if list_of_keys_t5_ind > 0:
            inverse_selected_list_t5 = cv2.bitwise_not(dic_for_selection_t5[list_of_keys_t5[list_of_keys_t5_ind - 1]])
            selected_list_t5 = (cv2.bitwise_and(selected_list_t5, inverse_selected_list_t5)).astype(np.uint8) * 255

            inverse_selected_list_t4 = cv2.bitwise_not(dic_for_selection_t4[list_of_keys_t5[list_of_keys_t5_ind - 1]])
            selected_list_t4 = (cv2.bitwise_and(selected_list_t4, inverse_selected_list_t4)).astype(np.uint8) * 255

        else:
            selected_list_t5 = selected_list_t5.astype(np.uint8) * 255
            selected_list_t4 = selected_list_t4.astype(np.uint8) * 255


        ## canny try
        edges_canny_temp_5 = cv2.Canny(selected_list_t5, 1, 250)
        edges_canny_temp_4 = cv2.Canny(selected_list_t4, 1, 250)

        and_edges_t4_t5 = cv2.bitwise_and(edges_canny_temp_5, edges_canny_temp_4)

        # Find contours in the grayscale image
        contours, hierarchy = cv2.findContours(and_edges_t4_t5, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the length of each contour
        for contour in contours:
            contour_length = cv2.arcLength(contour, True)
            print("Contour length:", contour_length)




        # # Define the kernel size for dilation
        # kernel_2 = np.ones((2, 2), np.uint8)
        #
        # # Apply dilation to the binary image
        # and_edges_t4_t5 = cv2.dilate(and_edges_t4_t5, kernel_2, iterations=1)


# # Sobel Try
#         if list_of_keys_t5_ind > 0:
#             inverse_selected_list_t5 = cv2.bitwise_not(dic_for_selection_t5[list_of_keys_t5[list_of_keys_t5_ind - 1]])
#             selected_list_t5 = (cv2.bitwise_and(selected_list_t5, inverse_selected_list_t5))
#
#             inverse_selected_list_t4 = cv2.bitwise_not(dic_for_selection_t4[list_of_keys_t5[list_of_keys_t5_ind - 1]])
#             selected_list_t4 = (cv2.bitwise_and(selected_list_t4, inverse_selected_list_t4))
#
#         else:
#             selected_list_t5 = selected_list_t5
#             selected_list_t4 = selected_list_t4
#
#         # Apply Sobel operator t5
#         sobelx_t5 = cv2.Sobel(selected_list_t5, cv2.CV_64F, 1, 0, ksize=7)
#         sobely_t5 = cv2.Sobel(selected_list_t5, cv2.CV_64F, 0, 1, ksize=7)
#         sobel_t5 = sobelx_t5 + sobely_t5
#
#         # Apply Sobel operator t4
#         sobelx_t4 = cv2.Sobel(selected_list_t4, cv2.CV_64F, 1, 0, ksize=7)
#         sobely_t4 = cv2.Sobel(selected_list_t4, cv2.CV_64F, 0, 1, ksize=7)
#         sobel_t4 = sobelx_t4 + sobely_t4
#
#         and_edges_t4_t5 = cv2.bitwise_and(sobel_t5, sobel_t4)
#         and_edges_t4_t5 = and_edges_t4_t5.astype(np.uint8) * 255


        #LoG

        # if list_of_keys_t5_ind > 0:
        #     inverse_selected_list_t5 = cv2.bitwise_not(dic_for_selection_t5[list_of_keys_t5[list_of_keys_t5_ind - 1]])
        #     selected_list_t5 = (cv2.bitwise_and(selected_list_t5, inverse_selected_list_t5))
        #
        #     inverse_selected_list_t4 = cv2.bitwise_not(dic_for_selection_t4[list_of_keys_t5[list_of_keys_t5_ind - 1]])
        #     selected_list_t4 = (cv2.bitwise_and(selected_list_t4, inverse_selected_list_t4))
        #
        # else:
        #     selected_list_t5 = selected_list_t5
        #     selected_list_t4 = selected_list_t4
        #
        # gaussian_image_t5 = cv2.GaussianBlur(selected_list_t5, (3, 3), 0)
        # log_t5 = cv2.Laplacian(gaussian_image_t5, cv2.CV_64F)
        #
        # gaussian_image_t4 = cv2.GaussianBlur(selected_list_t4, (3, 3), 0)
        # log_t4 = cv2.Laplacian(gaussian_image_t4, cv2.CV_64F)
        #
        # and_edges_t4_t5 = cv2.bitwise_and(log_t5, log_t4)
        # and_edges_t4_t5 = and_edges_t4_t5.astype(np.uint8) * 255

        and_list.append(and_edges_t4_t5)

        row_indexes, col_indexes = np.nonzero(and_edges_t4_t5)

        print("row_indexes", row_indexes, "col_indexes", col_indexes)

        clustering_list = []
        clustering_len_list = []
        clustering_needed = False
        clustering_type = "row"

        for row_ind, row_val in enumerate(row_indexes):
            if row_ind < (len(row_indexes)-1):
                if abs(row_val-row_indexes[row_ind+1]) > 3: # checking the distance between two neighbouring pixels are higher than 3
                    clustering_list.append(row_ind)
                    clustering_type = "row"
                    print("clustering_type", clustering_type)

        if len(clustering_list) == 0:

            for col_ind, col_val in enumerate(col_indexes):
                if col_ind < (len(col_indexes) - 1):
                    if abs(col_val-col_indexes[col_ind+1]) > 3:
                        clustering_list.append(col_ind)
                        clustering_type = "col"
                        print("clustering_type", clustering_type)


        print("clustering_list", clustering_list)

        if len(clustering_list) > 0:
            clustering_needed = True
        else:
            clustering_needed = False

        if clustering_needed:
            # clustering based on the neighbouring pixels locations based on distance
            clustering_len_list = [clustering_list[0] + 1] # 1st element of the clustering len list

            for index_clus in range(len(clustering_list)): # any intermediate element of clustering len list
                if index_clus > 0:
                    clustering_len_list.append(clustering_list[index_clus] - clustering_list[index_clus-1])

            if len(clustering_list) > 0: # last element of the clustering len list
                clustering_len_list.append(len(row_indexes) - (clustering_list[len(clustering_list)-1] + 1))

            print("clustering_len_list", clustering_len_list)
            # checking the min index of the clustering len index, please add the code based on the length value, if length ==2 then ignore, then go to the next length
            # min_clus_len_index = np.argmin(clustering_len_list)

            # index of the minimum element of clustering_len_list
            min_index_clus = clustering_len_list.index(min(clustering_len_list))

            if len(clustering_len_list) > 2: #Check if clustering_len_list has more than two elements

                if clustering_len_list[min_index_clus] in [1, 2]: # Check if the minimum element is 1 or 2
                    # If it is, find the next smallest element
                    next_min_clus = min(x for x in clustering_len_list if x not in [1, 2]) # find the next minimum length after 1 or 2

                    # next_min_clus = min([x for x in clustering_len_list if x not in [1, 2]])

                    # elements of the next smallest element even if the two elements are same or neighbouring element
                    next_min_clus_list = list(set([elem for elem in clustering_len_list if abs(next_min_clus - elem) < 3]))

                    # next_min_clus_list = [x_ele for x_ele in next_min_clus_list if x_ele not in [1, 2]]

                    duplicates = set([x_clus for x_clus in clustering_len_list if x_clus not in [1, 2] and clustering_len_list.count(x_clus) > 1])

                    if duplicates:
                        print("Duplicate elements exist in the list:")

                        next_min_freq_list = [0] * len(next_min_clus_list)
                        freq_len_list = [0] * len(next_min_clus_list)

                        # loop through elements in next_min_clus_list
                        for ind_nxt, ele_nxt in enumerate(next_min_clus_list):
                            # count the frequency of each element in clustering_len_list
                            count_freq = clustering_len_list.count(ele_nxt)
                            # multiply the frequency by the corresponding element in next_min_clus_list
                            next_min_freq_list[ind_nxt] = count_freq * ele_nxt
                            freq_len_list[ind_nxt] = count_freq

                        # next_min_clus_list = [x_2 for x_2 in next_min_clus_list if x_2 not in [1, 2]]# remove 1 and 2 out from the list

                        print("next_min_clus_list: ", next_min_clus_list)
                        print("next_min_freq_list: ", next_min_freq_list)
                        print("freq_len_list: ", freq_len_list)

                        cluster_coor_list_2 = []
                        min_clus_len_index_list_all = []

                        for next_min_clus_ele in next_min_clus_list:

                            min_clus_len_index_list = [i_33 for i_33, x_33 in enumerate(clustering_len_list) if x_33 == next_min_clus_ele]

                            #Collecting all the elments in min_clus_len_index_lists
                            for elem_min_clus_len_index_list in min_clus_len_index_list:
                                min_clus_len_index_list_all.append(elem_min_clus_len_index_list)


                            print("min_clus_len_index_list:", min_clus_len_index_list)
                            print("min_clus_len_index_list_all:", min_clus_len_index_list_all)

                            for min_clus_len_index in min_clus_len_index_list:

                                # when min length cluster index is 1st one, last one or any other one
                                min_clus_index = []
                                if min_clus_len_index == 0:  # 1st one
                                    min_clus_index = [0, clustering_list[0]]
                                elif min_clus_len_index == len(clustering_len_list) - 1:  # last one
                                    min_clus_index = [clustering_list[len(clustering_list) - 1] + 1, len(row_indexes) - 1]
                                else:  # middle one
                                    min_clus_index = [clustering_list[min_clus_len_index - 1] + 1,
                                                      clustering_list[min_clus_len_index]]

                                print("min_clus_index", min_clus_index)

                                if clustering_type == "row":

                                    for row_ind_sel, row_val_sel in enumerate(row_indexes):
                                        # checking if row index within the selected clustering range, eg: index from range 0 to 9
                                        if min_clus_index[0] <= row_ind_sel <= min_clus_index[1]:
                                            cluster_coor_list_2.append((col_indexes[row_ind_sel], row_val_sel))
                                        # checking if length of the line is only with one coordinate then taking the rest of the coordinates
                                        elif ((min_clus_index[0] - min_clus_index[1]) == 0) and (min_clus_index[0] == row_ind_sel):
                                            cluster_coor_list_2.append((col_indexes[row_ind_sel], row_val_sel))

                                elif clustering_type == "col":

                                    for col_ind_sel, col_val_sel in enumerate(col_indexes):
                                        # checking if row index within the selected clustering range, eg: index from range 0 to 9
                                        if min_clus_index[0] <= col_ind_sel <= min_clus_index[1]:
                                            cluster_coor_list_2.append((col_val_sel, row_indexes[col_ind_sel]))
                                        # checking if length of the line is only with one coordinate then taking the rest of the coordinates
                                        elif ((min_clus_index[0] - min_clus_index[1]) == 0) and (min_clus_index[0] == col_ind_sel):
                                            cluster_coor_list_2.append((col_val_sel, row_indexes[col_ind_sel]))

                            print("cluster_coor_list_2", cluster_coor_list_2)

                            dis_lin_pix_to_contu_list = []

                            key_for_clus = intersected_masks_list[list_of_keys_t5_ind]

                            print("key_for_clus[0]:", key_for_clus[0])

                            class_name_from_list_8 = list_of_keys[key_for_clus[0]]  # change unit
                            selected_list_8 = dic_for_selection[class_name_from_list_8]

                            result_with_one_8 = np.where(selected_list_8 == 1)
                            listOfCoordinates_8 = list(zip(result_with_one_8[0], result_with_one_8[1]))

                            for cord_8 in listOfCoordinates_8:
                                img_temp_8[cord_8] = 1

                            img_for_cv_temp_7 = img_temp_8.astype(np.uint8) * 255

                            # img_for_cv_temp_7 = selected_list_t7

                            cv2.imwrite("img_for_cv_temp_7.png", img_for_cv_temp_7)

                            lsd_unit_1 = cv2.createLineSegmentDetector(refine=1, scale=0.5, sigma_scale=0.6, quant=0.5,
                                                                       ang_th=5.5, log_eps=0, density_th=0.1)

                            lines_unit_1 = lsd_unit_1.detect(img_for_cv_temp_7)[0]

                            copy_selected_list_units_7 = img_for_cv_temp_7 * 0

                            drawn_img_unit_7 = lsd_unit_1.drawSegments(copy_selected_list_units_7, lines_unit_1)

                            cv2.imwrite("drawn_img_unit_7.png", drawn_img_unit_7)

                            # Create the kernel
                            kernel = np.ones((7, 7), np.uint8)

                            # Perform dilation
                            drawn_img_unit_7_dil = cv2.dilate(drawn_img_unit_7, kernel, iterations=2)
                            drawn_img_unit_7_dil = cv2.erode(drawn_img_unit_7_dil, kernel, iterations=2)

                            drawn_img_unit_gray_7 = cv2.cvtColor(drawn_img_unit_7, cv2.COLOR_BGR2GRAY)

                            # img_for_cv_temp_7_RGB = cv2.cvtColor(img_for_cv_temp_7, cv2.COLOR_GRAY2BGR)

                            binary_image_line_unit_7 = cv2.adaptiveThreshold(drawn_img_unit_gray_7, 255,
                                                                             cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                                             cv2.THRESH_BINARY, 15, -2)

                            substracted_img = drawn_img_unit_7_dil - drawn_img_unit_7

                            substracted_img = cv2.medianBlur(substracted_img, 5)

                            substracted_img = cv2.cvtColor(substracted_img, cv2.COLOR_BGR2GRAY)

                            # Find contours
                            contours, hierarchy = cv2.findContours(substracted_img, cv2.RETR_LIST,
                                                                   cv2.CHAIN_APPROX_SIMPLE)

                            # if len(contours) > 1:
                            #     # Filter for contours with larger area
                            #     min_area =200
                            #     large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                            #
                            # else:
                            #     large_contours = contours

                            single_contour = max(contours, key=cv2.contourArea)

                            # single_contour = large_contours

                            # single_contour = max(contours, key=cv2.contourArea)


                            # Create a black image with the same size as the original image
                            large_contours_img = np.zeros_like(substracted_img)

                            # Draw large contours on the image
                            cv2.drawContours(large_contours_img, single_contour, -1, (255, 0, 0), -1)

                            cv2.imwrite("large_contours_img.png", large_contours_img)

                            # # Check if the pixel is inside the contour
                            # nonzero_indices = np.nonzero(large_contours_img)
                            # nonzero_x, nonzero_y = nonzero_indices
                            # nonzero_points = list(zip(nonzero_x, nonzero_y))

                            # Calculate the moments of the contour
                            moments = cv2.moments(single_contour)

                            # Calculate the center of mass of the contour
                            center_x = int(moments['m10'] / moments['m00'])
                            center_y = int(moments['m01'] / moments['m00'])

                            # nonzero_points = list(zip(center_x, center_y))


                            min_dis_lin_pix_to_contu_list = []

                            for cluster_coor_list_elem in cluster_coor_list_2:

                                dis_lin_pix_to_contu_list = []

                                if len(contours) > 1: # only taking when it has a contour

                                    # for nozero_coord in nonzero_points:
                                    dis_lin_pix_to_contu = round(np.sqrt((center_y - cluster_coor_list_elem[0]) ** 2 + (center_x - cluster_coor_list_elem[1]) ** 2), 3)
                                    dis_lin_pix_to_contu_list.append(dis_lin_pix_to_contu)

                                    # print("dis_lin_pix_to_contu_list_len", len(dis_lin_pix_to_contu_list))

                                    # min_dis_lin_pix_to_contu_list.append(sum(dis_lin_pix_to_contu_list)/len(dis_lin_pix_to_contu_list))
                                    # min_dis_lin_pix_to_contu_list.append(min(dis_lin_pix_to_contu_list))
                                    min_dis_lin_pix_to_contu_list.append(dis_lin_pix_to_contu_list)

                        print("min_dis_lin_pix_to_contu_list", min_dis_lin_pix_to_contu_list)
                        print("min_dis_lin_pix_to_contu_list_len", len(min_dis_lin_pix_to_contu_list))

                        if len(contours) > 1: # only taking when it has a contour

                            # next_min_clus_list,  each and every element represent how many elements that I need to consider to get the minimum value out of "min_dis_lin_pix_to_contu_list" list. As an example the minimum value out of 1st, 9 elements in the "min_dis_lin_pix_to_contu_list" then save that value in a seperate list, after that take minimum value out of next 7 elements and save it in the same seperate list
                            # min_vals_list = []
                            # start_idx = 0
                            # for num in next_min_freq_list:
                            #     min_val = min(min_dis_lin_pix_to_contu_list[start_idx:start_idx + num])
                            #     min_vals_list.append(min_val)
                            #     start_idx += num

                            # clustering_len_element = next_min_clus_list[min_vals_list.index(min(min_vals_list))]



                            # print("min_vals_list", min_vals_list)
                            # print("clustering_len_element", clustering_len_element)

                            freq_position_min_list = []
                            j_nxt = 0
                            for i_nxt in range(len(next_min_clus_list)):
                                for k_nxt in range(freq_len_list[i_nxt]):
                                    for l_nxt in range(next_min_clus_list[i_nxt]):
                                        freq_position_min_list.append(min_clus_len_index_list_all[j_nxt])
                                    j_nxt = j_nxt + 1

                            print("freq_position_min_list", freq_position_min_list)


                        min_clus_len_index = freq_position_min_list[min_dis_lin_pix_to_contu_list.index(min(min_dis_lin_pix_to_contu_list))]  # getting the index

                    else:
                        min_clus_len_index = clustering_len_list.index(next_min_clus)


                else:
                    min_clus_len_index = min_index_clus
            else: #Check if clustering_len_list has less than two elements
                if clustering_len_list[min_index_clus] == 1:
                    # If it is, find the next smallest element
                    next_min_clus = min(x for x in clustering_len_list if x != 1)
                    # index of the next smallest element
                    min_clus_len_index = clustering_len_list.index(next_min_clus)

                else:
                    min_clus_len_index = min_index_clus


            print("min_clus_len_index:", min_clus_len_index)

        # when min length cluster index is 1st one, last one or any other one
            min_clus_index = []
            if min_clus_len_index == 0: #1st one
                min_clus_index = [0, clustering_list[0]]
            elif min_clus_len_index == len(clustering_len_list) - 1: #last one
                min_clus_index = [clustering_list[len(clustering_list)-1]+1, len(row_indexes)-1]
            else:# middle one
                min_clus_index = [clustering_list[min_clus_len_index -1]+1, clustering_list[min_clus_len_index]]

            print("min_clus_index", min_clus_index)

            cluster_coor_list = []
            if clustering_type == "row":

                for row_ind_sel, row_val_sel in enumerate(row_indexes):
                    # checking if row index within the selected clustering range, eg: index from range 0 to 9
                    if min_clus_index[0] <= row_ind_sel <= min_clus_index[1]:
                        cluster_coor_list.append([col_indexes[row_ind_sel], row_val_sel])
                    # checking if length of the line is only with one coordinate then taking the rest of the coordinates
                    elif (min_clus_index[0] - min_clus_index[1]) == 0:
                        cluster_coor_list.append([col_indexes[row_ind_sel], row_val_sel])

                print("cluster_coor_list", cluster_coor_list)

            elif clustering_type == "col":

                for col_ind_sel, col_val_sel in enumerate(col_indexes):
                    # checking if row index within the selected clustering range, eg: index from range 0 to 9
                    if min_clus_index[0] <= col_ind_sel <= min_clus_index[1]:
                        cluster_coor_list.append([col_val_sel, row_indexes[col_ind_sel]])
                    # checking if length of the line is only with one coordinate then taking the rest of the coordinates
                    elif (min_clus_index[0] - min_clus_index[1]) == 0:
                        cluster_coor_list.append([col_val_sel, row_indexes[col_ind_sel]])

                print("cluster_coor_list", cluster_coor_list)



        else:
            cluster_coor_list = []

            if clustering_type == "row":
                for row_ind_sel, row_val_sel in enumerate(row_indexes):
                    cluster_coor_list.append([col_indexes[row_ind_sel], row_val_sel])

                print("cluster_coor_list_not_needed", cluster_coor_list)

            elif clustering_type == "col":
                for col_ind_sel, col_val_sel in enumerate(col_indexes):
                    cluster_coor_list.append([col_val_sel, row_indexes[col_ind_sel]])

                print("cluster_coor_list_not_needed", cluster_coor_list)

        cluster_coor_list = np.array(cluster_coor_list)
        cluster_coor_list = cluster_coor_list.reshape((-1, 1, 2))

        saving_image_ori = img_for_cv

        saving_image_ori_rgb = cv2.cvtColor(saving_image_ori, cv2.COLOR_GRAY2RGB)

        # color, thickness and isClosed
        color_cl = (0, 0, 255)
        thickness_cl = 3
        isClosed_cl = False

        # drawPolyline
        cv2.polylines(saving_image_ori_rgb, [cluster_coor_list], isClosed_cl, color_cl,
                                                 thickness_cl)

        final_completed_image_list.append(saving_image_ori_rgb)

    for final_indx, final_elem in enumerate(final_completed_image_list):

        new_name_1 = intersected_masks_list[final_indx][0]
        new_name_2 = intersected_masks_list[final_indx][1]
        intersected_masks_name = "final_images/Intersected_mask_%d_%d_%d.jpg" % (new_name_1, new_name_2, final_indx)
        print(intersected_masks_name)
        cv2.imwrite(intersected_masks_name, final_elem)

    for and_indx, and_elem in enumerate(and_list):

        and_name_1 = intersected_masks_list[and_indx][0]
        and_name_2 = intersected_masks_list[and_indx][1]
        intersected_line_masks_name = "final_images_lines/Intersected_line_mask_%d_%d_%d.jpg" % (and_name_1, and_name_2, and_indx)
        # print(intersected_masks_name)
        cv2.imwrite(intersected_line_masks_name, and_elem)











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
    image_ori_rgb = cv2.cvtColor(img_for_cv, cv2.COLOR_GRAY2RGB)
    resized_image_ori_rgb = cv2.resize(image_ori_rgb, dim, interpolation=cv2.INTER_AREA)

    # resized_image_ori_3 = cv2.resize(img_for_cv_temp, dim, interpolation=cv2.INTER_AREA)
    # resized_image_comm_5 = cv2.resize(and_edges_t4_t5, dim, interpolation=cv2.INTER_AREA)

    resized_img_for_cv_temp_6 = cv2.resize(img_for_cv_temp_6, dim, interpolation=cv2.INTER_AREA)


    resized_image_ori_BB = cv2.resize(img_for_BB, dim, interpolation=cv2.INTER_AREA)
    resized_drawn_img = cv2.resize(drawn_img_gray, dim, interpolation=cv2.INTER_AREA)
    resized_binary_image_line_unit = cv2.resize(binary_image_line_unit, dim, interpolation=cv2.INTER_AREA)
    resized_binary_image_line_unit_RGB = cv2.resize(binary_image_line_unit_RGB, dim, interpolation=cv2.INTER_AREA)

    # resized_image_ori_rgb = cv2.cvtColor(resized_image_ori, cv2.COLOR_GRAY2RGB)

    # resized_image_ori_rgb_3 = cv2.cvtColor(resized_image_ori_3, cv2.COLOR_GRAY2RGB)

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
    # cv2.imshow('resized_image_comm_5', resized_image_comm_5)
    cv2.imshow('resized_img_for_cv_temp_6', resized_img_for_cv_temp_6)

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
