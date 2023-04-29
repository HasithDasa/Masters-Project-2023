from json import load
import numpy as np
import cv2
from timeit import default_timer as timer

start_time_prep = timer()

def rle_decode(mask_rle, shape):
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


with open('training_done.json') as f:
    data = load(f)
    img_no = 8
    data1 = data['images'][img_no]  # selecting the image
    img_name = data1['image_name']
    img_name = img_name.replace('.png', '')
    img_width = data1['width']
    img_height = data1['height']
    img_temp_shape = (img_height, img_width)
    img_temp = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_5 = np.zeros(img_temp_shape, dtype=np.uint8)
    img_temp_4 = np.zeros(img_temp_shape, dtype=np.uint8)
    data2 = data1['labels']  # selecting the label key


    list_of_keys = []
    list_of_keys_t5 = []
    dic_for_selection = {}
    dic_for_selection_t4 = {}
    dic_for_selection_t5 = {}


    bbox_list = []
    lg_bbox_list =[]
    # co_with_ones = []
    bbox_ones_mask_list = []
    mask_list = []
    intersected_comm_ones_list = []
    save_comm_list_1 = []
    save_comm_list_2 = []

    # slope_angles_for_classification= []

# function => converting to opencv format (def opencv_format)

    for i in range(len(data2)):
        data3 = data2[i]  # selecting the class label
        mask_rle = data3['mask']
        bbox = data3['bbox']
        class_name = data3['class_name']


        if class_name == 'First Section Cutting' or class_name == 'Redundant Top End' or class_name == 'Redundant Bottom End' or class_name == 'Tip Cutting' or class_name == 'Non-Viable Part' or class_name == 'Second Section Cutting' or class_name == 'Third Section Cutting' or class_name == 'Fourth Section Cutting':
            # making mask
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

        # considering large bounding boxes to eliminate unwanted intersections of small bounding boxes
        if class_name == "Raw Cutting":
            # where exactly the mask situated in main image
            lg_row_start = bbox[1]
            lg_row_end = bbox[3]

            lg_col_start = bbox[0]
            lg_col_end = bbox[2]

            # collecting large bbox information
            lg_bbox_list.append([lg_col_start, lg_col_end, lg_row_start, lg_row_end])

    # print("lg_bbox_list", lg_bbox_list)

    edge_only_list =[]

    for j in range(len(list_of_keys)):
        # print(list_of_keys)

        # img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)

        class_name_from_list = list_of_keys[j]
        selected_list = dic_for_selection[class_name_from_list]
        mask_list.append(selected_list)
        # co_with_ones.append(selected_list)

        result_with_one = np.where(selected_list == 1)
        listOfCoordinates = list(zip(result_with_one[0], result_with_one[1]))

        # co_with_ones.append(listOfCoordinates)

        for cord in listOfCoordinates:
            img_temp_3[cord] = 1

        #img_contour_only = img_temp_3.astype(np.uint8) * 255


        contours_bb, hierarchy_bb = cv2.findContours(img_temp_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        co_with_ones_only_edge_elem_coord = []


        for co_with_ones_only_edge_elem in contours_bb:

            for point in co_with_ones_only_edge_elem:
                y, x = point[0]
                co_with_ones_only_edge_elem_coord.append((x, y))
            # co_with_ones_only_edge.append(co_with_ones_only_edge_elem_coord)



        edge_only_list.append(co_with_ones_only_edge_elem_coord)

        img_temp_3 = np.zeros(img_temp_shape, dtype=np.uint8)

    def process_first_for_loop(bbox_list, edge_only_list):
        bbox_ones_mask_list = []
        intersected_comm_ones_list = []


        for mask_index in range(len(bbox_list)):
            for mask_index_checked in range(len(edge_only_list)):
                for ones_index in range(len(edge_only_list[mask_index_checked])):
                    if (bbox_list[mask_index][0] <= edge_only_list[mask_index_checked][ones_index][1] < bbox_list[mask_index][1]) and (bbox_list[mask_index][2] <= edge_only_list[mask_index_checked][ones_index][0] < bbox_list[mask_index][3]):
                        if mask_index != mask_index_checked:
                            bbox_ones_mask_list.append([mask_index, mask_index_checked])
                            intersected_comm_ones_list.append([(edge_only_list[mask_index_checked][ones_index][0],
                                                                edge_only_list[mask_index_checked][ones_index][1]),
                                                               mask_index, mask_index_checked])
        return bbox_ones_mask_list, intersected_comm_ones_list # remove brackets for non parallel

    def process_second_for_loop(lg_bbox_list, edge_only_list):

        mask_inside_lg_bbox_list = []
        for lg_bb_ind, lg_bb_ele in enumerate(lg_bbox_list):
            for mask_index_2, mask_ele_2 in enumerate(edge_only_list):
                ones_count = 0
                for co_with_ones_ele2 in mask_ele_2:
                    if (lg_bb_ele[0] <= co_with_ones_ele2[1] < lg_bb_ele[1]) and (
                            lg_bb_ele[2] <= co_with_ones_ele2[0] < lg_bb_ele[3]):
                        ones_count += 1
                if ones_count == len(mask_ele_2):
                    mask_inside_lg_bbox_list.append([lg_bb_ind, mask_index_2])
        return mask_inside_lg_bbox_list


    mask_inside_lg_bbox_list = process_second_for_loop(lg_bbox_list, edge_only_list)
    bbox_ones_mask_list, intersected_comm_ones_list = process_first_for_loop(bbox_list, edge_only_list)

    unique_sublists = set()

    for sublist in bbox_ones_mask_list:
        if sublist[0] < sublist[1]:
            unique_sublists.add(tuple(sublist))
        else:
            unique_sublists.add(tuple(reversed(sublist)))

    # Convert the set back to a list of lists
    intersected_masks_list = [list(sublist) for sublist in unique_sublists]

    # Group sublists by their first element
    grouped_list = {}


    for sublist in mask_inside_lg_bbox_list:
        key = sublist[0]
        if key not in grouped_list:
            grouped_list[key] = []
        grouped_list[key].append(sublist)

    # Create new list with second elements of sublists in each group
    intersected_bbox_in_bigbb_list = []
    for group in grouped_list.values():
        new_sublist = [sublist[1] for sublist in group]
        intersected_bbox_in_bigbb_list.append(new_sublist)

    # Create a set of all unique elements in intersected_bbox_in_bigbb_list
    all_elements = set()
    for sublist in intersected_bbox_in_bigbb_list:
        all_elements.update(sublist)

    intersected_masks_list_final = []

    for mask in intersected_masks_list:
        found = False
        for bbox in intersected_bbox_in_bigbb_list:
            if all(elem in bbox for elem in mask):
                intersected_masks_list_final.append(mask)
                found = True
                break
        if not found:
            continue

# function => releasing image unit (def rel_img_unit)

    #going on one by one on intersected masks

    for ele_ind, ele_int_list in enumerate(intersected_masks_list_final):

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

    end_time_prep = timer()
    elapsed_time_prep = (end_time_prep - start_time_prep) * 1000
    print("elapsed_time_prep", elapsed_time_prep)

