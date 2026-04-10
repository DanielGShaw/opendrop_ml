#!/usr/bin/env python
# coding=utf-8
from opendrop_ml.modules.core.classes import ExperimentalDrop, ExperimentalSetup
from opendrop_ml.modules.preprocessing.preprocessing import extract_edges_cv
from opendrop_ml.utils.enums import ThresholdSelect

import matplotlib.pyplot as plt
import numpy as np
import cv2

# from __future__ import print_function
# from typing import List, Tuple
# import time
# import datetime

BLUR_SIZE = 3
VERSION_CV2 = cv2.__version__


def extract_drop_profile(
    raw_experiment: ExperimentalDrop, user_inputs: ExperimentalSetup
) -> None:
    if user_inputs.threshold_method == ThresholdSelect.USER_SELECTED:
        # profile_edges = detect_edges(raw_experiment.cropped_image, raw_experiment, user_inputs.drop_region)
        # profile, raw_experiment.ret = detect_edges(raw_experiment.cropped_image, raw_experiment, user_inputs.drop_region)
        raw_experiment.contour, raw_experiment.ret = detect_edges(
            raw_experiment.cropped_image,
            raw_experiment,
            user_inputs.drop_region,
            1,
            user_inputs.threshold_val,
        )

        if 1:
            plt.close("all")
            fig = plt.figure()
            plt.imshow(raw_experiment.cropped_image)
            plt.plot(raw_experiment.contour[:, 0],
                     raw_experiment.contour[:, 1], "r,")
            plt.title(
                f"Extracted drop profile\nTheshold value of : {raw_experiment.ret}"
            )
            plt.axis("equal")
            plt.show()
            plt.close(fig)

    elif user_inputs.threshold_method == ThresholdSelect.AUTOMATED:
        if raw_experiment.ret == None:
            raw_experiment.contour, raw_experiment.ret = extract_edges_cv(
                raw_experiment.cropped_image, return_threshold_value=True
            )

            if user_inputs.threshold_boole:
                # Clear all existing figures to avoid conflicts with residual plots
                plt.close("all")
                fig = plt.figure()  # Explicitly create a new figure window
                plt.imshow(raw_experiment.cropped_image)
                plt.plot(
                    raw_experiment.contour[:,
                                           0], raw_experiment.contour[:, 1], "r,"
                )
                plt.title(
                    f"Extracted drop profile\nThreshold value: {raw_experiment.ret:.2f}"
                )
                plt.axis("equal")
                plt.show()
                plt.close(fig)

        else:
            # if a threshold value has been selected then use this
            raw_experiment.contour = extract_edges_cv(
                raw_experiment.cropped_image,
                threshold_val=raw_experiment.ret,
                return_threshold_value=False,
            )

    # needle_crop = image_crop(raw_experiment.image, user_inputs.needle_region)
    # raw_experiment.needle_data, ret = detect_edges(needle_crop, raw_experiment, user_inputs.needle_region, raw_experiment.ret, 2)

    # # detect needle edges
    # needle_crop = image_crop(raw_experiment.image, user_inputs.needle_region)
    # raw_experiment.needle_data = detect_edges(needle_crop, user_inputs.needle_region)


def image_crop(image, points):  # loaded in conan.py
    # return image[min(y):max(y), min(x),max(x)]
    return image[
        int(points[0][1]): int(points[1][1]), int(points[0][0]): int(points[1][0])
    ]


def detect_edges(
    image: np.ndarray,
    raw_experiment: ExperimentalDrop,
    points,
    n_contours: int,
    threshold_value: int,
):

    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if 1:
        # apply Gaussian blur to drop edge
        blur = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)

    #    if ret == -1:
    # +cv2.THRESH_OTSU) # calculate thresholding
    ret, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
    # else:
    #     ret, thresh = cv2.threshold(blur,ret,255,cv2.THRESH_BINARY) # calculate thresholding
    # these values seem to agree with
    # - http://www.academypublisher.com/proc/isip09/papers/isip09p109.pdf
    # - http://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    # edges = cv2.Canny(thresh,0.5*ret,ret) # detect edges using Canny edge detection

    # error in PDT code - shouldn't threshold before Canny - otherwise Canny is useless
    # edges = cv2.Canny(blur,0.5*ret,ret) # detect edges using Canny edge detection

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contour_lengths = [length for length in cv2.arcLength(contours,0)] #list to hold all areas
    contour_lengths = [
        cv2.arcLength(contour, 0) for contour in contours
    ]  # list to hold all areas

    indexed_contour_lengths = np.array(contour_lengths).argsort()[::-1]
    indexed_contours_to_return = indexed_contour_lengths[:n_contours]

    image_height = raw_experiment.image.shape[0]

    offset = [0, 0]  # [0, image.shape[1]]
    points = []
    for index in indexed_contours_to_return:
        current_contour = contours[index][:, 0]
        for i in range(current_contour.shape[0]):
            current_contour[i, :] = current_contour[i, :] + offset
            # points.append(current_contour[current_contour[:,1].argsort()])

    size = 0
    cropped_image_height = image.shape[0]
    cropped_image_width = image.shape[1]
    # Trim edges from contour #current_contour error
    for i in range(current_contour.shape[0]):
        if current_contour[i, 0] != 0:
            if current_contour[i, 0] != cropped_image_width - 1:
                if current_contour[i, 1] != 0:
                    if current_contour[i, 1] != (cropped_image_height - 1):
                        size = size + 1
    #    print(current_contour.shape[0])
    #    print(size)
    contour_trimmed = np.zeros((size, 2))

    index = 0
    for i in range(current_contour.shape[0]):
        if current_contour[i, 0] != 0:
            if current_contour[i, 0] != cropped_image_width - 1:
                if current_contour[i, 1] != 0:
                    if current_contour[i, 1] != (cropped_image_height - 1):
                        contour_trimmed[index, :] = current_contour[i, :]
                        index = index + 1

    contour_x = contour_trimmed[:, 0]
    contour_y = contour_trimmed[:, 1]

    if 0:
        plt.axis("equal")
        plt.imshow(image)
        plt.plot(contour_x, contour_y, "rs", markerfacecolor="none")
        plt.show()
        # cv2.imshow('img',image)
        # cv2.imshow('img',thresh)
        # cv2.drawContours(image,contour_trimmed)
        # cv2.drawContours(image,contours,0,(0,255,0),10)
        # cv2.drawContours(image,contours,1,(255,255,0),10)
        # cv2.waitKey(0)

        # plt.imshow(image, origin='upper', cmap = 'gray')

    # find line between first and last contour points to estimate surface line
    N = np.shape(contour_trimmed)[0]
    A = 1
    xx = np.concatenate(
        (contour_trimmed[0:A, 0], contour_trimmed[N - A: N + 1, 0])
    )
    yy = np.concatenate(
        (contour_trimmed[0:A, 1], contour_trimmed[N - A: N + 1, 1])
    )

    coefficients = np.polyfit(xx, yy, 1)
    line = np.poly1d(coefficients)
    return contour_trimmed.astype(float), ret
    #

    # points = largest_contour[largest_contour[:,1].argsort()]

    # # determines the largest contour.
    # # hierarchy describes parent-child relationship
    # # this routine determines the length of each contour
    # # and returns the largest
    # drop_index = 0
    # maxLength = 0.0
    # for i in range(np.max(hierarchy+1)):
    #     length = cv2.arcLength(contours[i],0)
    #     # print(i, length)
    #     if length > maxLength:
    #         maxLength = length
    #         drop_index = i

    # # the largest contour
    # largest_contour = contours[drop_index][:,0]

    # # converts the data to (x, y) data where (0, 0) is the lower-left pixel
    # image_height = raw_experiment.image.shape[0]
    # offset = [points[0][0], image_height - points[0][1]]
    # for i in range(largest_contour.shape[0]):
    #     largest_contour[i,1] = - largest_contour[i,1]
    #     largest_contour[i,:] = largest_contour[i,:] + offset
    # points = largest_contour[largest_contour[:,1].argsort()]

    # return points, ret


# def calculate_needle_diameter(raw_experiment, fitted_drop_data, tolerances):


