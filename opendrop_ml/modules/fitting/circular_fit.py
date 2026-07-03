#!/usr/bin/env python
# coding=utf-8

"""This code serves as a discrete instance of the circular fit method of
contact angle analysis.

Circular fit code taken from the most recent version of conan -
conan-ML_cv1.1/modules/select_regions.py"""

# Circular fit from the most recent version of conan - conan-ML_v1.1/modules/select_regions.py
from opendrop_ml.modules.preprocessing.preprocessing import preprocess

# from scipy.spatial import distance
# from scipy.integrate import solve_ivp
import scipy.optimize as opt

# import numba
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
import cv2
import math
import time


def intersection(center, radius, p1, p2, display=False):
    """find the two points where a secant intersects a circle"""
    x1, y1 = p1
    x2, y2 = p2

    dx, dy = x2 - x1, y2 - y1

    A = dx**2 + dy**2
    B = 2 * (dx * (x1 - center[0]) + dy * (y1 - center[1]))
    C = (x1 - center[0]) ** 2 + (y1 - center[1]) ** 2 - radius**2

    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        # No intersection points
        if display:
            print("Not a secant!")
        return None
    elif discriminant == 0:
        # One intersection point
        t = -B / (2 * A)
        x = x1 + t * dx
        y = y1 + t * dy
        return ((x, y), (x, y))  # np.array([[x, y],[x, y]])
    else:
        # Two intersection points
        t1 = (-B + np.sqrt(discriminant)) / (2 * A)
        t2 = (-B - np.sqrt(discriminant)) / (2 * A)
        x_int_1 = x1 + t1 * dx
        y_int_1 = y1 + t1 * dy
        x_int_2 = x1 + t2 * dx
        y_int_2 = y1 + t2 * dy
        # np.array([[x1, y1], [x2, y2]])
        return ((x_int_1, y_int_1), (x_int_2, y_int_2))


def circle_closest_point(xp, yp, xc, yc, r, n=1000, display=False):
    """
    xp (float): The x-coordinate of the reference point
    yp (float): The y-coordinate of the reference point
    xc (float): The x-coordinate of the circles's center.
    yc (float): The y-coordinate of the circle's center.
    r (float): The radius of the circle.
    n (int): The number of discrete points used to draw the circle.
    display (Boolean): Set to True to output figures and information.

    Returns:
        The distance between the reference point and the circle edge, and
        the coordinates of the closest point on the circle edge.

    """

    t = np.linspace(0, 2 * np.pi, n)

    x = xc + r * np.cos(t)
    y = yc + r * np.sin(t)

    dist = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
    idx = list(dist).index(min(dist))

    # ddistdt = ((b ** 2 - a ** 2) * np.cos(t) + a * np.sin(np.deg2rad(th)) * yp - a * np.sin(np.deg2rad(th)) * yc + a * np.cos(np.deg2rad(th)) * xp - a * np.cos(np.deg2rad(th)) * xc) * np.sin(t) + ((-b * np.cos(np.deg2rad(th)) * yp) + b * np.cos(np.deg2rad(th)) * yc + b * np.sin(np.deg2rad(th)) * xp - b * np.sin(np.deg2rad(th)) * xc) * np.cos(t)
    # idx = np.where(ddistdt[1:] * ddistdt[:-1] < 0)[0]  # find zeros
    # m = (ddistdt[idx + 1] - ddistdt[idx]) / (t[idx + 1] - t[idx])  # slope
    if display:
        plt.figure(1)
        plt.plot(x, y, "-", xp, yp, "r+", x[idx], y[idx], "r+")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Circle, Point, and Zeros")

        plt.figure(2)
        plt.plot(t, dist, "m.", t[idx], dist[idx], "cx")
        plt.xlabel("t")
        plt.ylabel("Distance")
        plt.title("Distance Function")

        print(f"xp: {xp}, x[idx]: {x[idx]}")
        print(f"yp: {yp}, y[idx]: {y[idx]}")
        print("Error is: ", dist[idx])

        plt.show()
        plt.close()

    return dist[idx], [x[idx], y[idx]]


def circle_fit_errors(contour, h, k, r, display=False):
    """
    Calculates the minimum distance between a point and the edge of a translated circle.

    Parameters:
        contour (array): The array of x, y coordindate points
        h (float): The x-coordinate of the circle's center.
        k (float): The y-coordinate of the circle's center.
        r (float): The radius of the circle.
        display (boolean): Set to true to show figures.

    Returns:
        dictionary: The MAE, MSE, RMSE, and maximum error of the contour as compared against the
        fitted circle.
    """

    errors = []

    for point in contour:
        dist2edge, edge_point = circle_closest_point(
            point[0], point[1], h, k, r, display=display
        )
        errors.append(dist2edge)

    error_measures = {}

    error_measures["MAE"] = sum([abs(error) for error in errors]) / len(errors)
    error_measures["MSE"] = sum([error**2 for error in errors]) / len(errors)
    error_measures["RMSE"] = np.sqrt(sum([error**2 for error in errors]) / len(errors))
    error_measures["Maximum error"] = max(errors)

    return error_measures


def circular_fit_img(img, display=False):
    """Call this function to perform the circular fit.
    For best results, peprocessing must be done before calling this function.
    """
    # begin with method specific preprocessing of img data
    start_time = time.time()

    img_processed, CPs_chosen, longest_contour = preprocess(img, display=display)
    idx_l = np.where((longest_contour == CPs_chosen[0]).all(axis=1))[0][0]
    idx_r = np.where((longest_contour == CPs_chosen[1]).all(axis=1))[0][0]
    if idx_l > idx_r:
        idx_l, idx_r = idx_r, idx_l
    drop = longest_contour[idx_l : idx_r + 1]
    CPs = {0: CPs_chosen[0], 1: CPs_chosen[1]}

    # of form [first y value of baseline, gradient]
    a = [CPs[0][1], (CPs[1][1] - CPs[0][1]) / (CPs[1][0] - CPs[0][0])]

    # define baseline as between the two contact points

    x = drop[:, 0]
    y = drop[:, 1]
    rise = CPs[1][1] - CPs[0][1]
    run = CPs[1][0] - CPs[0][0]
    slope = rise / run
    c = CPs[0][1] - (slope * CPs[0][0])
    baseline_x = np.linspace(1, img.shape[1], 100)
    baseline_y = slope * baseline_x + c

    # timers
    fit_preprocessing_time = time.time() - start_time
    fit_start_time = time.time()

    # Center estimates
    # x estimate is where between the lowest and highest points of the top section for a hydrophobic drop
    x_m = min(x) + (max(x) - min(x)) / 2
    # for full contour, y estimate is the halfway between max y and min y
    y_m = min(y) + ((max(y) - min(y)) / 2)

    # perform circular fit
    def calc_R(xc, yc):
        """calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = opt.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    # Ri_2       = calc_R(*center_2)
    Ri_2 = np.sqrt((x - xc_2) ** 2 + (y - yc_2) ** 2)
    R_2 = Ri_2.mean()

    if 1:  # round circle parameters so that it works for pixelated points
        center_2 = [round(xc_2), round(yc_2)]
        R_2 = round(R_2)

    if display:
        print("center: ", center_2)
        print("radius: ", R_2)
        print("contact points: ", CPs)

    outputs = intersection(center_2, R_2, CPs[0], CPs[1])
    if outputs is None:
        mid_x = (CPs[0][0] + CPs[1][0]) / 2
        mid_y = (CPs[0][1] + CPs[1][1]) / 2
        CPs_midpoint = np.array([mid_x, mid_y])
        gap, closest = circle_closest_point(
            CPs_midpoint[0],
            CPs_midpoint[1],
            center_2[0],
            center_2[1],
            R_2,
            display=False,
        )
        if gap < 2:
            left = closest
            right = closest
            outputs = [[left[0], left[1]], [right[0], right[1]]]
        else:
            fit_preprocessing_time = time.time() - start_time
            fit_time = time.time() - fit_start_time
            analysis_time = time.time() - start_time
            timings = {}
            timings["method specific preprocessing time"] = fit_preprocessing_time
            timings["fit time"] = fit_time
            timings["analysis time"] = analysis_time
            errors = "circular_fit_img: no line-circle intersection and fallback gap too large"
            return [], center_2, R_2, [], errors, timings
    a, b = outputs
    intercepts = [a, b]
    intercepts = sorted(list(intercepts), key=lambda x: x[0])

    # print('baseline y value: ',baseline[0][1])
    # print('a,b: ',a,b)

    CA = []
    for CP_2 in intercepts:
        first = True
        CPx_2 = CP_2[0]
        CPy_2 = CP_2[1]
        # vector from midpoint to intersect
        Vx = CPx_2 - xc_2
        Vy = CPy_2 - yc_2

        # slope of vector
        mV = Vy / Vx
        # print(mV)

        # normal vector slope
        norm = -1 / mV
        # print('norm is: ',norm)
        m = norm - slope
        # print('true norm is: ',norm)

        if CP_2[1] > center_2[1] and first == True:  # high CA angle left
            # CA.append(math.degrees(np.pi+abs(np.arctan(m))))
            CA.append(180 - abs(math.degrees(np.arctan(m))))
        elif CP_2[1] > center_2[1] and first == False:  # high CA angle right
            # CA.append(math.degrees(abs(np.arctan(m))))
            CA.append(abs(math.degrees(np.arctan(m))))
        elif CP_2[1] < center_2[1] and first == True:  # low CA angle left
            CA.append(abs(np.rad2deg(np.arctan(m))))
        elif CP_2[1] < center_2[1] and first == False:  # low CA angle right
            CA.append(abs(np.rad2deg(np.pi + np.arctan(m))))
        else:
            CA.append(90)

        first = False

    fit_time = time.time() - fit_start_time

    errors = circle_fit_errors(drop, center_2[0], center_2[1], R_2)

    if display:  # show fitted circle
        circle1 = plt.Circle((center_2[0], center_2[1]), 2, color="r")
        # now make a circle with no fill, which is good for hi-lighting key results
        circle2 = plt.Circle((center_2[0], center_2[1]), R_2, color="r", fill=False)

        ax = plt.gca()
        ax.axis("equal")

        ax.add_patch(circle1)
        ax.add_patch(circle2)

        fig = plt.gcf()
        fig.set_size_inches(10, 10)

        plt.imshow(img)
        plt.title("fitted circle \nRMSE of " + str(errors["RMSE"]))
        plt.plot(baseline_x, baseline_y, "r")
        plt.plot(CPs[0][0], CPs[0][1], "o", color="green", label="Left CP")
        plt.plot(CPs[1][0], CPs[1][1], "o", color="orange", label="Right CP")
        plt.plot(
            intercepts[0][0], intercepts[0][1], "o", color="magenta", label="Intercepts"
        )
        plt.plot(intercepts[1][0], intercepts[1][1], "o", color="magenta")
        plt.legend()
        plt.show()
        plt.close()

    analysis_time = time.time() - start_time

    timings = {}
    timings["method specific preprocessing time"] = fit_preprocessing_time
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return CA, center_2, R_2, intercepts, errors, timings


def circular_fit(drop, display=False):
    """Call this function to perform the circular fit.
    For best results, peprocessing must be done before calling this function.
    """
    # begin with method specific preprocessing of img data
    start_time = time.time()
    CPs = [drop[0], drop[-1]]
    # of form [first y value of baseline, gradient]
    a = [CPs[0][1], (CPs[1][1] - CPs[0][1]) / (CPs[1][0] - CPs[0][0])]

    # define baseline as between the two contact points

    x, y = drop[:, 0], drop[:, 1]
    rise = CPs[1][1] - CPs[0][1]
    run = CPs[1][0] - CPs[0][0]
    slope = rise / run
    c = CPs[0][1] - (slope * CPs[0][0])
    baseline_x = np.linspace(1, max(drop[:, 0]), 100)
    baseline_y = slope * baseline_x + c

    # Center estimates
    # x estimate is where between the lowest and highest points of the top section for a hydrophobic drop
    x_m = min(x) + (max(x) - min(x)) / 2
    # for full contour, y estimate is the halfway between max y and min y
    y_m = min(y) + ((max(y) - min(y)) / 2)

    # perform circular fit
    def calc_R(xc, yc):
        """calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = opt.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    # Ri_2       = calc_R(*center_2)
    Ri_2 = np.sqrt((x - xc_2) ** 2 + (y - yc_2) ** 2)
    R_2 = Ri_2.mean()

    if 1:  # round circle parameters so that it works for pixelated points
        center_2 = [round(xc_2), round(yc_2)]
        R_2 = round(R_2)

    if display:
        print("center: ", center_2)
        print("radius: ", R_2)
        print("contact points: ", CPs)

    outputs = intersection(center_2, R_2, CPs[0], CPs[1])
    if outputs is None:
        mid_x = (CPs[0][0] + CPs[1][0]) / 2
        mid_y = (CPs[0][1] + CPs[1][1]) / 2
        CPs_midpoint = np.array([mid_x, mid_y])
        gap, closest = circle_closest_point(
            CPs_midpoint[0],
            CPs_midpoint[1],
            center_2[0],
            center_2[1],
            R_2,
            display=False,
        )
        if display:
            print("gap between CPs_midpoint and closest point of circle: ", gap)
        if gap < 2:  # in case circle fit just misses intercept
            if display:
                print("just missed")
            left = closest
            right = closest
            outputs = [[left[0], left[1]], [right[0], right[1]]]
        else:
            fit_time = time.time() - start_time
            analysis_time = time.time() - start_time
            timings = {}
            timings["fit time"] = fit_time
            timings["analysis time"] = analysis_time
            errors = (
                "circular_fit: no line-circle intersection and fallback gap too large"
            )
            return [], center_2, R_2, [], errors, timings
    a, b = outputs
    intercepts = [a, b]
    intercepts = sorted(list(intercepts), key=lambda x: x[0])

    # print('baseline y value: ',baseline[0][1])
    # print('intercepts: ',intercepts)

    CA = []
    for CP_2 in intercepts:
        first = True
        CPx_2 = CP_2[0]
        CPy_2 = CP_2[1]
        # vector from midpoint to intersect
        Vx = CPx_2 - xc_2
        Vy = CPy_2 - yc_2

        # slope of vector
        mV = Vy / Vx
        # print(mV)

        # normal vector slope
        norm = -1 / mV
        # print('norm is: ',norm)
        m = norm - slope
        # print('true norm is: ',norm)
        # print('m: ', m)

        if CP_2[1] > center_2[1] and first == True:  # high CA angle left
            # CA.append(math.degrees(np.pi+abs(np.arctan(m))))
            CA.append(180 - abs(math.degrees(np.arctan(m))))
        elif CP_2[1] > center_2[1] and first == False:  # high CA angle right
            # CA.append(math.degrees(abs(np.arctan(m))))
            CA.append(abs(math.degrees(np.arctan(m))))
        elif CP_2[1] < center_2[1] and first == True:  # low CA angle left
            CA.append(abs(np.rad2deg(np.arctan(m))))
        elif CP_2[1] < center_2[1] and first == False:  # low CA angle right
            CA.append(abs(np.rad2deg(np.pi + np.arctan(m))))
        else:
            CA.append(90)

        first = False

    fit_time = time.time() - start_time

    errors = circle_fit_errors(drop, center_2[0], center_2[1], R_2)

    if display:  # show fitted circle
        circle1 = plt.Circle((center_2[0], center_2[1]), 2, color="r")
        # now make a circle with no fill, which is good for hi-lighting key results
        circle2 = plt.Circle((center_2[0], center_2[1]), R_2, color="r", fill=False)

        ax = plt.gca()
        ax.axis("equal")

        ax.add_patch(circle1)
        ax.add_patch(circle2)

        fig = plt.gcf()
        fig.set_size_inches(10, 10)

        plt.title("fitted circle \nRMSE of " + str(errors["RMSE"]))
        plt.plot(baseline_x, baseline_y, "r")
        plt.plot(CPs[0][0], CPs[0][1], "o", color="green", label="Left CP")
        plt.plot(CPs[1][0], CPs[1][1], "o", color="orange", label="Right CP")
        plt.plot(
            intercepts[0][0], intercepts[0][1], "o", color="magenta", label="Intercepts"
        )
        plt.plot(intercepts[1][0], intercepts[1][1], "o", color="magenta")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()

    analysis_time = time.time() - start_time

    timings = {}
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return CA, center_2, R_2, intercepts, errors, timings


if 0:
    from opendrop_ml.utils.os import resource_path

    IMG_PATH = resource_path("../RICOphobic_cropped.png")
    img = cv2.imread(IMG_PATH)
    angles, centre, R_2, intercepts, errors, timings = circular_fit_img(
        img, display=False
    )

    if 1:
        print("angles: ", angles)
        print("circle centre: ", centre)
        print("circle radius: ", R_2)
        print("intercept points: ", intercepts)
        print("errors: ", errors)
        print("timings: ", timings)

    print("done")
