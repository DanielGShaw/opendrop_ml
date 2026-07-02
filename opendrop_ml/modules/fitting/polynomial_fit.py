#!/usr/bin/env python
# coding=utf-8

"""This code serves as a discrete instance of the polynomial fit method of
contact angle analysis.

Polynomial fit code taken from the most recent version of conan -
conan-ML_cv1.1/modules/select_regions.py"""

# Polynomial fit from the most recent version of conan - conan-ML_v1.1/modules/select_regions.py

from opendrop_ml.modules.preprocessing.preprocessing import preprocess

# from scipy.spatial import distance
# from scipy.integrate import solve_ivp
# import scipy.optimize as opt
# import numba
# import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def polynomial_closest_point(xp, yp, poly_points, display=False):
    """
    xp (float): The x-coordinate of the reference point
    yp (float): The y-coordinate of the reference point
    poly_points (array): The array of x, y coordinates outputted by the polynomial fit
    display (Boolean): Set to True to output figures and information.

    Returns:
        The distance between the reference point and the polynomial fit, and
        the coordinates of the closest point on the polynomial fit.

    """

    x = poly_points[:, 0]
    y = poly_points[:, 1]

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
        for t, d in enumerate(dist):
            plt.plot(t, d, "m.")
        plt.plot(idx, dist[idx], "cx")
        plt.xlabel("index value of list")
        plt.ylabel("Distance")
        plt.title("Distance Function")

        print(f"xp: {xp}, x[idx]: {x[idx]}")
        print(f"yp: {yp}, y[idx]: {y[idx]}")
        print("Error is: ", dist[idx])

        plt.show()
        plt.close()

    return dist[idx], [x[idx], y[idx]]


def points_to_angle(x1, y1, x2, y2):
    """Calculate the angle in degrees between two points in a 2D plane."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    return np.degrees(np.arctan2(dy, dx))


def gradient_to_contact_angles(gradient):
    """Convert a local tangent gradient to (left, right) contact angles."""
    if np.isinf(gradient):
        return 90.0, 90.0

    gradient = -gradient
    base_angle_deg = np.degrees(np.arctan(gradient))
    if base_angle_deg < 0:
        base_angle_deg += 180

    left_angle = base_angle_deg % 180
    right_angle = (180 - left_angle) % 180
    return left_angle, right_angle


def polynomial_fit_errors(pts1, pts2, fit_left, fit_right, display=False):
    """
    Calculates the minimum distance between a point and a point of the polynomial fit.

    Parameters:
        pts1 (array): The array of x, y coordindate points for the points near the left contact point
        pts2 (array): The array of x, y coordindate points for the points near the right contact point
        fit_left (np.polyfit): The fit found by np.polyfit for pts1
        fit_right (np.polyfit): The fit found by np.polyfit for pts2
        display (boolean): Set to true to show figures.

    Returns:
        dictionary: The MAE, MSE, RMSE, and maximum error of the contour as compared against the
        polynomial fit.
    """
    highresx_left = np.linspace(pts1[0, 0], pts1[-1, 0], 5 * pts1.shape[0])
    highresx_right = np.linspace(pts2[0, 0], pts2[-1, 0], 5 * pts2.shape[0])
    highres_fit_left = fit_left(highresx_left)
    highres_fit_right = fit_right(highresx_right)

    poly_points_left = np.array(list(zip(highresx_left, highres_fit_left)))
    poly_points_right = np.array(list(zip(highresx_right, highres_fit_right)))

    error_measures = {}

    # for left
    errors_left = []
    for point in pts1:
        dist2edge, edge_point = polynomial_closest_point(
            point[0], point[1], poly_points_left, display=display
        )
        errors_left.append(dist2edge)

    error_measures["MAE left"] = sum([abs(error) for error in errors_left]) / len(
        errors_left
    )
    error_measures["MSE left"] = sum([error**2 for error in errors_left]) / len(
        errors_left
    )
    error_measures["RMSE left"] = np.sqrt(
        sum([error**2 for error in errors_left]) / len(errors_left)
    )
    error_measures["Maximum error left"] = max(errors_left)

    # for right
    errors_right = []
    for point in pts2:
        dist2edge, edge_point = polynomial_closest_point(
            point[0], point[1], poly_points_right, display=display
        )
        errors_right.append(dist2edge)

    error_measures["MAE right"] = sum([abs(error) for error in errors_right]) / len(
        errors_right
    )
    error_measures["MSE right"] = sum([error**2 for error in errors_right]) / len(
        errors_right
    )
    error_measures["RMSE right"] = np.sqrt(
        sum([error**2 for error in errors_right]) / len(errors_right)
    )
    error_measures["Maximum error right"] = max(errors_right)

    error_measures["MAE"] = (
        sum([error_measures["MAE left"], error_measures["MAE right"]]) / 2
    )
    error_measures["MSE"] = (
        sum([error_measures["MSE left"], error_measures["MSE right"]]) / 2
    )
    error_measures["RMSE"] = (
        sum([error_measures["RMSE left"], error_measures["RMSE right"]]) / 2
    )
    error_measures["Maximum error"] = max(
        [error_measures["Maximum error left"], error_measures["Maximum error right"]]
    )

    return error_measures


def polynomial_fit_img(img, num_points=15, polynomial_degree=2, display=False):
    """Takes in input contact angle experimental image, and outputs the fitted
    angles and intercept coordinates. For best results, preprocessing should be
    performed before calling this function.

    The first and last num_points many points are are fitted to a polynomial, and contact
    angles are calculated by taking the gradient of the polynomial at the contact
    points. The tilt of the surface is accounted for by taking the gradient between
    contact points and balancing the polynomial angle against the surface gradient.

    Coords is the coordinates of the drop outline in the form of an array or list.
    num_points is the number of points included in the polynomial fit. The default number
        of points used is 10.
    display can be set to "True" to output figures."""

    # begin with method specific preprocessing of img data
    start_time = time.time()

    img_processed, CPs_chosen, longest_contour = preprocess(img, display=display)
    idx_l = np.where((longest_contour == CPs_chosen[0]).all(axis=1))[0][0]
    idx_r = np.where((longest_contour == CPs_chosen[1]).all(axis=1))[0][0]
    if idx_l > idx_r:
        idx_l, idx_r = idx_r, idx_l
    tangent_drop = longest_contour[idx_l : idx_r + 1]
    CPs = {0: CPs_chosen[0], 1: CPs_chosen[1]}

    intercepts = [
        [tangent_drop[0, 0], tangent_drop[0, 1]],
        [tangent_drop[-1, 0], tangent_drop[-1, 1]],
    ]

    # timers
    fit_preprocessing_time = time.time() - start_time
    fit_start_time = time.time()

    # start the fit
    Ndrop = np.shape(tangent_drop)[0]
    pts1 = tangent_drop[:num_points]
    pts2 = tangent_drop[-num_points:]
    pts2 = pts2[::-1]  # reverse so that CP is first

    fit_local1 = np.polyfit(pts1[:, 0], pts1[:, 1], polynomial_degree)
    fit_local2 = np.polyfit(pts2[:, 0], pts2[:, 1], polynomial_degree)

    line_local1 = np.poly1d(fit_local1)
    line_local2 = np.poly1d(fit_local2)

    if polynomial_degree > 1:
        x_local1 = np.array([min(pts1[:, 0]), max(pts1[:, 0])])
        f_local1 = line_local1(x_local1)
        f_local1_prime = line_local1.deriv(1)

        x_local2 = np.array([min(pts2[:, 0]) - 10, max(pts2[:, 0]) + 10])
        f_local2 = line_local2(x_local2)
        f_local2_prime = line_local2.deriv(1)

        tangent1 = f_local1_prime(pts1[0, 0]) * (x_local1 - pts1[0, 0]) + pts1[0, 1]
        tangent2 = f_local2_prime(pts2[0, 0]) * (x_local2 - pts2[0, 0]) + pts2[0, 1]

        m1 = f_local1_prime(pts1[0, 0])
        m2 = f_local2_prime(pts2[0, 0])
    else:
        m1 = fit_local1[0]
        m2 = fit_local2[0]

    if display:
        plt.plot(longest_contour[:, 0], longest_contour[:, 1], "o", color="pink")
        plt.plot(pts1[:, 0], pts1[:, 1], "ro")
        plt.plot(pts1[:, 0], line_local1(pts1[:, 0]), "y-")
        plt.imshow(img)
        pts1_width = abs(pts1[:, 0][-1] - pts1[:, 0][0])
        pts1_height = abs(pts1[:, 1][-1] - pts1[:, 1][0])
        plt.xlim(min(pts1[:, 0]) - pts1_width, max(pts1[:, 0]) + pts1_width)
        plt.ylim(max(pts1[:, 1]) + pts1_height, min(pts1[:, 1]) - pts1_height)
        plt.title("fitted points left side")
        plt.show()
        plt.close()

        plt.plot(longest_contour[:, 0], longest_contour[:, 1], "o", color="pink")
        plt.plot(pts2[:, 0], pts2[:, 1], "ro")
        plt.plot(pts2[:, 0], line_local2(pts2[:, 0]), "y-")
        plt.imshow(img)
        pts2_width = abs(pts2[:, 0][-1] - pts2[:, 0][0])
        pts2_height = abs(pts2[:, 1][-1] - pts2[:, 1][0])
        plt.xlim(min(pts2[:, 0]) - pts2_width, max(pts2[:, 0]) + pts2_width)
        plt.ylim(max(pts2[:, 1]) + pts2_height, min(pts2[:, 1]) - pts2_height)
        plt.title("fitted points right side")
        plt.show()
        plt.close()

    surf_angle = points_to_angle(CPs[0][0], CPs[0][1], CPs[1][0], CPs[1][1])
    contact_angle1, _ = gradient_to_contact_angles(m1)
    _, contact_angle2 = gradient_to_contact_angles(m2)

    contact_angle1 += surf_angle
    contact_angle2 -= surf_angle

    if contact_angle1 < 1:
        widest = min(tangent_drop[:, 0])
        if widest < CPs[0][0]:
            contact_angle1 = abs(180 - contact_angle1)
    if contact_angle2 < 1:
        widest = max(tangent_drop[:, 0])
        if widest > CPs[1][0]:
            contact_angle2 = abs(180 - contact_angle2)

    fit_time = time.time() - fit_start_time

    errors = polynomial_fit_errors(pts1, pts2, line_local1, line_local2, False)

    analysis_time = time.time() - start_time

    timings = {}
    timings["method specific preprocessing time"] = fit_preprocessing_time
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return [contact_angle1, contact_angle2], intercepts, errors, timings


def polynomial_fit(profile, num_points=None, polynomial_degree=2, display=False):
    """Takes in input contact angle experimental image, and outputs the fitted
    angles and intercept coordinates. For best results, preprocessing should be
    performed before calling this function.

    The first and last num_points many points are are fitted to a polynomial, and contact
    angles are calculated by taking the gradient of the polynomial at the contact
    points. The tilt of the surface is accounted for by taking the gradient between
    contact points and balancing the polynomial angle against the surface gradient.

    Coords is the coordinates of the drop outline in the form of an array or list.
    num_points is the number of points included in the polynomial fit. The default number
        of points used is 10.
    display can be set to "True" to output figures."""

    if num_points is None:
        num_points = int(len(profile) * 0.05)
        if num_points < 10:
            num_points = 10

    start_time = time.time()
    CPs = [profile[0], profile[-1]]

    # start the fit
    tangent_drop = profile.copy()
    Ndrop = np.shape(tangent_drop)[0]
    pts1 = tangent_drop[:num_points]
    pts2 = tangent_drop[-num_points:]
    pts2 = pts2[::-1]  # reverse so that CP is first

    fit_local1 = np.polyfit(pts1[:, 0], pts1[:, 1], polynomial_degree)
    fit_local2 = np.polyfit(pts2[:, 0], pts2[:, 1], polynomial_degree)

    line_local1 = np.poly1d(fit_local1)
    line_local2 = np.poly1d(fit_local2)

    if polynomial_degree > 1:
        # x_local1 = np.array([min(pts1[:,0])-10,max(pts1[:,0])+10])
        x_local1 = np.array([min(pts1[:, 0]), max(pts1[:, 0])])
        f_local1 = line_local1(x_local1)
        f_local1_prime = line_local1.deriv(1)

        # x_local2 = np.array([min(pts2[:,0])-10,max(pts2[:,0])+10])
        x_local2 = np.array([min(pts2[:, 0]), max(pts2[:, 0])])
        f_local2 = line_local2(x_local2)
        f_local2_prime = line_local2.deriv(1)

        tangent1 = f_local1_prime(pts1[0, 0]) * (x_local1 - pts1[0, 0]) + pts1[0, 1]
        tangent2 = f_local2_prime(pts2[0, 0]) * (x_local2 - pts2[0, 0]) + pts2[0, 1]

        tangent_lines = (
            (
                (int(x_local1[0]), int(tangent1[0])),
                (int(x_local1[1]), int(tangent1[1])),
            ),
            (
                (int(x_local2[0]), int(tangent2[0])),
                (int(x_local2[1]), int(tangent2[1])),
            ),
        )

        m1 = f_local1_prime(pts1[0, 0])
        m2 = f_local2_prime(pts2[0, 0])
    else:
        m1 = fit_local1[0]
        m2 = fit_local2[0]

        tangent_lines = (
            (
                tuple(pts1[0]),
                (int(pts1[-1][0]), int((m1 * pts1[-1][0]) + fit_local1[1])),
            ),
            (
                tuple(pts2[0]),
                (int(pts2[-1][0]), int((m2 * pts2[-1][0]) + fit_local2[1])),
            ),
        )

    if display:
        plt.plot(profile[:, 0], profile[:, 1], "o", color="pink")
        plt.plot(pts1[:, 0], pts1[:, 1], "ro")
        plt.plot(pts1[:, 0], line_local1(pts1[:, 0]), "y-")
        # plt.imshow(img)
        pts1_width = abs(pts1[:, 0][-1] - pts1[:, 0][0])
        pts1_height = abs(pts1[:, 1][-1] - pts1[:, 1][0])
        plt.xlim(min(pts1[:, 0]) - pts1_width, max(pts1[:, 0]) + pts1_width)
        plt.ylim(max(pts1[:, 1]) + pts1_height, min(pts1[:, 1]) - pts1_height)
        plt.title("fitted points left side")
        plt.show()
        plt.close()

        plt.plot(profile[:, 0], profile[:, 1], "o", color="pink")
        plt.plot(pts2[:, 0], pts2[:, 1], "ro")
        plt.plot(pts2[:, 0], line_local2(pts2[:, 0]), "y-")
        # lt.imshow(img)
        pts2_width = abs(pts2[:, 0][-1] - pts2[:, 0][0])
        pts2_height = abs(pts2[:, 1][-1] - pts2[:, 1][0])
        plt.xlim(min(pts2[:, 0]) - pts2_width, max(pts2[:, 0]) + pts2_width)
        plt.ylim(max(pts2[:, 1]) + pts2_height, min(pts2[:, 1]) - pts2_height)
        plt.title("fitted points right side")
        plt.show()
        plt.close()

    surf_angle = points_to_angle(CPs[0][0], CPs[0][1], CPs[1][0], CPs[1][1])
    contact_angle1, _ = gradient_to_contact_angles(m1)
    _, contact_angle2 = gradient_to_contact_angles(m2)

    contact_angle1 += surf_angle
    contact_angle2 -= surf_angle

    if contact_angle1 < 1:
        widest = min(profile[:, 0])
        if widest < CPs[0][0]:
            contact_angle1 = abs(180 - contact_angle1)
    if contact_angle2 < 1:
        widest = max(profile[:, 0])
        if widest > CPs[1][0]:
            contact_angle2 = abs(180 - contact_angle2)

    fit_time = time.time() - start_time

    errors = polynomial_fit_errors(pts1, pts2, line_local1, line_local2, False)

    analysis_time = time.time() - start_time

    timings = {}
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return [contact_angle1, contact_angle2], CPs, tangent_lines, errors, timings


if 0:
    from opendrop_ml.utils.os import resource_path

    IMG_PATH = resource_path("../../RICOphobic_cropped.png")
    img = cv2.imread(IMG_PATH)

    angles, intercepts, errors, timings = polynomial_fit_img(img, display=False)

    if 1:
        print("angles: ", angles)
        print("intercetps: ", intercepts)
        print("errors: ", errors)
        print("timings: ", timings)

    print("done")
