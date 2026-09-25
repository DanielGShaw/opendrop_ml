#!/usr/bin/env python
# coding=utf-8

"""This code serves as a discrete instance of image preprocessing before contact
angle fit software is implemented

This includes automatic identification of the drop through Hough transform,
followed by cropping of the image to isolate the drop. Tilt correction is then
performed using the identified contact points of the drop.
"""

from sklearn.cluster import OPTICS  # for clustering algorithm
import numpy as np
import matplotlib.pyplot as plt
import cv2  # for image reading, contour detection
import math  # for tilt_correction
from scipy import ndimage  # for tilt_correction
from scipy import signal  # for CP ID
from scipy.odr import Model, Data, ODR  # for linear fit

# Diagnostic-only hook for isolate_contour(), used to trace intermediate
# values during cross-platform debugging (set by
# paper_results/evaluate/diagnose_contour_platform.py). None (the default) is
# zero-overhead -- production code never sets this.
_ISOLATE_CONTOUR_DEBUG_HOOK = None


def _debug_emit(event, **kwargs):
    if _ISOLATE_CONTOUR_DEBUG_HOOK is not None:
        _ISOLATE_CONTOUR_DEBUG_HOOK(event, kwargs)


def auto_crop(img, low=50, high=150, apertureSize=3, verbose=0):  # DS 08/06/23
    """
    Automatically identify where the crop should be placed within the original
    image

    This function utilizes the opencv circular and linear Hough transfrom
    implementations to identify the most circular object in the image
    (the droplet), and center it within a frame that extends by padding to each
    side.

    :param img: 2D numpy array of [x,y] coordinates of the edges of the
                  image
    :param low: Value of the weak pixels in the dual thresholding
    :param high: Value of the strong pixels in the dual thresholding
    :param apertureSize: The aperture size variable given to cv2.Canny during
                        edge detection
    :param verbose: Integer values from 0 to 2, giving varying degrees of detail
    :return: list of [left, right, top, bottom] values for the edges of the
             bounding box
    """

    if verbose >= 1:
        print("Performing auto-cropping, please wait...")

    # find edges in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high, apertureSize=apertureSize)

    # hough circle to find droplet - minRadius at 5% img width
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=max(img.shape),  # one circle
        param1=30,
        param2=10,
        minRadius=int(img.shape[1] * 0.02),
        maxRadius=0,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            if verbose >= 2:
                circle1 = plt.Circle(center, 1, color="r")
                # now make a circle with no fill, which is good for hi-lighting key results
                circle2 = plt.Circle(center, radius, color="r", fill=False)

                ax = plt.gca()
                ax.axis("equal")

                ax.add_patch(circle1)
                ax.add_patch(circle2)

                fig = plt.gcf()
                fig.set_size_inches(10, 10)

                plt.imshow(img)
                plt.title("Hough circle")
                plt.show()
                plt.close()
    else:
        # hough circle to find droplet - minRadius at 0
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            1,
            minDist=max(img.shape),  # one circle
            param1=30,
            param2=20,
            minRadius=0,
            maxRadius=0,
        )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            if verbose >= 2:
                circle1 = plt.Circle(center, 1, color="r")
                # now make a circle with no fill, which is good for hi-lighting key results
                circle2 = plt.Circle(center, radius, color="r", fill=False)

                ax = plt.gca()
                ax.axis("equal")

                ax.add_patch(circle1)
                ax.add_patch(circle2)

                fig = plt.gcf()
                fig.set_size_inches(10, 10)

                plt.imshow(img)
                plt.title("Hough circle")
                plt.show()
                plt.close()
    else:
        print("Hough circle failed to identify a drop")

    # crop image based on circle found (this prevents hough line identifying the needle)
    bottom = int(center[1] + (radius * 1.2))  # add 20% padding
    if bottom > img.shape[0]:
        bottom = img.shape[0]
    top = int(center[1] - (radius * 1.2))  # add 20% padding
    if top < 0:
        top = 0

    img = img[top:bottom, :]
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    center = (
        center[0],
        -(center[1] - bottom),
    )  # reassign circle center to new cropped image

    if verbose >= 2:
        plt.imshow(img)
        plt.title("image after top and bottom crop")
        plt.show()
        plt.close()

    # hough lines to find baseline
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:  # if the HoughLines function is successful
        if verbose >= 2:
            print("shape of image: ", img.shape)
        for i, line in enumerate(lines):
            if i == 0:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0)
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - img.shape[1] * (-b))
                y2 = int(y0 - 1000 * (a))
                if verbose >= 2:
                    plt.title("hough approximated findings")
                    plt.imshow(img)
                    circle1 = plt.Circle(center, 1, color="r")
                    circle2 = plt.Circle(center, radius, color="r", fill=False)
                    ax = plt.gca()
                    ax.add_patch(circle1)
                    ax.add_patch(circle2)
                    fig = plt.gcf()

                    plt.plot([x1, x2], [y1, y2], "r")
                    fig = plt.gcf()
        p1, p2 = (x1, y1), (x2, y2)  # baseline exists between these points
        if verbose >= 2:
            print("baseline goes from ", p1, " to ", p2)

        # now find bounds

        # find intercept of line and circle
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        a = dx**2 + dy**2
        b = 2 * (dx * (p1[0] - center[0]) + dy * (p1[1] - center[1]))
        c = (p1[0] - center[0]) ** 2 + (p1[1] - center[1]) ** 2 - radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant > 0:
            t1 = (-b + discriminant**0.5) / (2 * a)
            t2 = (-b - discriminant**0.5) / (2 * a)

            intersect1, intersect2 = None, None
            intersect1, intersect2 = (dx * t1 + p1[0], dy * t1 + p1[1]), (
                dx * t2 + p1[0],
                dy * t2 + p1[1],
            )

            if intersect1 == None or intersect2 == None:
                raise "found baseline does not intersect with found circle"

            if verbose >= 2:
                plt.plot(intersect1[0], intersect1[1], "o", color="orange")
                plt.plot(intersect2[0], intersect2[1], "o", color="orange")

                plt.show()
                plt.close()

            bottom = int(
                max([intersect1[1], intersect2[1]])
            )  # max value of intersect points
            top = int(center[1] - radius)  # assume top of drop is in image
            if center[1] < max([intersect1[1], intersect2[1]]):
                right = int(center[0] + radius)
            else:
                right = int(max([intersect1[0], intersect2[0]]))
            if center[1] < min([intersect1[1], intersect2[1]]):
                left = int(center[0] - radius)
            else:
                left = int(min([intersect1[0], intersect2[0]]))
        else:
            # bounds is (left,right,top,bottom)
            print("No baseline-drop intercept found")
            left = int(center[0] - radius)
            right = int(center[0] + radius)
            top = int(center[1] - radius)
            bottom = int(center[1] + radius)

    else:  # if the HoughLine function cannot identify a surface line, use the circle as a guide for bounds
        left = int(center[0] - radius)
        right = int(center[0] + radius)
        top = int(center[1] - radius)
        bottom = int(center[1] + radius)

    pad = int(max([right - left, bottom - top]) / 4)

    top -= pad
    bottom += pad
    left -= pad
    right += pad

    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if bottom > img.shape[0]:
        bottom = img.shape[0]
    if right > img.shape[1]:
        right = img.shape[1]

    if verbose >= 2:
        print("lower most y coord of drop: ", bottom)
        print("upper most y coord of drop: ", top)
        print("right most x coord of drop: ", right)
        print("left most x coord of drop: ", left)

    bounds = [left, right, top, bottom]
    new_img = img[top:bottom, left:right]

    if verbose >= 1:
        plt.title("cropped drop")
        plt.imshow(new_img)
        plt.show()
        plt.close()

    return new_img, bounds


def cluster_OPTICS(sample, out_style="coords", xi=None, eps=None, verbose=0):
    """Takes an array (or list) of the form [[x1,y1],[x2,y2],...,[xn,yn]].
    Clusters are outputted in the form of a dictionary.

    If out_style='coords' each dictionary entry is a group, and points are outputted in as a numpy array
    in coordinate form.
    If out_xy='xy' there are two dictionary entries for each group, one labeled as nx and one as ny
    (where n is the label of the group). Each of these are 1D numpy arrays

    If xi (float between 0 and 1) is not None and eps is None, then the xi clustering method is used.
    The optics algorithm defines clusters based on the minimum steepness on the reachability plot.
    For example, an upwards point in the reachability plot is defined by the ratio from one point to
    its successor being at most 1-xi.

    If eps (float) is not None and xi is None, then the dbscan clustering method is used. Where eps is the
    maximum distance between two samples for one to be considered as in the neighborhood of the other.

    https://stackoverflow.com/questions/47974874/algorithm-for-grouping-points-in-given-distance
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
    """
    if eps != None and xi == None:
        clustering = OPTICS(min_samples=2, cluster_method="dbscan", eps=eps).fit(sample)
    elif xi != None and eps == None:
        clustering = OPTICS(min_samples=2, xi=xi).fit(sample)
    else:
        raise "Error: only one of eps and xi can be chosen but not neither nor both"
    groups = list(set(clustering.labels_))

    if verbose == 2:
        print(clustering.labels_)
    elif verbose == 1:
        print(groups)
    elif verbose == 0:
        pass

    dic = {}
    for n in groups:
        if n not in dic:
            dic[n] = []
        for i in range(len(sample)):
            if clustering.labels_[i] == n:
                dic[n].append(list(sample[i]))
        dic[n] = np.array(dic[n])

    # separate points and graph
    dic2 = {}
    for k in dic.keys():
        x = []
        y = []
        for i in range(len(dic[k])):
            x.append(dic[k][i][0])
        dic2[str(k) + "x"] = np.array(x)
        for i in range(len(dic[k])):
            y.append(dic[k][i][1])
        dic2[str(k) + "y"] = np.array(y)

    if out_style == "coords":
        return dic
    elif out_style == "xy":
        return dic2


def distance1(P1, P2):
    """This function computes the distance between 2 points defined by
    P1 = (x1,y1) and P2 = (x2,y2)"""
    return ((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2) ** 0.5


def optimized_path(coords, start=None):
    """This function finds the nearest point to a point
    coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...]
    https://stackoverflow.com/questions/45829155/sort-points-in-order-to-have-a-continuous-curve-using-python
    """
    if isinstance(coords, list) == False:
        coords = coords.tolist()
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance1(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    path = np.array(path)
    return path


def hough_baseline(edges, HoughLinesThreshold=40, vertical_threshold=10, display=False):
    """
    Detect the baseline in an image using the Hough Line Transform.

    Args:
        edges (numpy.ndarray): A binary edge image obtained from an edge detection algorithm like Canny.
        HoughLinesThreshold (int, optional): The threshold value for the Hough Line Transform. Defaults to 40.
        vertical_threshold (int, optional): The maximum allowed deviation from the horizontal (in degrees) for
            a line to be considered non-vertical. Defaults to 10.
        display (bool, optional): Whether to display the detected baseline on the image. Defaults to False.

    Returns:
        list: A list containing the start and end points of the detected baseline, represented as [x1, y1, x2, y2].

    This function performs the following steps:
    1. Applies the Hough Line Transform to the input edge image to detect lines.
    2. Filters out vertical lines based on the specified `vertical_threshold`.
    3. Orders the remaining lines from flattest to highest magnitude gradient.
    4. Selects the flattest line as the baseline.
    5. If `display` is True, it plots the detected baseline on the input image.

    Example:
    An appropriate input can be generated using the following, or something similar
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(otsu_thresh,3,5,apertureSize = 3)

    """

    lines = cv2.HoughLines(edges, 1, np.pi / 180, HoughLinesThreshold)

    # Filter out vertical lines
    filtered_lines = []
    if lines is not None:  # if the HoughLines function is successful
        for line in lines:
            rho, theta = line[0]
            if abs(theta * 180 / np.pi - 90) < vertical_threshold:
                filtered_lines.append(line)
    else:
        print("No lines found in image")

    # Take the first non-vertical line
    lines = []
    if filtered_lines:
        for line in filtered_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0)
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - edges.shape[1] * (-b))
            y2 = int(y0 - 1000 * (a))
            p1, p2 = [x1, y1], [x2, y2]
            lines.append([p1, p2])

    # to take the best line order the lines from flattest to highest magnitude gradient
    gradients = []
    for i, line in enumerate(lines):
        x1, y1 = line[0]
        x2, y2 = line[1]

        if x1 == x2:  # Vertical line
            gradient = float("inf")
        else:
            gradient = (y2 - y1) / (x2 - x1)
        gradients.append((gradient, line))
    sorted_gradients = sorted(gradients, key=lambda x: abs(x[0]))
    sorted_lines = np.array([line for _, line in sorted_gradients])

    if display:
        print("sorted_lines found by Hough Line: ", sorted_lines)
        plt.imshow(edges)
        for i, line in enumerate(sorted_lines):
            plt.plot(line[:, 0], line[:, 1])
        plt.show()
        plt.close()
    return sorted_lines[0]


def hough_circle(edges, display):
    # hough circle to find droplet
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=max(edges.shape),  # one circle
        param1=40,
        param2=14,
        minRadius=int(edges.shape[1] * 0.05),
        maxRadius=0,
    )

    # check if the top of the circle is within 10% of the top of the contour
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
    circle_top = center[1] - radius
    edges_top = np.min(np.where(edges == 255)[0])
    if circle_top < edges_top - edges.shape[0] * 0.1:
        circles = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

    else:
        # hough circle to find droplet - minRadius at 0
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            1,
            minDist=max(edges.shape),  # one circle
            param1=30,
            param2=20,
            minRadius=0,
            maxRadius=0,
        )

    # if circle goes past the top of the image then set a maxRadius
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

        if radius > center[1]:  # top of circle is out of bounds of image
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                1,
                minDist=max(edges.shape),  # one circle
                param1=40,
                param2=15,
                minRadius=int(edges.shape[1] * 0.10),
                maxRadius=int(edges.shape[1] * 0.40),
            )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            if display:
                circle1 = plt.Circle(center, 1, color="r")
                # now make a circle with no fill, which is good for hi-lighting key results
                circle2 = plt.Circle(center, radius, color="r", fill=False)

                ax = plt.gca()
                ax.axis("equal")

                ax.add_patch(circle1)
                ax.add_patch(circle2)

                fig = plt.gcf()
                fig.set_size_inches(10, 10)

                plt.imshow(edges)
                plt.title("Hough circle")
                plt.show()
                plt.close()

    return center, radius


def extract_edges_CV(
    img, threshold_val=None, return_thresholed_value=False, display=False, blur=True
):
    """
    give the image and return a list of [x.y] coordinates for the detected edges

    """
    IGNORE_EDGE_MARGIN = 1
    img = img.astype("uint8")
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img

    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # blur size of 3

    if threshold_val == None:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Each contour has shape (n, 1, 2) where 'n' is the number of points. Presumably this is so each
    # point is a size 2 column vector, we don't want this so reshape it to a (n, 2)
    contours = [contour.reshape(contour.shape[0], 2) for contour in contours]

    # Sort the contours by arc length, descending order
    contours.sort(key=lambda c: cv2.arcLength(c, False), reverse=True)

    drop_profile = contours[0]

    # Ignore points of the drop profile near the edges of the drop image
    width, height = img.shape[1::-1]
    if not (width < IGNORE_EDGE_MARGIN or height < IGNORE_EDGE_MARGIN):
        mask = (
            (IGNORE_EDGE_MARGIN < drop_profile[:, 0])
            & (drop_profile[:, 0] < width - IGNORE_EDGE_MARGIN)
            & (IGNORE_EDGE_MARGIN < drop_profile[:, 1])
            & (drop_profile[:, 1] < height - IGNORE_EDGE_MARGIN)
        )
        drop_profile = drop_profile[mask]

    output = []
    for coord in drop_profile:
        if list(coord) not in output:
            output.append(list(coord))
    output = np.array(output)

    if return_thresholed_value == True:
        return output, ret
    else:
        return output


def line_circle_intersection(line_point1, line_point2, circle_center, circle_radius):
    """
    Calculate the intersection points of a line and a circle.

    Args:
    line_point1 (tuple): (x, y) of first point on the line
    line_point2 (tuple): (x, y) of second point on the line
    circle_center (tuple): (x, y) of circle center
    circle_radius (float): radius of the circle

    Returns:
    list: List of intersection points (can be empty, have one point, or two points)
    """

    # Convert inputs to float64: callers pass Hough output as integer arrays,
    # and on Windows (numpy's default int is 32-bit) the integer arithmetic
    # below overflows -- b**2 alone is ~1e12 for a full-width baseline.
    p1 = np.asarray(line_point1, dtype=np.float64)
    p2 = np.asarray(line_point2, dtype=np.float64)
    c = np.asarray(circle_center, dtype=np.float64)
    circle_radius = float(circle_radius)

    # Calculate the direction vector of the line
    direction = p2 - p1

    # Calculate the vector from circle center to line start
    f = p1 - c

    # Calculate quadratic equation coefficients
    a = np.dot(direction, direction)
    b = 2 * np.dot(f, direction)
    c = np.dot(f, f) - circle_radius**2

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    # Check the number of solutions
    if discriminant < 0:
        return []  # No intersection
    elif discriminant == 0:
        t = -b / (2 * a)
        return [p1 + t * direction]  # One intersection point
    else:
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        coords = np.array([p1 + t1 * direction, p1 + t2 * direction])
        sorted_coords = coords[coords[:, 0].argsort()]
        return sorted_coords  # Two intersection points


def tilt_correction(img, baseline, user_set_baseline=False):
    """img is an image input
    baseline is defined by two points in the image"""

    p1, p2 = baseline[0], baseline[1]
    x1, y1 = p1
    x2, y2 = p2

    if y1 == y2:  # image is level
        return img

    if user_set_baseline == True:
        img = img[: int(max([y1, y2])), :]

    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)

    # crop black edges created when rotating
    width = np.sin(np.deg2rad(rotate_angle))
    side = math.ceil(abs(width * rotate_img.shape[1]))
    roof = math.ceil(abs(width * rotate_img.shape[0]))
    rotate_img_crop = rotate_img[roof:-roof, side:-side]

    return rotate_img_crop


def isolate_contour(img_orig, crop=True, DISPLAY=False, blur=True):
    """
    Isolate the contour of a drop and its surrounding surface from an input image.

    Args:
        img_orig (numpy.ndarray): The input image containing the drop and surface.
        DISPLAY (bool, optional): Whether to display intermediate results and plots. Defaults to False.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - longest (numpy.ndarray): An array of (x, y) coordinates representing the contour of the drop and surface.
            - drop_only (numpy.ndarray): An array of (x, y) coordinates representing an approximate contour of the drop only.

    This function performs the following steps:
    1. Removes interference from the needle (if present) in the input image.
    2. Extracts edges from the input image using the `extract_edges_CV` function.
    3. Clusters the edges using the `cluster_OPTICS` function and finds the longest contour group.
    4. Attempts to find a surface line (baseline) using the `hough_baseline` function.
    5. If a baseline is found, performs tilt correction on the input image.
    6. Attempts to find a circular drop using the `hough_circle` function.
    7. Displays the identified shapes of interest (baseline and circle) if `DISPLAY` is True.
    8. Crops the image based on the detected baseline and circle, with some padding.
    9. Extracts edges from the cropped image and clusters them using `cluster_OPTICS`.
    10. Determines an approximate contour for the drop only, based on the detected baseline or circle.
    11. If `DISPLAY` is True, plots the contours of the drop and surface, as well as the drop only.
    12. Returns the contours of the drop and surface (`longest`), and the approximate contour of the drop only (`drop_only`).
    """
    ### declare flags
    info = {}
    BASELINE = np.array([[None, None], [None, None]])
    CIRCLE = None

    ### Remove interference from the needle
    ### won't work for high tilt systems, but shouldn't interfere

    # find longest group
    ### find edges
    edges, threshold = extract_edges_CV(
        img_orig, return_thresholed_value=True, blur=blur
    )

    groups = cluster_OPTICS(edges, xi=0.8)
    keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    # Rank (index into `keys`, largest-cluster-first) of the group judged to be
    # the drop, not a raw dict key -- cluster_OPTICS() labels are internal to
    # each independent OPTICS fit and aren't guaranteed to mean the same thing
    # across two separate fits (see the re-fit after tilt correction below).
    chosen_rank = 0
    maxkey = keys[chosen_rank]
    longest_contour = np.array(groups[maxkey])

    if crop == False:
        if DISPLAY:
            plt.imshow(img_orig)
            plt.plot(longest_contour[:, 0], longest_contour[:, 1])
            plt.show()
            plt.close()

        return longest_contour, threshold
    else:  # crop == True
        # check that the longest contour is not the needle
        # look for contour points which are close together and touch the top of the image
        top_of_image = []
        for coord in longest_contour:
            if coord[1] <= 10:
                maybe_needle = True
                top_of_image.append(coord)
            else:
                maybe_needle = False
        if maybe_needle == True:
            top_of_image = np.array(top_of_image)
            top_of_image = top_of_image[
                top_of_image[:, 0].argsort()
            ]  # order by x coord
            midpoint = (
                top_of_image[0, 0] + (top_of_image[-1, 0] - top_of_image[0, 0]) / 2
            )
            left_of_midpoint = top_of_image.copy()[top_of_image.copy()[:, 0] < midpoint]
            right_of_midpoint = top_of_image.copy()[
                top_of_image.copy()[:, 0] > midpoint
            ]
            left_of_midpoint = left_of_midpoint[left_of_midpoint[:, 0].argsort()][-1][
                0
            ]  # get max x
            right_of_midpoint = right_of_midpoint[right_of_midpoint[:, 0].argsort()][0][
                0
            ]  # get min x
            # if dist between these points is less than 15% of image width, it's probably the needle
            # 5< in case there is a lighting issue that isn't the needle
            if 5 < right_of_midpoint - left_of_midpoint < img_orig.shape[1] * 0.10:
                chosen_rank = 1
                maxkey = keys[chosen_rank]
                longest_contour = np.array(groups[maxkey])

        _debug_emit(
            "first_pick",
            chosen_rank=chosen_rank,
            key_sizes=[len(groups[k]) for k in keys],
            longest_contour=longest_contour,
        )

        if DISPLAY:
            plt.title("image with longest found contour that is not the needle")
            plt.imshow(img_orig)
            plt.plot(longest_contour[:, 0], longest_contour[:, 1], ".")
            plt.axis("equal")
            plt.show()
            plt.close()

        # convert edges to img shape for hough transform input
        canny_input = np.zeros((img_orig.shape[0], img_orig.shape[1]), dtype=np.uint8)
        for x, y in longest_contour:
            canny_input[y, x] = 255  # y,x because image coordinates

        if DISPLAY:
            plt.title("canny input")
            plt.imshow(canny_input)
            plt.axis("equal")
            plt.show()
            plt.close()

        ### Try find a surface line
        try:
            hough_line = np.array(hough_baseline(canny_input, 20, display=False))
            if min(hough_line[:, 1]) <= min(
                longest_contour[:, 1]
            ):  # check that line isn't just the drop of a flat drop
                hough_line = 0  # triggers exception to skip baseline correction
            elif hough_line[1, 1] - hough_line[0, 1] <= 1:
                # sometimes hough line will give slight gradient when there isn't any
                img = img_orig
            else:
                img = tilt_correction(img_orig, hough_line)

            # Recompute edge image after tilt correction. cluster_OPTICS() is
            # an independent OPTICS fit over different edge points, so its
            # cluster labels aren't guaranteed to line up with the labels
            # from the fit above -- reusing the old `maxkey` here previously
            # picked an unrelated (and platform-dependent, since OPTICS'
            # internal label assignment isn't guaranteed stable) cluster, or
            # raised a KeyError, discarding the needle-interference check's
            # rank-1 correction above. Re-sort the new fit's groups and take
            # the same *rank* the needle check chose, not the old label.
            edges = extract_edges_CV(img, blur=blur)
            groups = cluster_OPTICS(edges, xi=0.8)
            keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
            maxkey = keys[min(chosen_rank, len(keys) - 1)]
            longest_contour = np.array(groups[maxkey])
            _debug_emit(
                "second_pick",
                chosen_rank=chosen_rank,
                key_sizes=[len(groups[k]) for k in keys],
                longest_contour=longest_contour,
                tilted=not np.array_equal(img, img_orig),
            )
            canny_input = np.zeros(
                (img_orig.shape[0], img_orig.shape[1]), dtype=np.uint8
            )
            for x, y in longest_contour:
                canny_input[y, x] = 255  # y,x because image coordinates

            hough_line = hough_baseline(canny_input, display=False)
            BASELINE = np.array(hough_line)
            if DISPLAY:
                plt.title("Hough baseline after tilt")
                plt.imshow(img)
                plt.plot(BASELINE[:, 0], BASELINE[:, 1], "r")
                plt.axis("equal")
                plt.show()
                plt.close()
        except:
            img = img_orig
            if DISPLAY:
                print("No surface line could be found")

        ### Try find a roughly circular drop
        try:
            CIRCLE = hough_circle(canny_input, False)
        except:
            if DISPLAY:
                print("Hough circle failed to identify a drop")

        _debug_emit("shapes_found", CIRCLE=CIRCLE, BASELINE=BASELINE)

        if DISPLAY:
            plt.title("Identified shapes of interest")
            plt.imshow(img)
            try:
                center, radius = CIRCLE
                circle1 = plt.Circle(center, 1, color="r")
                # now make a circle with no fill, which is good for hi-lighting key results
                circle2 = plt.Circle(center, radius, color="r", fill=False)

                ax = plt.gca()
                ax.axis("equal")

                ax.add_patch(circle1)
                ax.add_patch(circle2)

                fig = plt.gcf()
                fig.set_size_inches(10, 10)
            except:
                pass
            try:
                plt.plot(BASELINE[:, 0], BASELINE[:, 1], "r")
            except:
                pass
            plt.axis("equal")
            plt.show()
            plt.close()

        ### crop the image accordingly
        padding = 0.05
        if CIRCLE != None and BASELINE.any() != None:  # both found
            center, radius = CIRCLE
            top = int(center[1] - radius - img.shape[0] * padding)
            bottom = int(max(BASELINE[:, 1]) + img.shape[0] * padding)
        elif CIRCLE != None:  # circle found but no line found - assume level surface
            center, radius = CIRCLE
            top = int(center[1] - radius - img.shape[0] * padding * 1.5)
            bottom = int(center[1] + radius + img.shape[0] * padding * 1.5)
        elif BASELINE.any() != None:  # line found but no circle - assume low CA drop
            top = min(longest_contour[:, 1])
            bottom = int(max(BASELINE[:, 1]) + img.shape[0] * padding)
        else:  # neither found
            top = 0
            bottom = img.shape[0]

        ### make sure the found bounds don't exceed the original image bounds
        if top < 0:
            top = 0
        if bottom > img.shape[0]:
            bottom = img.shape[0]

        _debug_emit("coarse_bounds", top=top, bottom=bottom)

        img_processed = img.copy()

        ### determine an approximate 'drop only' contour
        if (
            CIRCLE != None and BASELINE.any() != None
        ):  # set bounds using both the found circle and a line
            lowest_point = max(BASELINE[:, 1])
            intersections = line_circle_intersection(
                BASELINE[0], BASELINE[1], center, radius
            )
            if len(intersections) == 0:
                # if no intersect, ignore the baseline
                if DISPLAY:
                    print("intersects not found")
                BASELINE = np.array([[None, None], [None, None]])

            _debug_emit("intersections", intersections=intersections)

        if BASELINE.any() != None:
            div_line = (
                min(BASELINE[:, 1])
                - (min(BASELINE[:, 1]) - min(longest_contour[:, 1])) * 0.2
            )
        elif CIRCLE != None:
            div_line = center[1] + radius / 2
        else:
            # neither found use the top 50%
            div_line = (
                min(longest_contour[:, 1])
                + (max(longest_contour[:, 1]) - min(longest_contour[:, 1])) * 0.5
            )

        drop_only = np.array([[x, y] for x, y in longest_contour if y < div_line])

        ### now that the drop region has been roughly identified, repeat the edge detection over this cropped region
        ### doing this will result in a more accurate threshold value being selected, giving an more accurate edge

        if (
            CIRCLE != None and BASELINE.any() != None
        ):  # set bounds using both the found circle and a line
            lowest_point = max(BASELINE[:, 1])
            x1, y1 = intersections[0]
            x2, y2 = intersections[1]
            midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            is_high_angle = center[1] < midpoint[1] - radius * 0.4
            _debug_emit(
                "branch_decision",
                center=center,
                radius=radius,
                midpoint=midpoint.tolist(),
                margin=float(midpoint[1] - radius * 0.4 - center[1]),
                is_high_angle=bool(is_high_angle),
            )
            if is_high_angle:  # high angle system, slight bias for consistency
                try:  # account for Hough circle inaccuracy
                    right_of_center = []
                    left_of_center = []
                    for coord in longest_contour:
                        if coord[1] == center[1] and coord[0] > center[0]:
                            right_of_center.append(coord)
                        if coord[1] == center[1] and coord[0] < center[0]:
                            left_of_center.append(coord)
                    right_of_center = np.array(right_of_center)
                    left_of_center = np.array(left_of_center)
                    n_right_of_center = len(right_of_center)
                    n_left_of_center = len(left_of_center)
                    if n_right_of_center >= 1 and n_left_of_center >= 1:
                        right_of_center = right_of_center[
                            right_of_center[:, 0].argsort()
                        ][
                            0
                        ]  # min x coord
                        left_of_center = left_of_center[left_of_center[:, 0].argsort()][
                            -1
                        ]  # max x coord
                        right_dist = abs(center[0] - right_of_center[0])
                        left_dist = abs(center[0] - left_of_center[0])
                        radius = max([left_dist, right_dist])
                    _debug_emit(
                        "high_angle_radius_adjust",
                        n_right_of_center=n_right_of_center,
                        n_left_of_center=n_left_of_center,
                        radius_after=radius,
                    )
                except Exception as e:
                    _debug_emit("high_angle_radius_adjust_exception", error=str(e))
                bounds_temp = [
                    int(center[1] - radius * 1.2),  # top, 20% padding
                    int(
                        lowest_point + (lowest_point - center[1] + radius) * 0.4
                    ),  # bottom
                    int(center[0] - radius * 1.2),  # left
                    int(center[0] + radius * 1.2),
                ]  # right
            else:  # low angle system?
                # guess the CPs as the points closest to each circle/line intersect
                left_guess = min(
                    longest_contour, key=lambda x: distance1(intersections[0], x)
                )
                right_guess = min(
                    longest_contour, key=lambda x: distance1(intersections[1], x)
                )
                width = round(distance1(left_guess, right_guess))

                bounds_temp = [
                    int(center[1] - radius * 1.2),  # top, 20% padding
                    int(
                        lowest_point + (lowest_point - center[1] + radius) * 0.4
                    ),  # bottom
                    int(center[0] - width / 2 * 1.5),  # left
                    int(center[0] + width / 2 * 1.5),
                ]  # right
        elif CIRCLE != None:
            bounds_temp = [
                int(center[1] - radius * 1.2),  # 20% padding
                int(center[1] + radius * 1.2),
                int(center[0] - radius * 1.5),  # for high bond number systems (Bo=1)
                int(center[0] + radius * 1.5),
            ]  # for high bond number systems (Bo=1)
        else:
            height = max(longest_contour[:, 1]) - min(longest_contour[:, 1])
            width = max(drop_only[:, 0]) - min(drop_only[:, 0])
            bounds_temp = [
                int(min(longest_contour[:, 1]) - height * 0.4),
                int(max(longest_contour[:, 1]) + height * 0.4),
                int(min(drop_only[:, 0]) - width * 0.4),
                int(max(drop_only[:, 0]) + width * 0.4),
            ]

        ### keep to img bounds
        if bounds_temp[0] < 0:
            bounds_temp[0] = 0
        if bounds_temp[1] > img_processed.shape[0]:
            bounds_temp[1] = img_processed.shape[0]
        if bounds_temp[2] < 0:
            bounds_temp[2] = 0
        if bounds_temp[3] > img_processed.shape[1]:
            bounds_temp[3] = img_processed.shape[1]

        _debug_emit(
            "final_bounds",
            bounds_temp=bounds_temp,
            branch=(
                "circle+baseline"
                if (CIRCLE != None and BASELINE.any() != None)
                else (
                    "circle_only"
                    if CIRCLE != None
                    else "baseline_only" if BASELINE.any() != None else "neither"
                )
            ),
        )

        img_crop = img_processed[
            bounds_temp[0] : bounds_temp[1], bounds_temp[2] : bounds_temp[3]
        ]

        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if DISPLAY:
            print("cropped threshold value: ", ret)
            plt.title("drop region")
            plt.imshow(thresh)
            plt.show()
            plt.close()
        edges = extract_edges_CV(img_orig, ret, blur=blur)

        # mask contour to identified drop region
        x_mask = (edges[:, 0] >= bounds_temp[2]) & (edges[:, 0] <= bounds_temp[3])
        y_mask = (edges[:, 1] >= bounds_temp[0]) & (edges[:, 1] <= bounds_temp[1])
        mask = x_mask & y_mask

        # Apply the mask to get the filtered coordinates
        filtered_coords = edges[mask]
        groups = cluster_OPTICS(filtered_coords, eps=2)
        maxkey = max(groups, key=lambda k: len(groups[k]))
        filtered_coords = np.array(groups[maxkey])

        div_line = (
            min(filtered_coords[:, 1])
            + (max(filtered_coords[:, 1]) - min(filtered_coords[:, 1])) * 0.5
        )
        drop_only = np.array([[x, y] for x, y in filtered_coords if y < div_line])

        if DISPLAY:
            plt.imshow(img_processed)
            plt.plot(filtered_coords[:, 0], filtered_coords[:, 1], "r,")
            plt.plot(drop_only[:, 0], drop_only[:, 1], "b.")
            plt.show()
            plt.close()

        info["line"] = BASELINE
        info["circle"] = CIRCLE
        info["bounds"] = bounds_temp
        info["threshold"] = threshold

        return img_processed, filtered_coords, drop_only, info


def linear_fit(coords, display=False):
    """
    Fit a line to an array of coordinates using Orthogonal Distance Regression (ODR).
    This method correctly fits vertical lines if needed.

    Args:
        coords (list): A list of (x, y) coordinate tuples.
        display (bool): If True, displays the contour, fitted line, and RMSE.

    Returns:
        tuple: A tuple containing the slope (m), intercept (c), and RMSE value.
    """
    x_coords, y_coords = np.array(coords).T

    if np.std(x_coords) < 1e-6:  # Detect vertical line
        return np.inf, np.mean(x_coords), 0  # Slope = infinity, Intercept = mean x

    # Define linear function for ODR
    def linear_func(beta, x):
        return beta[0] * x + beta[1]

    # Use initial guess from OLS (to handle non-vertical cases well)
    A = np.vstack([x_coords, np.ones_like(x_coords)]).T
    m_init, c_init = np.linalg.lstsq(A, y_coords, rcond=None)[0]

    # Fit using ODR
    model = Model(linear_func)
    data = Data(x_coords, y_coords)
    odr = ODR(data, model, beta0=[m_init, c_init])
    out = odr.run()

    m, c = out.beta  # Best-fit parameters

    # Calculate RMSE using perpendicular distances
    distances = np.abs(y_coords - (m * x_coords + c)) / np.sqrt(1 + m**2)
    rmse_value = np.sqrt(np.mean(distances**2))

    # Display the plot if required
    if display:
        plt.scatter(x_coords, y_coords, color="blue", label="Data Points")

        # Plot fitted line
        x_line = np.linspace(min(x_coords), max(x_coords), 100)
        y_line = m * x_line + c
        plt.plot(
            x_line, y_line, color="red", label=f"Fitted Line: y = {m:.2f}x + {c:.2f}"
        )

        # Display the equation and RMSE
        plt.text(
            0.05,
            0.95,
            f"Slope: {m:.2f}\nIntercept: {c:.2f}\nRMSE: {rmse_value:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.5),
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("ODR Linear Fit")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    return m, c, rmse_value


def iterative_Lfit(
    contour, n_pts, height=0, prominence=0, width=3, display=False, fallback_points=None
):
    """Takes the contour of an contact angle experimental image, and returns the
    coordinates of the contact points. These point is found by tracking the gradient
    of lines fitted to every n_pts coordinates change along the contour.
    contour - a list or numpy array of coordinates of one half of the image, this is
        assumed to include the needle.
    n_pts - the gradient is calculated between coordinate i and coordinate i+n_pts
    height - used to find signal peak in graph of RMSE. Specifies the minimum height
        (vertical distance from the base) that a peak must have to be considered a
        valid peak. Must be a non-negative value.
    prominence - used to find signal peak in graph of RMSE. Specifies the minimum
        vertical distance that a peak must have from its surrounding data (i.e., the
        closest higher peak) to be considered a valid peak. Must be a non-negative value.
    width - used to find signal peak in graph of RMSE. Specifies the minimum width (in
        number of data points) that a peak must have to be considered a valid peak. Must
        be a positive integer, 1 means no width requirement.
    display - set to True to visualise progress
    fallback_points - [left_point, right_point] to return directly, without attempting
        gradient-based refinement, when contour is too sparse to reliably split into a
        left/right apex division (fewer points than the per-side gradient windows need).
        Callers pass whichever candidate points (widest_points or CP_guesses) they used
        to build the restricted contour in the first place, matching the fallback
        already used elsewhere in CPID() (`final_CPs = widest_points`) when refinement
        is unreliable -- this is the same pattern, not a new heuristic.
    Note that the point where the drop meets the needle is a point of gradient change
        and so this point is not returned.
    The points which this function identifies are the points of low gradient which occur
        after large changes in gradient - this can be visualised by setting display
        to True
    """

    if n_pts < 12:
        n_pts = 12
    if display:
        print(f"iterative fit approach using {n_pts} points")

    # define output variables
    CPs_output = []

    # determine drop_guess using n_pts between widest points
    if len(contour) >= 110:
        # find drop apex and split into left and right sides
        divline = int(
            min(contour[:, 1]) + (max(contour[:, 1] - min(contour[:, 1]) * 0.2))
        )

        top = np.array([[x, y] for x, y in contour if y <= divline])
        apex = np.array([[x, y] for x, y in top if y == min(top[:, 1])])
        xapex = min(apex[:, 0]) + (max(apex[:, 0]) - min(apex[:, 0])) / 2

        left = np.array([[x, y] for x, y in contour if x <= xapex])
        right = np.array([[x, y] for x, y in contour if x > xapex])
    elif len(contour) == 2:
        left = contour[0]
        right = contour[1]
    else:
        # Too few points to reliably apex-split (needs >=110) and not already a
        # pre-split [left, right] pair (==2). Degrade to the caller-supplied
        # candidate points rather than falling through with left/right unassigned
        # (that used to raise UnboundLocalError at `contours = [left, right]` below).
        if fallback_points is None or len(fallback_points) != 2:
            raise ValueError(
                f"iterative_Lfit: contour has {len(contour)} points, which is neither "
                ">=110 (full apex-split) nor exactly 2 (pre-split [left, right] pair), "
                "and no fallback_points were supplied to degrade to."
            )
        print(
            f"WARNING: iterative_Lfit contour is sparse ({len(contour)} points, "
            "need >=110 to apex-split); using the caller's candidate points "
            "directly instead of refining. Predictions may be unreliable."
        )
        return [
            [np.asarray(fallback_points[0]).tolist()],
            [np.asarray(fallback_points[1]).tolist()],
        ]

    contours = [left, right]

    for side, contour in enumerate(contours):
        if type(contour) != list:
            contour = contour.tolist()
        # order the contour
        if side == 0:
            top_down = sorted(contour[::-1], key=lambda x: x[1])
        else:
            top_down = sorted(contour, key=lambda x: x[1])
        contour = optimized_path(top_down, start=top_down[0])

        if display:
            jet = plt.get_cmap("jet")
            colors = iter(jet(np.linspace(0, 1, len(contour))))
            for n, coord in enumerate(contour):
                plt.plot(contour[n, 0], contour[n, 1], "o", color=next(colors))
            plt.title("ordered points, starting from blue")
            plt.axis("equal")
            plt.show()
            plt.close()

        # perform iterative linear fits
        gradients = []
        RMSEs = []
        for i, coord in enumerate(contour):
            if i < len(contour) - n_pts:
                m, c, rmse = linear_fit(contour[i : i + n_pts])
                gradients.append(m)
                RMSEs.append(rmse)

        # set the gradient of vertical lines to be the same as the previous point
        for i, m in enumerate(gradients):
            if str(m) == "inf" or str(m) == "-inf":
                gradients[i] = gradients[i - 1]

        # A symmetric corner gives two adjacent windows mathematically equal
        # RMSEs, which ODR's convergence tolerance leaves ~1e-10 apart in an
        # order that differs by platform. Rounding makes such ties exact
        # plateaus, which find_peaks and index() resolve deterministically.
        RMSEs_for_peaks = np.round(RMSEs, 6).tolist()

        ### return the indices of each RMSE peak
        RMSE_peaks_indices = signal.find_peaks(
            RMSEs_for_peaks, height=max(RMSEs_for_peaks) * 0.3, prominence=0.3, width=5
        )[0]
        RMSE_peaks_indices = sorted(RMSE_peaks_indices.tolist())
        if display == True:
            print("found indices using signal.find_peaks: ", RMSE_peaks_indices)
            if len(RMSE_peaks_indices) == 0:
                print("No peaks found in RMSE plot")

        # return the index of the max RMSE point, often this is missed by the signal package - so add it manually
        max_RMSE_idx = RMSEs_for_peaks.index(np.nanmax(RMSEs_for_peaks))
        if max_RMSE_idx not in RMSE_peaks_indices:
            RMSE_peaks_indices.append(max_RMSE_idx)
        RMSE_peaks_indices = sorted(RMSE_peaks_indices)

        # include index of points where the gradient is near zero and changes sign - to catch reflections
        threshold = (max(gradients) - min(gradients)) * 0.3
        sign_changes = np.where(abs(np.diff(np.sign(gradients))) >= 1)[0]
        near_zero_indices = np.where(np.abs(gradients) < threshold)[0]
        gradient_change_indices = [
            val for val in near_zero_indices if val in sign_changes
        ]

        # put indices of gradient change and high RMSE together into a list
        indices = (
            gradient_change_indices + RMSE_peaks_indices[::-1]
        )  # order this way to leave the most likely index last

        # remove any duplicates
        indices = list(set(indices))

        if display == True:
            print(
                "found indices using RMSE peaks and points of gradient change: ",
                sorted(indices),
            )

        # don't use the first or last points
        indices = [idx for idx in indices if idx != 0 and idx != len(RMSEs) - 1]
        # don't use points at the very start of the curve
        indices = [idx for idx in indices if idx >= 3 * n_pts]
        # don't use points where the RMSE immediately before or after is 0 - this might just be where the line is verical
        indices = [
            index
            for index in indices
            if RMSEs[index - 1] != 0 and RMSEs[index + 1] != 0
        ]

        # set variables for exclusion search
        RMSE_h = max(RMSEs) - min(RMSEs)
        RMSE_jump_too_high = 0.25

        indices.sort()
        # Filter out indices with too high RMSE jumps at high gradients
        indices = [
            idx
            for idx in indices
            if len(indices) <= 1
            or (
                (
                    abs(RMSEs[idx] - RMSEs[idx - 1]) <= RMSE_h * RMSE_jump_too_high
                    and abs(RMSEs[idx] - RMSEs[idx + 1]) <= RMSE_h * RMSE_jump_too_high
                )
                or abs(gradients[idx]) <= 0.5
            )
        ]

        # take out any indices near the drop apex (first 15% of drop contour points)
        indices = [value for value in indices if value > len(contour) * 0.15]

        # remove any points that are now out of range of the length of the contour
        indices = [idx for idx in indices if idx <= len(gradients)]

        # take out any indices for points higher than the 60% contour height
        div_line = min(contour[:, 1]) + (max(contour[:, 1]) - min(contour[:, 1])) * 0.4
        indices = [idx for idx in indices if contour[idx, 1] >= div_line]

        # now order lowest to highest
        indices = sorted(indices)

        # if none left use the max RMSE
        if len(indices) < 1:
            indices.append(max_RMSE_idx)

        if display:
            print()
            print("indices of interest found: ", indices)

            plt.title("plot of fit RMSEs")
            plt.plot(range(len(RMSEs)), RMSEs, ".")
            h = max(RMSEs) - min(RMSEs)
            w = len(RMSEs)
            for i, idx in enumerate(indices):
                if i == 0:
                    plt.plot(idx, RMSEs[idx], "r+")
                else:
                    plt.plot(idx, RMSEs[idx], "+", color="pink")
                label = f"({idx:.2f}, {RMSEs[idx]:.2f})"
                plt.annotate(
                    label,
                    xy=(idx, RMSEs[idx]),
                    xytext=(idx + w * 0.1, RMSEs[idx] - h * 0.1),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                )
            plt.xlabel("coordinate index")
            plt.ylabel("RMSE")
            plt.show()
            plt.close()

            plt.title("plot of gradients")
            plt.plot(range(len(gradients)), gradients, ".")
            for i, idx in enumerate(indices):
                if i == 0:
                    plt.plot(idx, gradients[idx], "r+")
                else:
                    plt.plot(idx, gradients[idx], "+", color="pink")
                label = f"({idx:.2f}, {gradients[idx]:.2f})"
                plt.annotate(
                    label,
                    xy=(idx, gradients[idx]),
                    xytext=(idx + w * 0.1, gradients[idx] - h * 0.1),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                )

            plt.xlabel("coordinate index")
            plt.ylabel("gradient")
            plt.show()
            plt.close()

        CP_indices = []
        for idx in indices:
            CP_indices.append(idx + round(n_pts / 2))

        if display:
            print("index of identified points of interest: ", CP_indices)
            plt.title("Points of interest")
            plt.plot(contour[:, 0], contour[:, 1], ",")
            h = max(contour[:, 1]) - min(contour[:, 1])
            w = len(contour[:, 0])
            for i, idx in enumerate(CP_indices):
                if i == 0:
                    plt.plot(contour[idx, 0], contour[idx, 1], "r+")
                    label = f"({contour[idx,0]:.2f}, {contour[idx,1]:.2f})"
                    plt.annotate(
                        label,
                        xy=(contour[idx, 0], contour[idx, 1]),
                        xytext=(contour[idx, 0], contour[idx, 1] + 100),
                        textcoords="data",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    )  # h*0.05
                else:
                    plt.plot(contour[idx, 0], contour[idx, 1], "r+")
            plt.plot(contour[:n_pts, 0], contour[:n_pts, 1], "g.")
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.show()
            plt.close()

        CPs_output.append(contour[CP_indices].tolist())

    return CPs_output


def find_leftmost_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Find the leftmost (minimum x) coordinate for each unique y value in the input array.

    This function takes a 2D numpy array of x,y coordinates and returns a new array
    containing only the leftmost coordinate for each unique y value.

    Args:
        coordinates (np.ndarray): A 2D numpy array of shape (n, 2) where each row
                                  represents an (x, y) coordinate pair.

    Returns:
        np.ndarray: A 2D numpy array of shape (m, 2) where m is the number of unique
                    y values in the input. Each row contains the leftmost (x, y)
                    coordinate for a unique y value.

    Raises:
        ValueError: If the input array is not 2D or doesn't have exactly 2 columns.

    Example:
        >>> coords = np.array([[1, 2], [3, 2], [2, 3], [4, 3], [0, 1], [5, 1]])
        >>> find_leftmost_coordinates(coords)
        array([[0, 1],
               [1, 2],
               [2, 3]])
    """
    # Check if the input array has the correct shape
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Input must be a 2D array with 2 columns (x, y)")

    # Sort the array by y values (column 1) and then by x values (column 0)
    sorted_coords = coordinates[np.lexsort((coordinates[:, 0], coordinates[:, 1]))]

    # Find the indices where y values change
    y_changes = np.diff(sorted_coords[:, 1]).nonzero()[0]

    # Get the leftmost coordinates for each unique y value
    leftmost_coords = np.vstack((sorted_coords[0], sorted_coords[y_changes + 1]))

    return leftmost_coords


def find_rightmost_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Find the rightmost (maximum x) coordinate for each unique y value in the input array.

    This function takes a 2D numpy array of x,y coordinates and returns a new array
    containing only the rightmost coordinate for each unique y value.

    Args:
        coordinates (np.ndarray): A 2D numpy array of shape (n, 2) where each row
                                  represents an (x, y) coordinate pair.

    Returns:
        np.ndarray: A 2D numpy array of shape (m, 2) where m is the number of unique
                    y values in the input. Each row contains the rightmost (x, y)
                    coordinate for a unique y value.

    Raises:
        ValueError: If the input array is not 2D or doesn't have exactly 2 columns.

    Example:
        >>> coords = np.array([[1, 2], [3, 2], [2, 3], [4, 3], [0, 1], [5, 1]])
        >>> find_rightmost_coordinates(coords)
        array([[5, 1],
               [3, 2],
               [4, 3]])
    """
    # Check if the input array has the correct shape
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Input must be a 2D array with 2 columns (x, y)")

    # Sort the array by y values (column 1) and then by x values in descending order (column 0)
    sorted_coords = coordinates[np.lexsort((-coordinates[:, 0], coordinates[:, 1]))]

    # Find the indices where y values change
    y_changes = np.diff(sorted_coords[:, 1]).nonzero()[0]

    # Get the rightmost coordinates for each unique y value
    rightmost_coords = np.vstack((sorted_coords[0], sorted_coords[y_changes + 1]))

    return rightmost_coords


def calculate_contact_angle(gradient, side="left"):
    """
    Calculates the internal contact angle from a given gradient for either the left or right side,
    taking into account the flipped y-axis in image coordinates.

    Parameters:
    gradient (float): The slope of the line (gradient).
    side (str): Either 'left' or 'right', specifying which side's contact angle to calculate.

    Returns:
    float: The contact angle in degrees, measured from the horizontal.
    """
    if np.isnan(gradient):
        gradient = np.inf

    # Adjust the gradient for the image coordinate system where the y-axis is flipped
    gradient = -gradient

    # Calculate the angle in radians using the arctangent of the adjusted gradient
    angle_radians = np.arctan(gradient)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    if side == "left":
        # For the left side, ensure the angle is counter-clockwise from the horizontal
        if angle_degrees < 0:
            angle_degrees += 180
    elif side == "right":
        # For the right side, ensure the angle is clockwise from the horizontal
        angle_degrees = -angle_degrees
        if angle_degrees < 0:
            angle_degrees += 180
    else:
        raise ValueError("The 'side' parameter must be either 'left' or 'right'.")

    return angle_degrees


def _find_point_index(contour, point, context=""):
    """Find the index of a point in a contour array, raising a clear error if not found.

    Parameters
    ----------
    contour : np.ndarray
        The contour array to search in.
    point : np.ndarray or list
        The point to find.
    context : str
        Description of the search context for error messages.

    Returns
    -------
    int
        The index of the first matching row.

    Raises
    ------
    ValueError
        If the point is not found in the contour.
    """
    matches = np.where((contour == point).all(axis=1))[0]
    if len(matches) == 0:
        raise ValueError(
            f"Point {point} not found in contour (shape {contour.shape}). "
            f"Context: {context}. "
            f"This may be caused by floating-point mismatch. "
            f"Try running with display=True to visualise the contour."
        )
    return matches[0]


def check_for_high_angle(contour, display=False):

    # check the shape of the input contour
    if len(contour) > 110:
        # input is a single contour, needle out of drop

        # find drop apex and split into left and right sides
        divline = int(
            min(contour[:, 1]) + (max(contour[:, 1] - min(contour[:, 1]) * 0.2))
        )

        top = np.array([[x, y] for x, y in contour if y <= divline])
        apex = np.array([[x, y] for x, y in top if y == min(top[:, 1])])
        xapex = min(apex[:, 0]) + (max(apex[:, 0]) - min(apex[:, 0])) / 2

        left = np.array([[x, y] for x, y in contour if x <= xapex])
        right = np.array([[x, y] for x, y in contour if x > xapex])
    elif len(contour) == 2:
        # input is several contours, needle inside drop
        # assuming two contours
        left = contour[0]
        right = contour[1]

        # check that the first point of each contour is the highest point on the contour
        if left[0, 1] < min(left[:, 1]):
            # take the highest and left most point as the starting point of the left contour
            apex = min(left, key=lambda point: (point[1], point[0]))
            start = _find_point_index(
                left, apex, "check_for_high_angle: left contour apex"
            )
            left = left[start:]
        if right[0, 1] < min(right[:, 1]):
            # take the highest and right most point as the starting point of the right contour
            apex = min(right, key=lambda point: (point[1], -point[0]))
            start = _find_point_index(
                right, apex, "check_for_high_angle: right contour apex"
            )
            right = right[start:]
    else:
        # Too few points to apex-split (needs >110) and not already a
        # pre-split [left, right] pair (==2). Raise rather than falling
        # through with left/right unassigned (that used to raise
        # UnboundLocalError at `for i, edge in enumerate([left, right])`
        # below) -- same fix as iterative_Lfit's identical sparse-contour
        # case.
        raise ValueError(
            f"check_for_high_angle: contour has {len(contour)} points, which is "
            "neither >110 (full apex-split) nor exactly 2 (pre-split [left, right] "
            "pair)."
        )
    CP_guesses = []
    widest_points = []

    for i, edge in enumerate([left, right]):
        if i == 0:
            edge = find_rightmost_coordinates(
                edge
            )  # to ignore any rough surface points
        elif i == 1:
            edge = find_leftmost_coordinates(edge)

        if len(edge) <= 10:
            raise ValueError(
                f"check_for_high_angle: {'left' if i == 0 else 'right'} edge has "
                f"{len(edge)} points after filtering, need >10 to skip the apex "
                "region and read the widest point at index 10."
            )
        widest = edge[10]
        CP_guess = None

        curves_cut_in = False
        for idx, coord in enumerate(edge):
            if (
                idx > 10
            ):  # skip the first 10 to miss any edge detection errors at the apex
                if i == 0:  # left
                    if CP_guess is None:
                        if (
                            coord[0] < widest[0]
                            and coord[1] > widest[1]
                            and abs(coord[0] - widest[0]) < 15
                        ):
                            widest = coord
                        if (
                            coord[0] > widest[0]
                            and coord[1] > widest[1]
                            and curves_cut_in == False
                        ):
                            CP_guess = coord
                    else:  # CP_guess is not None
                        if (
                            coord[0] < widest[0]
                            and coord[1] > widest[1]
                            and coord[1] <= CP_guess[1]
                        ):
                            widest = coord
                        if coord[0] < CP_guess[0]:
                            curves_cut_in = True
                        if coord[0] >= CP_guess[0] and coord[1] > CP_guess[1]:
                            CP_guess = coord
                        elif (
                            coord[0] < CP_guess[0] and coord[1] > CP_guess[1]
                        ):  # if it starts widening again after CP_guess found then stop
                            break
                elif i == 1:  # right
                    if CP_guess is None:
                        if (
                            coord[0] > widest[0]
                            and coord[1] > widest[1]
                            and abs(coord[0] - widest[0]) < 15
                        ):
                            widest = coord
                        if (
                            coord[0] < widest[0]
                            and coord[1] > widest[1]
                            and curves_cut_in == False
                        ):
                            CP_guess = coord
                    else:  # CP_guess is not None
                        if (
                            coord[0] > widest[0]
                            and coord[1] > widest[1]
                            and coord[1] <= CP_guess[1]
                        ):
                            widest = coord
                        if coord[0] > CP_guess[0]:
                            curves_cut_in = True
                        if coord[0] <= CP_guess[0] and coord[1] > CP_guess[1]:
                            CP_guess = coord
                        elif (
                            coord[0] > CP_guess[0] and coord[1] > CP_guess[1]
                        ):  # if it starts widening again after CP_guess found then stop
                            break
        if CP_guess is None:
            CP_guess = widest

        # load outputs to lists
        CP_guesses.append(CP_guess)
        widest_points.append(widest)

    # convert lists to arrays
    CP_guesses = np.array(CP_guesses)
    widest_points = np.array(widest_points)

    if display:
        plt.title(
            f"check outputted vals\nwidest points:{widest_points}\nCP guesses:{CP_guesses}"
        )
        plt.plot(left[:, 0], left[:, 1], "C0,")
        plt.plot(right[:, 0], right[:, 1], "C0,")
        plt.plot(CP_guesses[0, 0], CP_guesses[0, 1], "C1.", label="CP guess left")
        plt.plot(CP_guesses[1, 0], CP_guesses[1, 1], "C2.", label="CP guess right")
        plt.plot(
            widest_points[:, 0],
            widest_points[:, 1],
            "C3.",
            label="widest points guesses",
        )
        # plt.plot(contour[:,0], contour[:,1], 'C0,')
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.legend()
        plt.show()
        plt.close()
    return CP_guesses, widest_points


KINK_RATIO_THRESHOLD = 3.0
# Edge noise on real images produces bottom-minus-top steps of up to ~12 deg
# over a 10-point window even at a smooth equator (max 11.8 deg over 98
# experimental high-angle drops); the ratio alone amplifies that on large
# drops, so a step must also clear this floor in degrees.
KINK_MIN_DEG = 14.0


def _reflective_kink(top1, bottom1, top2, bottom2, widest_points, n_pts):
    """True when the tangent step at the widest point is too large to be the
    arc curvature of a smooth drop over an n_pts window at radius = half the
    width, and also above the real-image edge-noise floor."""
    width = abs(float(widest_points[1][0]) - float(widest_points[0][0]))
    if width < 1:
        return False
    arc_deg = np.degrees(n_pts / (width / 2))
    kink_deg = max(bottom1 - top1, bottom2 - top2)
    return kink_deg > KINK_MIN_DEG and kink_deg / arc_deg > KINK_RATIO_THRESHOLD


def CPID(contour, display=False):
    """
    Identifies the contact points (CPs) from a given contour based on its shape and angles.

    This function first determines candidate CPs and the widest points of the contour. Depending on the number of contour points,
    it calculates an appropriate number of points for a linear fit (Lfit) approach. The function then checks whether the candidate CPs
    match the widest points. If they do, it applies an Lfit method to determine the final CPs. Otherwise, it estimates contact angles
    at the widest points and selects the CPs based on predefined angle thresholds.

    Parameters:
    ----------
    contour : np.ndarray
        A 2D array representing the contour coordinates.
    display : bool, optional
        If True, prints debugging information about the CP determination process (default is False).

    Returns:
    -------
    np.ndarray
        A 2D array containing the coordinates of the final contact points.

    Notes:
    ------
    - If the number of contour points is 110 or more, the function treats it as a single contour.
    - If there are exactly 2 contours, it processes them separately.
    - If the determined CPs match the widest points, an Lfit approach is used.
    - Otherwise, the function estimates contact angles and classifies the system as reflective, hydrophobic, or low-angle reflective.

    """
    # check for CP guesses and the widest points
    CP_guesses, widest_points = check_for_high_angle(contour, display=display)

    if len(contour) == 2:  # if working with two contours (left and right)
        contour = np.concatenate(contour, axis=0)
        if display:
            print("contour concatenated into a single line")

    # determine drop_guess using n_pts between widest points
    if len(contour) > 110:
        # for a single contour
        widest_index1 = _find_point_index(
            contour, widest_points[0], "CPID: widest_points[0] in single contour"
        )
        widest_index2 = _find_point_index(
            contour, widest_points[1], "CPID: widest_points[1] in single contour"
        )
        dist_between_widest = widest_index2 - widest_index1

        percentage = 2.5
        Lfit_n_pts = int(dist_between_widest * percentage / 100)
    elif len(contour) == 2:
        # for two contours
        widest_index1 = _find_point_index(
            contour[0], widest_points[0], "CPID: widest_points[0] in left contour"
        )
        widest_index2 = _find_point_index(
            contour[1], widest_points[1], "CPID: widest_points[1] in right contour"
        )
        dist_between_widest = widest_index2 + widest_index1

        percentage = 2.5
        Lfit_n_pts = int(dist_between_widest * percentage / 100)
    else:
        if display:
            print(
                f"Number of contours is not correct for CPID, contour.shape[0] == {contour.shape[0]}"
            )
            print("Taking first 2 contours")

    n_pts = 10

    # first check if any of the CP_guesses are the same as any of the widest points
    if np.any(np.all(CP_guesses[:, None] == widest_points, axis=2)):
        # use Lfit
        restriction1 = max(0, widest_index1 - 4 * n_pts)
        restriction2 = min(len(contour), widest_index2 + 4 * n_pts)
        restricted_contour = contour[restriction1:restriction2]

        Lfit_out = iterative_Lfit(
            restricted_contour, Lfit_n_pts, display=display, fallback_points=widest_points
        )
        left_Lfit, right_Lfit = Lfit_out[0][0], Lfit_out[1][0]

        # take the right most point of those determined by Lfit and widest points
        final_left = max([left_Lfit, widest_points[0]], key=lambda point: point[0])
        # take the left most point of those determined by Lfit and widest points
        final_right = min([right_Lfit, widest_points[1]], key=lambda point: point[0])

        final_CPs = np.array([final_left, final_right])
        if display:
            print(f"final_CPs outputted by Lfit: {final_CPs}")

    else:
        shuffle = 0

        # angles of left side
        if len(contour) >= 110:
            widest1_bottom_m, widest1_bottom_c, widest1_bottom_rmse = linear_fit(
                contour[widest_index1 - shuffle - n_pts : widest_index1 - shuffle]
            )
            widest1_top_m, widest1_top_top_c, widest1_top_rmse = linear_fit(
                contour[widest_index1 : widest_index1 + n_pts]
            )
        elif len(contour) == 2:
            widest1_bottom_m, widest1_bottom_c, widest1_bottom_rmse = linear_fit(
                contour[0][widest_index1 - shuffle : widest_index1 + n_pts - shuffle]
            )
            widest1_top_m, widest1_top_top_c, widest1_top_rmse = linear_fit(
                contour[0][widest_index1 - n_pts : widest_index1]
            )
        widest1_bottom_angle = calculate_contact_angle(widest1_bottom_m, "left")
        widest1_top_angle = calculate_contact_angle(widest1_top_m, "left")

        # angles of right side
        if len(contour) >= 110:
            widest2_bottom_m, widest2_bottom_c, widest2_bottom_rmse = linear_fit(
                contour[widest_index2 + shuffle : widest_index2 + n_pts + shuffle]
            )
            widest2_top_m, widest2_top_c, widest2_top_rmse = linear_fit(
                contour[widest_index2 - n_pts : widest_index2]
            )
        elif len(contour) == 2:
            widest2_bottom_m, widest2_bottom_c, widest2_bottom_rmse = linear_fit(
                contour[1][widest_index2 + shuffle : widest_index2 + n_pts + shuffle]
            )
            widest2_top_m, widest2_top_c, widest2_top_rmse = linear_fit(
                contour[1][widest_index2 - n_pts : widest_index2]
            )
        widest2_bottom_angle = calculate_contact_angle(widest2_bottom_m, "right")
        widest2_top_angle = calculate_contact_angle(widest2_top_m, "right")

        error_margin = 18
        final_CPs = []

        # is the top angle at the widest point high?
        if (
            widest1_top_angle > 90 - error_margin
            or widest2_top_angle > 90 - error_margin
        ):
            # if the angle under the widest point is big enough then it should be reflective
            if widest1_bottom_angle > 100 or widest2_bottom_angle > 100:
                if display:
                    print("large angles under widest points - low angle reflective")
                    print(
                        f"angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                    )
                    print(
                        f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                    )
                final_CPs = widest_points

            # check if the angles over and under the widest point are close to each other, and the bottom angle is harsh enough - then reflective
            elif (
                180 - widest1_top_angle - widest1_bottom_angle < 8
                and widest1_bottom_angle > 99
            ) or (
                180 - widest2_top_angle - widest2_bottom_angle < 8
                and widest1_bottom_angle > 99
            ):
                if display:
                    print(
                        "angles on either side of widest points are similar - reflective"
                    )
                    print(
                        f"angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                    )
                    print(
                        f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                    )
                final_CPs = widest_points

            # A reflection meets the drop at the contact line with a tangent
            # step of ~2*(90 - theta); a genuine high-angle drop's equator is
            # smooth, so its bottom-minus-top angle is only the arc curvature
            # over the n_pts window. Normalising by that expected arc makes the
            # test resolution-independent. Thresholds calibrated against the
            # 20-180 deg synthetic sweep and the experimental edge-noise floor;
            # the residual ambiguity is |theta - 90| <~ 5 deg, where the step
            # is at the noise level.
            elif _reflective_kink(
                widest1_top_angle, widest1_bottom_angle,
                widest2_top_angle, widest2_bottom_angle,
                widest_points, n_pts,
            ):
                if display:
                    print("kink at widest point - reflective")
                    print(
                        f"angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                    )
                    print(
                        f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                    )
                final_CPs = widest_points

            # if not reflective, then is the angle under the widest point high enough for it to a high angle system?
            elif 180 - widest1_bottom_angle > 79 or 180 - widest2_bottom_angle > 79:
                # high angle systems that may appear to the code as reflective
                if display:
                    print("looks reflective but is high angle")
                    print(
                        f"angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                    )
                    print(
                        f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                    )
                final_CPs = CP_guesses

            # otherwise use Lfit if unsure
            else:
                # restrict contour to just past CP_guesses
                CP_guess_index1 = _find_point_index(
                    contour, CP_guesses[0], "CPID: CP_guesses[0] for Lfit restriction"
                )
                CP_guess_index2 = _find_point_index(
                    contour, CP_guesses[1], "CPID: CP_guesses[1] for Lfit restriction"
                )
                restricted_contour = contour[
                    max(0, CP_guess_index1 - 4 * n_pts) : CP_guess_index2 + 4 * n_pts
                ]
                final = iterative_Lfit(
                    restricted_contour, Lfit_n_pts, display=display,
                    fallback_points=CP_guesses,
                )
                left_CP, right_CP = final[0][0], final[1][0]
                final_CPs = np.array([left_CP, right_CP])

                if display:
                    print("assumed low angle reflective")
                    print(
                        f"angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                    )
                    print(
                        f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                    )
        elif widest1_top_angle < 65 or widest2_top_angle < 65:
            if display:
                print(
                    f"low angle - angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                )
            final_CPs = widest_points
        # Drops near 75 deg on a reflective substrate measure a top angle just
        # under the cascade's entry gate above, yet show the largest tangent
        # step of all; without this the Lfit below searches near the CP
        # guesses, which sit on the substrate inside the reflection.
        elif (
            65 < widest1_top_angle < 90 - error_margin
            or 65 < widest2_top_angle < 90 - error_margin
        ) and _reflective_kink(
            widest1_top_angle, widest1_bottom_angle,
            widest2_top_angle, widest2_bottom_angle,
            widest_points, n_pts,
        ):
            if display:
                print("kink at widest point - reflective")
                print(
                    f"angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                )
                print(
                    f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                )
            final_CPs = widest_points
        elif (
            65 < widest1_top_angle < 90 - error_margin
            or 65 < widest2_top_angle < 90 - error_margin
        ):
            if display:
                print(
                    f"maybe high angle, Lfit - angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                )
                print(
                    f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                )

            # restrict contour to just past CP_guesses
            CP_guess_index1 = _find_point_index(
                contour, CP_guesses[0], "CPID: CP_guesses[0] for maybe-high-angle Lfit"
            )
            CP_guess_index2 = _find_point_index(
                contour, CP_guesses[1], "CPID: CP_guesses[1] for maybe-high-angle Lfit"
            )
            restricted_contour = contour[
                max(0, CP_guess_index1 - 4 * n_pts) : CP_guess_index2 + 4 * n_pts
            ]
            final = iterative_Lfit(
                restricted_contour, Lfit_n_pts, display=display,
                fallback_points=CP_guesses,
            )
            left_CP, right_CP = final[0][0], final[1][0]
            final_CPs = np.array([left_CP, right_CP])

        else:  # should do the full Lfit approach here
            final_CPs = CP_guesses
            if display:
                print(
                    f"assumed hydrophobic - angles preceding widest point are {widest1_top_angle}, {widest2_top_angle}"
                )
                print(
                    f"angles proceeding widest point are {widest1_bottom_angle}, {widest2_bottom_angle}"
                )
    return final_CPs


def preprocess(img, display=False, blur=True):
    """This code serves as a discrete instance of image preprocessing before contact
    angle fit software is implemented.
    """
    img_processed, longest_contour, drop_guess, info = isolate_contour(
        img, DISPLAY=display, blur=blur
    )

    CPs_chosen = CPID(longest_contour, display=display)

    if display:
        plt.imshow(img_processed)
        plt.plot(longest_contour[:, 0], longest_contour[:, 1], ",")
        plt.plot(
            CPs_chosen[0, 0],
            CPs_chosen[0, 1],
            ".",
            label="Left contact points: " + str(CPs_chosen[0]),
        )
        plt.plot(
            CPs_chosen[1, 0],
            CPs_chosen[1, 1],
            ".",
            label="Right contact points: " + str(CPs_chosen[0]),
        )
        plt.legend()
        plt.show()
        plt.close()

    # define drop profile contour as the contour between contact points
    idx_leftCP = _find_point_index(
        longest_contour,
        CPs_chosen[0],
        "preprocess: left contact point in longest_contour",
    )
    idx_rightCP = _find_point_index(
        longest_contour,
        CPs_chosen[1],
        "preprocess: right contact point in longest_contour",
    )
    # ensure idx_leftCP is smaller than idx_rightCP
    if idx_leftCP > idx_rightCP:
        idx_leftCP, idx_rightCP = idx_rightCP, idx_leftCP

    return img_processed, CPs_chosen, longest_contour


def extract_edges_cv(
    img: np.ndarray, threshold_val=None, return_threshold_value=False, display=False
):
    """
    Give the image and return a list of [x,y] coordinates for the detected edges.
    Ported from preprocessing_old.py to maintain compatibility with existing callers.
    """
    IGNORE_EDGE_MARGIN = 1
    img = img.astype("uint8")
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = img

    if threshold_val is None:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [contour.reshape(contour.shape[0], 2) for contour in contours]
    contours.sort(key=lambda c: cv2.arcLength(c, False), reverse=True)
    drop_profile = contours[0]

    width, height = img.shape[1::-1]
    if not (width < IGNORE_EDGE_MARGIN or height < IGNORE_EDGE_MARGIN):
        mask = (
            (IGNORE_EDGE_MARGIN < drop_profile[:, 0])
            & (drop_profile[:, 0] < width - IGNORE_EDGE_MARGIN)
            & (IGNORE_EDGE_MARGIN < drop_profile[:, 1])
            & (drop_profile[:, 1] < height - IGNORE_EDGE_MARGIN)
        )
        drop_profile = drop_profile[mask]

    output = []
    for coord in drop_profile:
        if list(coord) not in output:
            output.append(list(coord))
    output = np.array(output)

    if return_threshold_value:
        return output, ret
    else:
        return output
