#!/usr/bin/env python
# coding=utf-8

"""This code serves as a discrete instance of the Bashforth-Adams fit of the
Bashforth-Adams fit of the Young-Laplace equation, i.e. the BA_fit for
contact angle analysis.

Bashforth-Adams Young-Laplace fit code taken from most recent version on conan -
conan-ML_v1.1/modules/BA_fit.py
This was based on the BA fit of DropPy.
"""

from opendrop_ml.modules.preprocessing.preprocessing import preprocess
from opendrop_ml.utils.enums import FitType

from scipy.spatial import distance
from scipy.integrate import solve_ivp

# from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
import scipy.optimize as opt

# import numba
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from datetime import datetime


def dist(param, points):
    """
    Calculate the total distance from the calculated circle to the points

    :param param:  list of 3 elements with the center of a circle and its
                   radius
    :param points: list of (x, y) points that should be lying on the circle
    :return: the sum of squared errrors from the points on the circle with
             the provided parameters
    """
    *z, r = param
    ar = [(np.linalg.norm(np.array(z) - np.array(point)) - r) ** 2 for point in points]
    return np.sum(ar)


def bashforth_adams(t, y, a, b):
    """
    Rate of change of point on droplet profile with respect to the parametric
    angle ϕ.

    :param t: Parameteric angle ϕ
    :param y: (x, z) point on the droplet profile
    :param a: capillary length in units px
    :param b: radius of curvature at apex in units of px
    :return: rate of change of each x and z with respect to ϕ

    #radio of a and b (b^2/a^2) should be bond number
    """
    x, z = y
    t = t / 180 * np.pi
    # dxdphi = b*x*np.cos(t) / (a**2 * b * x * z + 2 * x - b * np.sin(t))
    # dzdphi = b*x*np.sin(t) / (a**2 * b * x * z + 2 * x - b * np.sin(t))
    dxdphi = b * x * np.cos(t) / ((b / a**2) * x * z + 2 * x - b * np.sin(t))
    dzdphi = b * x * np.sin(t) / ((b / a**2) * x * z + 2 * x - b * np.sin(t))
    return dxdphi, dzdphi


def sim_bashforth_adams(h, a=1, b=1, num=500, all_the_way=False):
    """
    Simulates the full profile of the Bashforth-Adams droplet from the apex

    Starts at x = +-1e-5, z = 0 and integrates to z = h along the curve
    defined by the ``bashforth-adams`` function

    :param h: Height of the droplet in px
    :param a: Capillary length of the fluid
    :param b: Curvature at the apex
    :param num: Number of coordinate points outputted
    :param all_the_way: Boolean to determine whether to stop at z==h or ϕ==180
    :return: List of ϕ and (x, z) coordinates where the solver executed
    """

    def height(t, y, a, b):
        return y[1] - h

    height.terminal = True
    if all_the_way:
        height.terminal = False

    sol_l = solve_ivp(
        bashforth_adams,
        (0, -180),
        (1e-5, 0),
        args=(
            a,
            b,
        ),
        method="BDF",
        t_eval=np.linspace(0, -180, num=num),
        events=height,
    )
    sol_r = solve_ivp(
        bashforth_adams,
        (0, 180),
        (1e-5, 0),
        args=(
            a,
            b,
        ),
        method="BDF",
        t_eval=np.linspace(0, 180, num=num),
        events=height,
    )

    # Order from left contact point to right (left side reversed, then right
    # side forward).  This ensures fit[idx_left:idx_right] spans the full
    # visible curve, which is required for correct RMSE calculation.
    angles = np.hstack((sol_l.t[::-1], sol_r.t)).T
    pred = np.vstack(
        [
            np.hstack((sol_l.y[0][::-1], sol_r.y[0])),
            np.hstack((sol_l.y[1][::-1], sol_r.y[1])),
        ]
    ).T
    if 1:  # DS
        # print('angles: ', angles)
        # Bo = (b*b)/(a*a)
        # print('a is: ',a)
        # print('b is: ',b)
        # print('h is: ',h)
        # *) # Bond number out by a factor of 18 for some reason
        Bo = 18 * (b * b) / (a * a)
    return angles, pred, Bo


def fit_bashforth_adams(data):  # a=0.1,b= 3
    """
    Calculates the best-fit capillary length and curvature at the apex given
    the provided data for the points on the edge of the droplet

    :param data: list of (x, y) points of the droplet edges
    :param a: initial guess of capillary length
    :param b: initial guess of curvature at the apex
    :return: solution structure from scipy.opt.minimize
    """

    def calc_error(h, params):
        """
        Calulate the sum-squared error between the points on the curve and
        the measured data points

        :param h: Height of the droplet in pixels
        :param params: tuple of capillary length and curvature at apex
        :return: sum-squared error between points on the curve and data
        """
        a, b = params

        _, pred, Bo = sim_bashforth_adams(h, a=a, b=b)

        # print('a is: ',a)
        # print('b is: ',b)

        dist = distance.cdist(data, pred)
        return np.linalg.norm(np.min(dist, axis=1))

    h = np.max(data[:, 1])
    b = h / 2
    a = h / 10
    x_0 = (a, b)
    bounds = [[0, 10], [0, 100]]

    optimum = opt.minimize(
        lambda x: calc_error(h, x), x_0, method="Nelder-Mead", options={"disp": False}
    )
    return optimum


def calculate_angle(v1, v2):
    """
    Compute the angle between two vectors of equal length

    :param v1: numpy array
    :param v2: numpy array
    :return: angle between the two vectors v1 and v2 in degrees
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1_u, v2_u)) * 360 / 2 / np.pi


def fit_circle_leastsq(drop):
    """Fit a circle using least-squares (matching reference algorithm).

    Uses scipy.optimize.leastsq and rounds results to integer pixel coordinates,
    which is required for reliable intersection with a pixelated baseline.

    :param drop: array of (x,y) points on the drop contour
    :return: numpy array [cx, cy, r] with integer-rounded values
    """
    x, y = drop[:, 0], drop[:, 1]
    x_m = min(x) + (max(x) - min(x)) / 2
    y_m = min(y) + (max(y) - min(y)) / 2

    def calc_R(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_2, _ = opt.leastsq(f_2, (x_m, y_m))
    xc_2, yc_2 = center_2
    Ri_2 = np.sqrt((x - xc_2) ** 2 + (y - yc_2) ** 2)
    R_2 = Ri_2.mean()

    # Round to integer pixel coordinates for pixelated images
    return np.array([round(xc_2), round(yc_2), round(R_2)])


def fit_circle(points, width=None, start=False):
    """
    Compute the best-fit circle to ``points`` by minimizing ``dist`` from
    changing values of the centerpoint and radius

    :param points: list of (x,y) points lying on the droplet to fit
    :param width: estimated width of the droplet from the crop boundaries
    :param start: boolean flag to determine how many parameters to fit (just
                  radius if True, otherwise radius and centerpoint)
    :return: result structure from scipy.opt.minimize
    """
    if width is None:
        width = np.max(points[:, 0]) - np.min(points[:, 1])

    if start:
        # Try to fit a circle to the points that we have extracted,
        # only varying the radius about the center of all the points
        z = np.mean(points, axis=0)
        res = opt.minimize(lambda x: dist([*z, x], points), width / 2)

        # Get the results
        res["x"] = np.array([*z, res["x"][0]])
    else:
        # Fit this new set of points, using the full set of parameters
        res = opt.minimize(
            lambda x: dist(x, points),
            np.concatenate((np.mean(points, axis=0), [width / 4])),
        )

    return res


def generate_circle_vectors(intersection):
    """
    Using the intersection point with the baseline, compute the vector that
    points tangent to the circle

    :param intersection: (x,y) point on the circle that crosses the baseline
    :return: baseline vector and vector tangent to best-fit circle
    """
    x_t, y_t = intersection
    # For contact angle, want interior angle, so look at vector in
    # negative x direction (this is our baseline)
    v1 = np.array([-1, 0])

    # Now get line tangent to circle at x_t, y_t
    if y_t != 0:
        slope = -x_t / y_t
        v2 = np.array([1, slope])
        v2 = v2 / np.linalg.norm(v2)
        if y_t < 0:
            # We want the interior angle, so when the line is
            # above the origin (into more negative y), look left
            v2 = -v2
    else:
        v2 = np.array([0, 1])

    return v1, v2


def find_intersection(baseline_coeffs, circ_params):
    """
    Compute the intersection points between the best fit circle and best-fit
    baseline.

    For this we rely on several coordinate transformations, first a
    translation to the centerpoint of the circle and then a rotation to give
    the baseline zero-slope.

    :param baseline_coeffs: Numpy array of coefficients to the baseline
                            polynomial
    :param circ_params: centerpoint and radius of best-fit circle
    :return: (x,y) point of intersection between these two shapes
    """
    *z, r = circ_params
    b, m = baseline_coeffs[0:2]
    # Now we need to actually get the points of intersection
    # and the angles from these fitted curves. Rather than brute force
    # numerical solution, use combinations of coordinate translations and
    # rotations to arrive at a horizontal line passing through a circle.
    # First step will be to translate the origin to the center-point
    # of our fitted circle
    # x = x - z[0], y = y - z[1]
    # Circle : x**2 + y**2 = r**2
    # Line : y = m * x + (m * z[0] + b - z[1])
    # Now we need to rotate clockwise about the origin by an angle q,
    # s.t. tan(q) = m
    # Our transformation is defined by the typical rotation matrix
    #   [x;y] = [ [ cos(q) , sin(q) ] ;
    #             [-sin(q) , cos(q) ] ] * [ x ; y ]
    # Circle : x**2 + y**2 = r**2
    # Line : y = (m*z[0] + b[0] - z[1])/sqrt(1 + m**2)
    # (no dependence on x - as expected)

    # With this simplified scenario, we can easily identify the points
    # (x,y) where the line y = B
    # intersects the circle x**2 + y**2 = r**2
    # In our transformed coordinates, only keeping the positive root,
    # this is:

    B = (m * z[0] + b - z[1]) / np.sqrt(1 + m**2)

    if B > r:
        raise ValueError("The circle and baseline do not appear to intersect")
    x_t = np.sqrt(r**2 - B**2)
    y_t = B

    # TODO:// replace the fixed linear baseline with linear
    # approximations near the intersection points

    return x_t, y_t


def save_circle_baseline_debug_plot(profile, baseline_coeffs, circ_params):
    """
    Save an overlay of contour points, fitted circle, and fitted baseline.
    Returns the output path.
    """
    logs_dir = os.path.join(os.path.expanduser("~"), "OpenDrop", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(
        logs_dir,
        f"yl_intersection_debug_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png",
    )

    b, m = baseline_coeffs[0:2]
    cx, cy, r = circ_params

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        profile[:, 0], profile[:, 1], ".", color="tab:blue", alpha=0.7, label="Contour"
    )

    theta = np.linspace(0, 2 * np.pi, num=400)
    x_circle = cx + r * np.cos(theta)
    y_circle = cy + r * np.sin(theta)
    ax.plot(
        x_circle, y_circle, "-", color="tab:green", linewidth=1.5, label="Fitted circle"
    )

    x_min = float(np.min(profile[:, 0]))
    x_max = float(np.max(profile[:, 0]))
    x_line = np.linspace(x_min, x_max, 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, "-", color="tab:red", linewidth=1.5, label="Baseline fit")

    ax.set_title("YL Debug: Contour vs Baseline and Circle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def yl_closest_point(xp, yp, yl_points, display=False):
    """
    xp (float): The x-coordinate of the reference point
    yp (float): The y-coordinate of the reference point
    yl_points (array): The array of x, y coordinates outputted by the YL fit
    display (Boolean): Set to True to output figures and information.

    Returns:
        The distance between the reference point and the YL fit, and
        the coordinates of the closest point on the YL fit.

    """

    x = yl_points[:, 0]
    y = yl_points[:, 1]

    dist = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
    idx = np.argmin(dist)

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


def yl_fit_errors(contour, yl_points, display=False):
    """
    Calculates the minimum distance between a point and a point of the YL fit.

    Parameters:
        contour (array): The array of x, y coordindate points
        yl_points (array): The array of x, y coordinate points of the YL fit
        display (boolean): Set to true to show figures.

    Returns:
        dictionary: The MAE, MSE, RMSE, and maximum error of the contour as compared against the
        YL fit.
    """

    errors = []

    for point in contour:
        dist2edge, edge_point = yl_closest_point(
            point[0], point[1], yl_points, display=display
        )
        errors.append(dist2edge)

    error_measures = {}

    error_measures["MAE"] = sum([abs(error) for error in errors]) / len(errors)
    error_measures["MSE"] = sum([error**2 for error in errors]) / len(errors)
    error_measures["RMSE"] = np.sqrt(sum([error**2 for error in errors]) / len(errors))
    error_measures["Maximum error"] = max(errors)

    return error_measures


def analyze_frame(img, lim=10, fit_type=FitType.BASHFORTH_ADAMS, display=False):
    """This is the function which must be called to perform the BA fit.
    For best results, preprocessing must be perfromed before calling this function.
    """
    # begin with method specific preprocessing of img data
    start_time = time.time()

    img = img.astype("uint8")

    img_processed, CPs_chosen, longest_contour = preprocess(img, display=display)
    idx_l = np.where((longest_contour == CPs_chosen[0]).all(axis=1))[0][0]
    idx_r = np.where((longest_contour == CPs_chosen[1]).all(axis=1))[0][0]
    if idx_l > idx_r:
        idx_l, idx_r = idx_r, idx_l
    profile = longest_contour[idx_l : idx_r + 1]
    CPs = {0: CPs_chosen[0], 1: CPs_chosen[1]}

    # Baseline coefficients [y-intercept, gradient].
    # Compute the true y-intercept (c = y0 - m*x0) so that
    # find_intersection computes the correct perpendicular distance B.
    _m = (CPs[1][1] - CPs[0][1]) / (CPs[1][0] - CPs[0][0])
    _c = CPs[0][1] - _m * CPs[0][0]
    a = [_c, _m]

    # if display:
    #    plt.imshow(img)
    #    plt.title('prepared contour and contact points')
    #    plt.plot(profile[:,0],profile[:,1],'o')
    #    plt.plot(CPs[0][0],CPs[0][1],'yo')
    #    plt.plot(CPs[1][0],CPs[1][1],'yo')
    #    plt.show()
    #    plt.close()

    # define baseline as between the two contact points
    x = np.linspace(CPs[0][1], CPs[1][1], 50)
    y = np.linspace(CPs[0][0], CPs[1][0], 50)
    baseline = np.array([y, x]).T
    # print(baseline)

    # circle object is an array of xy pairs
    circle = profile.astype(dtype="int")

    fit_preprocessing_time = time.time() - start_time
    fit_start_time = time.time()

    # now fit the circle, this will be used as a starting point for the BA fit
    if fit_type in FitType:
        width = max(profile[:, 0])

        res = fit_circle_leastsq(circle)
        *z, r = res

        theta = np.linspace(0, 2 * np.pi, num=500)
        x = z[0] + r * np.cos(theta)
        y = z[1] + r * np.sin(theta)

        points = circle

        res[2] += 1  # to account for if the line is the next pixel over

        try:
            x_t, y_t = find_intersection(a, res)
        except ValueError as e:
            debug_plot_path = save_circle_baseline_debug_plot(profile, a, res)
            raise ValueError(f"{e}. Debug plot saved to: {debug_plot_path}") from e
        # print('x_t is: ',x_t)
        # print('y_t is: ',y_t)

        v1, v2 = generate_circle_vectors([x_t, y_t])

        phi = {i: calculate_angle(v1, v2) for i in ["left", "right"]}
        if fit_type == "circular":
            baseline_width = 2 * x_t

            volume = 2 / 3 * np.pi * r**3 + np.pi * r**2 * y_t - np.pi * y_t**3 / 3

            # Fitted circle
            theta = np.linspace(0, 2 * np.pi, num=100)
            x = z[0] + r * np.cos(theta)
            y = z[1] + r * np.sin(theta)
            fit = np.array([x, y]).T
        else:
            if display:
                # print('points: ',points)
                plt.title("points before near edge sort")
                plt.plot(points[:, 0], points[:, 1])
                plt.show()
                plt.close()
            if 0:  # Get points within 10 pixels of the circle edge
                # not appropriate for high bond numbers but will also work for low angle drops
                points = np.array(
                    [
                        (x, y)
                        for x, y in circle
                        if (x - z[0]) ** 2 + (y - z[1]) ** 2 >= (r - 10) ** 2
                    ]
                )
                if display:
                    # print('points: ',points)
                    plt.title(
                        "points near drop edge\nh="
                        + str(h)
                        + "\nlength="
                        + str(len(points))
                    )
                    plt.plot(points[:, 0], points[:, 1])
                    plt.show()
                    plt.close()
                    # print(points)

            b = min(points[:, 1])
            points[:, 1] = -points[:, 1] + max(points[:, 1])
            center = res[0]  # circle center x-coordinate
            points[:, 0] = points[:, 0] - center
            h = np.max(points[:, 1])
            points = np.vstack([points[:, 0], h - points[:, 1]]).T

            if display:
                print("Running Bashforth-Adams fit...\n")
            cap_length, curv = fit_bashforth_adams(points).x
            thetas, pred, Bo = sim_bashforth_adams(
                h, cap_length, curv, profile.shape[0]
            )

            # theta uses circle-based angle average
            theta = (phi["left"] + phi["right"]) / 2

            R0 = pred[np.argmax(thetas), 0] - pred[np.argmin(thetas), 0]
            baseline_width = R0

            P = 2 * cap_length / curv
            volume = np.pi * R0 * (R0 * h + R0 * P - 2 * np.sin(theta))
            x = pred[:, 0] + center
            y = pred[:, 1] + b  # y-offset to align with image coordinates
            fit = np.array([x, y]).T

            # Find fit point closest to each contact point, then read the
            # angle at that index — this allows independent left/right angles
            dist = np.sqrt((fit[:, 0] - CPs[0][0]) ** 2 + (fit[:, 1] - CPs[0][1]) ** 2)
            idx_left = np.argmin(dist)
            dist = np.sqrt((fit[:, 0] - CPs[1][0]) ** 2 + (fit[:, 1] - CPs[1][1]) ** 2)
            idx_right = np.argmin(dist)

            phi["left"] = -thetas[idx_left]
            phi["right"] = thetas[idx_right]

            fit_time = time.time() - fit_start_time

            # calculate r squared value of fit
            errors = yl_fit_errors(profile, fit[idx_left:idx_right], False)

    else:
        raise ValueError("Unknown fit type! Try another.")

    # drop symmetry r2 score
    # flip
    drop = profile
    drop[:, 1] = -drop[:, 1]
    # find apex index
    apex_indices = np.argwhere(drop[:, 1] == np.max(drop[:, 1]))
    apex_index = int(np.mean(apex_indices))
    # translate so that apex is on x=0, bottom point on y=0
    drop[:, 1] = drop[:, 1] - min(drop[:, 1])
    drop[:, 0] = drop[:, 0] - drop[:, 0][apex_index]

    left = drop[:apex_index]
    right = drop[apex_index:]

    # flip left side, so both facing right
    left[:, 0] = -left[:, 0]
    left = left[::-1]  # reverse order so coords start from apex

    if len(left[:, 0]) > len(right[:, 0]):
        diff = len(left[:, 0]) - len(right[:, 0])
        left = left[diff:]
        sym_errors = yl_fit_errors(left, right, False)
    if len(left[:, 0]) < len(right[:, 0]):
        diff = len(right[:, 0]) - len(left[:, 0])
        right = right[diff:]
        sym_errors = yl_fit_errors(left, right, False)
    else:
        sym_errors = yl_fit_errors(left, right, False)

    if display:
        plt.imshow(img)
        plt.plot(
            fit[idx_left:idx_right, 0], fit[idx_left:idx_right, 1], ",", color="cyan"
        )
        # plt.plot(fit[:,0],fit[:,1],'-', color='black', linewidth=2.5)
        # plt.plot(fit[:,0],fit[:,1],'-', color='white', linewidth=1)
        plt.title(
            "angle of "
            + str(phi["left"])
            + "\nBond number of "
            + str(Bo)
            + "\nfit RMSE of "
            + str(errors["RMSE"])
            + "\ndrop symmetry RMSE "
            + str(sym_errors["RMSE"])
        )
        plt.show()
        plt.close()

    analysis_time = time.time() - start_time

    timings = {}
    timings["method specific preprocessing time"] = fit_preprocessing_time
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return (
        (phi["left"], phi["right"]),
        Bo,
        baseline_width,
        volume,
        fit[idx_left:idx_right],
        baseline,
        errors,
        sym_errors,
        timings,
    )


def yl_fit(profile, lim=10, fit_type=FitType.BASHFORTH_ADAMS, display=False):
    """This is the function which must be called to perform the BA fit.
    For best results, preprocessing must be perfromed before calling this function.
    """
    # begin with method specific preprocessing of img data
    start_time = time.time()
    CPs = [profile[0], profile[-1]]

    # Baseline coefficients [y-intercept, gradient].
    # Compute the true y-intercept (c = y0 - m*x0) so that
    # find_intersection computes the correct perpendicular distance B.
    _m = (CPs[1][1] - CPs[0][1]) / (CPs[1][0] - CPs[0][0])
    _c = CPs[0][1] - _m * CPs[0][0]
    a = [_c, _m]

    # if display:
    #    plt.imshow(img)
    #    plt.title('prepared contour and contact points')
    #    plt.plot(profile[:,0],profile[:,1],'o')
    #    plt.plot(CPs[0][0],CPs[0][1],'yo')
    #    plt.plot(CPs[1][0],CPs[1][1],'yo')
    #    plt.show()
    #    plt.close()

    # define baseline as between the two contact points
    x = np.linspace(CPs[0][1], CPs[1][1], 50)
    y = np.linspace(CPs[0][0], CPs[1][0], 50)
    baseline = np.array([y, x]).T
    # print(baseline)

    # circle object is an array of xy pairs
    circle = profile.astype(dtype="int")

    # now fit the circle, this will be used as a starting point for the BA fit
    if fit_type in FitType:
        width = max(profile[:, 0])

        res = fit_circle_leastsq(circle)
        *z, r = res

        theta = np.linspace(0, 2 * np.pi, num=500)
        x = z[0] + r * np.cos(theta)
        y = z[1] + r * np.sin(theta)

        points = circle

        res[2] += 1  # to account for if the line is the next pixel over

        try:
            x_t, y_t = find_intersection(a, res)
        except ValueError as e:
            debug_plot_path = save_circle_baseline_debug_plot(profile, a, res)
            raise ValueError(f"{e}. Debug plot saved to: {debug_plot_path}") from e
        # print('x_t is: ',x_t)
        # print('y_t is: ',y_t)

        v1, v2 = generate_circle_vectors([x_t, y_t])

        phi = {i: calculate_angle(v1, v2) for i in ["left", "right"]}
        if fit_type == "circular":
            baseline_width = 2 * x_t

            volume = 2 / 3 * np.pi * r**3 + np.pi * r**2 * y_t - np.pi * y_t**3 / 3

            # Fitted circle
            theta = np.linspace(0, 2 * np.pi, num=100)
            x = z[0] + r * np.cos(theta)
            y = z[1] + r * np.sin(theta)
            fit = np.array([x, y]).T
        else:
            if display:
                # print('points: ',points)
                plt.title("points before near edge sort")
                plt.plot(points[:, 0], points[:, 1])
                plt.show()
                plt.close()
            if 0:  # Get points within 10 pixels of the circle edge
                # not appropriate for high bond numbers but will also work for low angle drops
                points = np.array(
                    [
                        (x, y)
                        for x, y in circle
                        if (x - z[0]) ** 2 + (y - z[1]) ** 2 >= (r - 10) ** 2
                    ]
                )
                if display:
                    # print('points: ',points)
                    plt.title(
                        "points near drop edge\nh="
                        + str(h)
                        + "\nlength="
                        + str(len(points))
                    )
                    plt.plot(points[:, 0], points[:, 1])
                    plt.show()
                    plt.close()
                    # print(points)

            b = min(points[:, 1])
            points[:, 1] = -points[:, 1] + max(points[:, 1])
            center = res[0]  # circle center x-coordinate
            points[:, 0] = points[:, 0] - center
            h = np.max(points[:, 1])
            points = np.vstack([points[:, 0], h - points[:, 1]]).T

            if display:
                print("Running Bashforth-Adams fit...\n")
            cap_length, curv = fit_bashforth_adams(points).x
            thetas, pred, Bo = sim_bashforth_adams(
                h, cap_length, curv, profile.shape[0]
            )

            # theta uses circle-based angle average
            theta = (phi["left"] + phi["right"]) / 2

            R0 = pred[np.argmax(thetas), 0] - pred[np.argmin(thetas), 0]
            baseline_width = R0

            P = 2 * cap_length / curv
            volume = np.pi * R0 * (R0 * h + R0 * P - 2 * np.sin(theta))
            x = pred[:, 0] + center
            y = pred[:, 1] + b  # y-offset to align with image coordinates
            fit = np.array([x, y]).T

            # Find fit point closest to each contact point, then read the
            # angle at that index — this allows independent left/right angles
            dist = np.sqrt((fit[:, 0] - CPs[0][0]) ** 2 + (fit[:, 1] - CPs[0][1]) ** 2)
            idx_left = np.argmin(dist)
            dist = np.sqrt((fit[:, 0] - CPs[1][0]) ** 2 + (fit[:, 1] - CPs[1][1]) ** 2)
            idx_right = np.argmin(dist)

            phi["left"] = -thetas[idx_left]
            phi["right"] = thetas[idx_right]

            fit_time = time.time() - start_time

            # calculate r squared value of fit
            errors = yl_fit_errors(profile, fit[idx_left:idx_right], False)

    else:
        raise ValueError("Unknown fit type! Try another.")

    # drop symmetry r2 score
    # flip
    drop = profile.copy()
    drop[:, 1] = -drop[:, 1]
    # find apex index
    apex_indices = np.argwhere(drop[:, 1] == np.max(drop[:, 1]))
    apex_index = int(np.mean(apex_indices))
    # translate so that apex is on x=0, bottom point on y=0
    drop[:, 1] = drop[:, 1] - min(drop[:, 1])
    drop[:, 0] = drop[:, 0] - drop[:, 0][apex_index]

    left = drop[:apex_index]
    right = drop[apex_index:]

    # flip left side, so both facing right
    left[:, 0] = -left[:, 0]
    left = left[::-1]  # reverse order so coords start from apex

    if len(left[:, 0]) > len(right[:, 0]):
        diff = len(left[:, 0]) - len(right[:, 0])
        left = left[diff:]
    if len(left[:, 0]) < len(right[:, 0]):
        diff = len(right[:, 0]) - len(left[:, 0])
        right = right[diff:]

    sym_errors = yl_fit_errors(left, right, False)

    if display:
        plt.plot(profile[:, 0], profile[:, 1], "b")
        plt.plot(fit[:, 0], fit[:, 1], ",", color="black", linewidth=5)
        plt.plot(fit[:, 0], fit[:, 1], ",", color="white", linewidth=2.5)
        plt.title(
            "angle of "
            + str(phi["left"])
            + "\nBond number of "
            + str(Bo)
            + "\nfit RMSE of "
            + str(errors["RMSE"])
            + "\ndrop symmetry RMSE "
            + str(sym_errors["RMSE"])
        )
        plt.show()
        plt.close()

    analysis_time = time.time() - start_time

    timings = {}
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return (
        (phi["left"], phi["right"]),
        Bo,
        baseline_width,
        volume,
        fit[idx_left:idx_right],
        baseline,
        errors,
        sym_errors,
        timings,
    )


if 0:
    from opendrop_ml.utils.os import resource_path

    IMG_PATH = resource_path("../RICOphobic_cropped.png")
    img = cv2.imread(IMG_PATH)
    angles, Bo, baseline_width, volume, fit, baseline, errors, sym_errors, timings = (
        analyze_frame(img, 15, FitType.BASHFORTH_ADAMS, display=False)
    )

    print("angles: ", angles)
    print("Bond number: ", Bo)
    print("baseline_width: ", baseline_width)
    print("volume: ", volume)
    print("errors: ", errors)
    print("sym errors: ", sym_errors)
    print("timings: ", timings)
