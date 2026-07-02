"""This code serves as a discrete instance of the elipse fit method of
contact angle analysis.

This is base on the circular fit code taken from the most recent version
of conan - conan-ML_cv1.1/modules/select_regions.py"""

from opendrop_ml.modules.preprocessing.preprocessing import preprocess

# from scipy.spatial import distance
# from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse

# import scipy.optimize as opt
# import numba
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time


def rot(a):
    """
    simple rotation matrix in 2D
    """
    return np.array([[+np.cos(a), -np.sin(a)], [+np.sin(a), +np.cos(a)]])


def normalize_data(x, y):
    """
    Normalize coordinate data into approximately [-1, 1] range for stable ellipse fitting.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_center, y_center = np.mean(x), np.mean(y)
    x_scale = max(abs(x.min() - x_center), abs(x.max() - x_center))
    y_scale = max(abs(y.min() - y_center), abs(y.max() - y_center))

    x_norm = (x - x_center) / x_scale
    y_norm = (y - y_center) / y_scale

    return x_norm, y_norm, x_center, y_center, x_scale, y_scale


def denormalize_ellipse(a, x_center, y_center, x_scale, y_scale):
    """
    Convert normalized conic coefficients back to original coordinate space.
    """
    A, B, C, D, E, F = a

    A_prime = A / x_scale**2
    B_prime = B / (x_scale * y_scale)
    C_prime = C / y_scale**2
    D_prime = (
        -2 * A * x_center / x_scale**2
        - B * y_center / (x_scale * y_scale)
        + D / x_scale
    )
    E_prime = (
        -B * x_center / (x_scale * y_scale)
        - 2 * C * y_center / y_scale**2
        + E / y_scale
    )
    F_prime = (
        A * x_center**2 / x_scale**2
        + B * x_center * y_center / (x_scale * y_scale)
        + C * y_center**2 / y_scale**2
        - D * x_center / x_scale
        - E * y_center / y_scale
        + F
    )

    return np.array([A_prime, B_prime, C_prime, D_prime, E_prime, F_prime])


def fit_ellipse(x, y):
    """
    main fit from the original publication:
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    # Fit in normalized space for numerical stability, then map coefficients back.
    x, y, x_center, y_center, x_scale, y_scale = normalize_data(x, y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = +2
    C[2, 0] = +2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(E)
    avec = V[:, n]
    return denormalize_ellipse(avec, x_center, y_center, x_scale, y_scale)


def ell_parameters(a):
    """
    New function substituting the original 3 functions for
    axis, centre and angle.
    We start by noting that the linear term is due to an offset.
    Getting rid of it is equivalent to find the offset.
    Starting with the Eq.
    xT A x + bT x + c = 0 and transforming x -> x - t
    we get a new linear term. By demanding that this term vanishes
    we get the Eq.
    b = (AT + A ) t.
    Hence, an easy way to write down how to get t
    """
    RAD = 180.0 / np.pi
    DEGREE = 1.0 / RAD

    A = np.array([[a[0], a[1] / 2.0], [a[1] / 2.0, a[2]]])
    b = np.array([a[3], a[4]])
    t = np.dot(np.linalg.inv(np.transpose(A) + A), b)
    """
    the transformation changes the constant term, which we need
    for proper scaling
    """
    c = a[5]
    cnew = c - np.dot(t, b) + np.dot(t, np.dot(A, t))
    Anew = A / (-cnew)
    # ~cnew = cnew / (-cnew) ### debug only
    """
    now it is in the form xT A x - 1 = 0
    and we know that A is a rotation of the matrix
        ( 1 / a²   0 )
    B = (            )
        ( 0   1 / b² )
    where a and b are the semi axes of the ellipse
    it is hence A = ST B S
    We note that rotation does not change the eigenvalues, which are
    the diagonal elements of matrix B. Moreover, we note that
    the matrix of eigenvectors rotates B into A
    """
    E, V = np.linalg.eig(Anew)
    """
    so we have
    B = VT A V
    and consequently
    A = V B VT
    where V is of a form as given by the function rot() from above
    """
    # ~B = np.dot( np.transpose(V), np.dot( Anew, V ) ) ### debug only
    phi = np.arccos(V[0, 0])
    """
    checking the sin for changes in sign to detect angles above 180°
    """
    if V[0, 1] < 0:
        phi = 2 * np.pi - phi
    # cw vs ccw and periodicity of pi
    phi = float(phi)
    phi = -phi % np.pi

    #    for i in range(len(E)):
    #        if E[i]<0:
    #            E[i]=0.1

    return np.sqrt(1.0 / E), phi * RAD, -t
    """
    That's it. One might put some additional work/thought in the 180°
    and cw vs ccw thing, as it is a bit messy.
    """


def ellipse_line_intersection(xc, yc, a, b, theta, x0, y0, x1, y1, display=False):
    """
    Finds the intersection between an ellipse and a line defined by two points.

    Parameters:
    xc (float): X-coordinate of the center of the ellipse.
    yc (float): Y-coordinate of the center of the ellipse.
    a (float): Length of the semi-major axis of the ellipse.
    b (float): Length of the semi-minor axis of the ellipse.
    theta (float): Angle of rotation of the ellipse in radians.
    x0 (float): X-coordinate of the first point along the line.
    y0 (float): Y-coordinate of the first point along the line.
    x1 (float): X-coordinate of the second point along the line.
    y1 (float): Y-coordinate of the second point along the line.
    display (Boolean): Set to True to show figures

    Returns:
    tuple: A tuple containing two tuples, where each inner tuple contains the X and Y coordinates of the intercept and the gradient of the ellipse at the intercept.

    """
    # Plot the ellipse and the baseline
    if display:
        ell = Ellipse(
            [xc, yc],
            2 * a,
            2 * b,
            theta,
            facecolor=(1, 0, 0, 0),
            edgecolor=(1, 0, 0, 1),
        )
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.add_patch(ell)
        ax.plot(x0, y0, "yo")
        ax.plot(x1, y1, "go")
        ax.plot()
        plt.title("Drawn ellipse with line points in green and yellow")
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()

    # Calculate the slope and intercept of the line
    m = (y1 - y0) / (x1 - x0)
    c = y0 - m * x0

    # equation for ellipse in quadratic coefficient form
    A = np.sin(theta) ** 2 / b**2 + np.cos(theta) ** 2 / a**2
    B = (
        -2 * np.sin(theta) * np.cos(theta) / b**2
        + 2 * np.sin(theta) * np.cos(theta) / a**2
    )
    C = np.cos(theta) ** 2 / b**2 + np.sin(theta) ** 2 / a**2
    D = (
        -2 * xc * np.sin(theta) ** 2 / b**2
        + 2 * yc * np.sin(theta) * np.cos(theta) / b**2
        - 2 * xc * np.cos(theta) ** 2 / a**2
        - 2 * yc * np.sin(theta) * np.cos(theta) / a**2
    )
    E = (
        2 * xc * np.sin(theta) * np.cos(theta) / b**2
        - 2 * yc * np.cos(theta) ** 2 / b**2
        - 2 * xc * np.sin(theta) * np.cos(theta) / a**2
        - 2 * yc * np.sin(theta) ** 2 / a**2
    )
    F = (
        -1
        + xc**2 * np.sin(theta) ** 2 / b**2
        - 2 * xc * yc * np.sin(theta) * np.cos(theta) / b**2
        + yc**2 * np.cos(theta) ** 2 / b**2
        + xc**2 * np.cos(theta) ** 2 / a**2
        + 2 * xc * yc * np.sin(theta) * np.cos(theta) / a**2
        + yc**2 * np.sin(theta) ** 2 / a**2
    )

    # sub (mx + b) in for y, expand and simplify
    A_new = A + B * m + C * m**2
    B_new = B * c + 2 * C * m * c + D + E * m
    C_new = C * c**2 + E * c + F

    # Calculate the discriminant of the quadratic equation
    disc = B_new**2 - 4 * A_new * C_new

    # If the discriminant is negative, there are no intercepts
    if disc < 0:
        if display:
            print("The line does not intersect the ellipse.")
        return None

    # If the discriminant is zero, there is one intercept
    elif disc == 0:
        x_int = -B_new / (2 * A_new)
        y_int = m * x_int + c
        x_int_rot = x_int * np.cos(theta) - y_int * np.sin(theta)
        y_int_rot = x_int * np.sin(theta) + y_int * np.cos(theta)
        grad = -(np.cos(theta) ** 2 * x_int_rot) / (np.sin(theta) ** 2 * y_int_rot)

        return ((x_int, y_int, grad),)

    # If the discriminant is positive, there are two intercepts
    else:
        x_int_1 = (-B_new + np.sqrt(disc)) / (2 * A_new)
        x_int_2 = (-B_new - np.sqrt(disc)) / (2 * A_new)
        y_int_1 = m * x_int_1 + c
        y_int_2 = m * x_int_2 + c

        grad_1 = -(
            2
            * ((-xc + x_int_1) * np.sin(theta) + (-yc + y_int_1) * np.cos(theta))
            * np.sin(theta)
            / b**2
            + 2
            * ((-xc + x_int_1) * np.cos(theta) + (-yc + y_int_1) * np.sin(theta))
            * np.cos(theta)
            / a**2
        ) / (
            2
            * ((-xc + x_int_1) * np.sin(theta) + (-yc + y_int_1) * np.cos(theta))
            * np.cos(theta)
            / b**2
            + 2
            * ((-xc + x_int_1) * np.cos(theta) + (-yc + y_int_1) * np.sin(theta))
            * np.sin(theta)
            / a**2
        )
        grad_2 = -(
            2
            * ((-xc + x_int_1) * np.sin(theta) + (-yc + y_int_2) * np.cos(theta))
            * np.sin(theta)
            / b**2
            + 2
            * ((-xc + x_int_2) * np.cos(theta) + (-yc + y_int_2) * np.sin(theta))
            * np.cos(theta)
            / a**2
        ) / (
            2
            * ((-xc + x_int_2) * np.sin(theta) + (-yc + y_int_2) * np.cos(theta))
            * np.cos(theta)
            / b**2
            + 2
            * ((-xc + x_int_2) * np.cos(theta) + (-yc + y_int_2) * np.sin(theta))
            * np.sin(theta)
            / a**2
        )

        return ((x_int_1, y_int_1, grad_1), (x_int_2, y_int_2, grad_2))


def ellipse_closest_point(xp, yp, xc, yc, a, b, th, n=1000, display=False):
    """
    xp (float): The x-coordinate of the reference point
    yp (float): The y-coordinate of the reference point
    xc (float): The x-coordinate of the ellipse's center.
    yc (float): The y-coordinate of the ellipse's center.
    a (float): The semi-major axis length of the ellipse.
    b (float): The semi-minor axis length of the ellipse.
    th (float): The rotation angle of the ellipse in degrees.
    n (int): The number of discrete points used to draw the ellipse.
    display (Boolean): Set to True to output figures and information.

    Returns:
        The distance between the reference point and the ellipse edge, and
        the coordinates of the closest point on the ellipse edge.

    """

    t = np.linspace(0, 2 * np.pi, n)

    x = (
        xc
        + a * np.cos(t) * np.cos(np.deg2rad(th))
        - b * np.sin(t) * np.sin(np.deg2rad(th))
    )
    y = (
        yc
        + a * np.cos(t) * np.sin(np.deg2rad(th))
        + b * np.sin(t) * np.cos(np.deg2rad(th))
    )

    # tRandom = np.random.randint(n)

    # xp = x[tRandom]
    # yp = y[tRandom]

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
        plt.title("Ellipse, Point, and Zeros")

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


def ellipse_fit_errors(contour, h, k, a, b, theta, display):
    """
    Calculates the minimum distance between a point and the edge of a rotated and translated ellipse.

    Parameters:
        contour (array): The array of x, y coordindate points
        h (float): The x-coordinate of the ellipse's center.
        k (float): The y-coordinate of the ellipse's center.
        a (float): The semi-major axis length of the ellipse.
        b (float): The semi-minor axis length of the ellipse.
        theta (float): The rotation angle of the ellipse in degrees.
        display (boolean): Set to true to show figures.

    Returns:
        dictionary: The MAE, MSE, RMSE, and maximum error of the contour as compared against the
        fitted ellipse.
    """

    errors = []

    for point in contour:
        dist2edge, edge_point = ellipse_closest_point(
            point[0], point[1], h, k, a, b, theta
        )
        errors.append(dist2edge)

    error_measures = {}

    error_measures["MAE"] = sum([abs(error) for error in errors]) / len(errors)
    error_measures["MSE"] = sum([error**2 for error in errors]) / len(errors)
    error_measures["RMSE"] = np.sqrt(sum([error**2 for error in errors]) / len(errors))
    error_measures["Maximum error"] = max(errors)

    return error_measures


def ellipse_fit_img(img, display=False):
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

    # define baseline as between the two contact points

    x = drop[:, 0]
    y = drop[:, 1]
    rise = CPs[1][1] - CPs[0][1]
    run = CPs[1][0] - CPs[0][0]
    slope = rise / run
    baseline = [(CPs[0][0], CPs[0][1]), slope]
    c = CPs[0][1] - (slope * CPs[0][0])
    baseline_x = np.linspace(1, img.shape[1], 100)
    baseline_y = slope * baseline_x + c

    # timers
    fit_preprocessing_time = time.time() - start_time
    fit_start_time = time.time()

    # fit
    avec = fit_ellipse(x, y)
    (a, b), phi_deg, t = ell_parameters(avec)
    ellipse_params = np.array([a, b, phi_deg, t[0], t[1]], dtype=float)
    if (not np.all(np.isfinite(ellipse_params))) or a <= 0 or b <= 0:
        fit_time = time.time() - fit_start_time
        analysis_time = time.time() - start_time
        timings = {}
        timings["method specific preprocessing time"] = fit_preprocessing_time
        timings["fit time"] = fit_time
        timings["analysis time"] = analysis_time
        errors = (
            f"ellipse_fit: invalid ellipse parameters "
            f"(a={a}, b={b}, phi={phi_deg}, center={t})"
        )
        if display:
            print(errors)
        return [], [], t, (a, b), phi_deg, errors, timings

    ell = Ellipse(
        t, 2 * a, 2 * b, phi_deg, facecolor=(1, 0, 0, 0), edgecolor=(1, 0, 0, 1)
    )

    if display:
        print("centre points: " + str(t))
        print("a and b: " + str(a) + ", " + str(b))
        print("phi (°): " + str(phi_deg))

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.add_patch(ell)
        ax.scatter(x, y)
        ax.plot(baseline_x, baseline_y)
        ax.plot(CPs[0][0], CPs[0][1], "yo")
        ax.plot(CPs[1][0], CPs[1][1], "go")
        ax.plot()
        plt.title("Fitted ellipse")
        plt.imshow(img)
        plt.show()
        plt.close()

    # Find intercepts, and gradients at intercepts
    outputs = ellipse_line_intersection(
        t[0],
        t[1],
        a,
        b,
        math.radians(phi_deg),
        CPs[0][0],
        CPs[0][1],
        CPs[1][0],
        CPs[1][1],
        display=display,
    )
    if outputs is None:
        mid_x = (CPs[0][0] + CPs[1][0]) / 2
        mid_y = (CPs[0][1] + CPs[1][1]) / 2
        CPs_midpoint = np.array([mid_x, mid_y])

        gap, closest = ellipse_closest_point(
            CPs_midpoint[0],
            CPs_midpoint[1],
            t[0],
            t[1],
            a,
            b,
            phi_deg,
            n=1000,
            display=False,
        )
        if display:
            print("gap between CPs_midpoint and closest point of ellipse: ", gap)
        if gap < 2:  # in case ellipse fit misses intercept
            left = closest
            right = closest
            outputs = [[left[0], left[1], 0], [right[0], right[1], 0]]
        else:
            fit_time = time.time() - fit_start_time
            analysis_time = time.time() - start_time
            timings = {}
            timings["method specific preprocessing time"] = fit_preprocessing_time
            timings["fit time"] = fit_time
            timings["analysis time"] = analysis_time
            errors = (
                "ellipse_fit: no line-ellipse intersection and fallback gap too large"
            )
            return [], [], t, (a, b), phi_deg, errors, timings
    else:
        left, right = list(sorted(outputs, key=lambda x: x[0]))

    m_left, m_right = left[2], right[2]
    intercepts = [[left[0], left[1]], [right[0], right[1]]]

    CA = []
    for output in outputs:
        first = True
        m = output[2] - slope

        if output[1] > t[1] and first == True:  # high CA angle left
            # CA.append(math.degrees(np.pi+np.arctan(m)))
            CA.append(180 - abs(math.degrees(np.arctan(m))))
        elif output[1] > t[1] and first == False:  # high CA angle right
            CA.append(abs(math.degrees(np.arctan(m))))
        elif output[1] < t[1] and first == True:  # low CA angle left
            CA.append(abs(np.rad2deg(np.arctan(m))))
        elif output[1] < t[1] and first == False:  # low CA angle right
            CA.append(abs(np.rad2deg(np.pi + np.arctan(m))))

        first = False

    # if CA[0]!=CA[1]:
    #    CA[1] = 180 - CA[1]

    fit_time = time.time() - fit_start_time

    # using MAE, MSE, RMSE, max_error as error measure
    errors = ellipse_fit_errors(drop, t[0], t[1], a, b, phi_deg, display)

    analysis_time = time.time() - start_time

    timings = {}
    timings["method specific preprocessing time"] = fit_preprocessing_time
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return CA, intercepts, t, (a, b), phi_deg, errors, timings


def ellipse_fit(drop, display=False):
    """Call this function to perform the ellipse fit.
    For best results, peprocessing must be done before calling this function.

    Make sure that the drop coordinate array consists of float values.
    """

    # begin with method specific preprocessing of img data
    start_time = time.time()
    CPs = [drop[0], drop[-1]]

    # define baseline as between the two contact points

    x, y = drop[:, 0], drop[:, 1]
    rise = CPs[1][1] - CPs[0][1]
    run = CPs[1][0] - CPs[0][0]
    slope = rise / run
    baseline = [(CPs[0][0], CPs[0][1]), slope]
    c = CPs[0][1] - (slope * CPs[0][0])
    baseline_x = np.linspace(1, max(drop[:, 0]), 100)
    baseline_y = slope * baseline_x + c

    # Center estimates - it doesn't even take these
    # x estimate is where between the lowest and highest points of the top section for a hydrophobic drop
    x_m = min(x) + (max(x) - min(x)) / 2
    # for full contour, y estimate is the halfway between max y and min y
    y_m = min(y) + ((max(y) - min(y)) / 2)

    # fit
    avec = fit_ellipse(x, y)
    (a, b), phi_deg, t = ell_parameters(avec)
    ellipse_params = np.array([a, b, phi_deg, t[0], t[1]], dtype=float)
    if (not np.all(np.isfinite(ellipse_params))) or a <= 0 or b <= 0:
        fit_time = time.time() - start_time
        analysis_time = time.time() - start_time
        timings = {}
        timings["fit time"] = fit_time
        timings["analysis time"] = analysis_time
        errors = (
            f"ellipse_fit: invalid ellipse parameters "
            f"(a={a}, b={b}, phi={phi_deg}, center={t})"
        )
        if display:
            print(errors)
        return [], [], t, (a, b), phi_deg, errors, timings

    ell = Ellipse(
        t,
        2 * a,
        2 * b,
        phi_deg,
        facecolor=(1, 0, 0, 0),
        edgecolor=(1, 0, 0, 1),
        label="Fitted ellipse",
    )

    if display:
        print("drop contour: ", drop)
        print("centre points: " + str(t))
        print("a and b: " + str(a) + ", " + str(b))
        print("phi (°): " + str(phi_deg))

        if 1:  # plot
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.add_patch(ell)
            ax.scatter(x, y, label="contour")
            ax.plot(baseline_x, baseline_y, label="baseline")
            ax.plot(CPs[0][0], CPs[0][1], "yo", label="left contact point")
            ax.plot(CPs[1][0], CPs[1][1], "go", label="right contact point")
            ax.plot()
            ax.legend()
            plt.title("Fitted ellipse")
            plt.axis("equal")
            plt.show()
            plt.close()

    # Find intercepts, and gradients at intercepts
    outputs = ellipse_line_intersection(
        t[0],
        t[1],
        a,
        b,
        math.radians(phi_deg),
        CPs[0][0],
        CPs[0][1],
        CPs[1][0],
        CPs[1][1],
        display=display,
    )
    if outputs is None:
        mid_x = (CPs[0][0] + CPs[1][0]) / 2
        mid_y = (CPs[0][1] + CPs[1][1]) / 2
        CPs_midpoint = np.array([mid_x, mid_y])

        gap, closest = ellipse_closest_point(
            CPs_midpoint[0],
            CPs_midpoint[1],
            t[0],
            t[1],
            a,
            b,
            phi_deg,
            n=1000,
            display=False,
        )
        if display:
            print("gap between CPs_midpoint and closest point of ellipse: ", gap)
        if gap < 2:  # in case ellipse fit misses intercept
            left = closest
            right = closest
            outputs = [[left[0], left[1], 0], [right[0], right[1], 0]]
        else:
            fit_time = time.time() - start_time
            analysis_time = time.time() - start_time
            timings = {}
            timings["fit time"] = fit_time
            timings["analysis time"] = analysis_time
            errors = (
                "ellipse_fit: no line-ellipse intersection and fallback gap too large"
            )
            return [], [], t, (a, b), phi_deg, errors, timings
    else:
        left, right = list(sorted(outputs, key=lambda x: x[0]))

    m_left, m_right = left[2], right[2]
    intercepts = [[left[0], left[1]], [right[0], right[1]]]

    CA = []
    for output in outputs:
        first = True
        m = output[2] - slope

        if output[1] > t[1] and first == True:  # high CA angle left
            # CA.append(math.degrees(np.pi+np.arctan(m)))
            CA.append(180 - abs(math.degrees(np.arctan(m))))
        elif output[1] > t[1] and first == False:  # high CA angle right
            CA.append(abs(math.degrees(np.arctan(m))))
        elif output[1] < t[1] and first == True:  # low CA angle left
            CA.append(abs(np.rad2deg(np.arctan(m))))
        elif output[1] < t[1] and first == False:  # low CA angle right
            CA.append(abs(np.rad2deg(np.pi + np.arctan(m))))

        first = False

    # if CA[0]!=CA[1]:
    #    CA[1] = 180 - CA[1]

    fit_time = time.time() - start_time

    try:  # using MAE, MSE, RMSE, max_error as error measure
        errors = ellipse_fit_errors(drop, t[0], t[1], a, b, phi_deg, display)
    except:
        errors = "something went wrong fitting the ellipse..."
        print(errors)
    analysis_time = time.time() - start_time

    timings = {}
    timings["fit time"] = fit_time
    timings["analysis time"] = analysis_time

    return CA, intercepts, t, (a, b), phi_deg, errors, timings


if 0:
    from opendrop_ml.utils.os import resource_path

    IMG_PATH = resource_path("../RICOphobic_cropped.png")
    img = cv2.imread(IMG_PATH)

    angles, intercepts, center, (a, b), theta, errors, timings = ellipse_fit_img(
        img, display=True
    )
    print("angles: ", angles)
    print("intercepts: ", intercepts)
    print("center coordinates: ", center)
    print("a: ", a)
    print("b: ", b)
    print("angle of ellipse rotation: ", theta)
    print("errors: ", errors)
    print("timings: ", timings)
