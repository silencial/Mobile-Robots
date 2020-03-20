"""

Dubins path planner sample code

authored by Atsushi Sakai(@Atsushi_twi)
modified by Aditya Vamsikrishna(adityavk@uw.edu) and Gilwoo Lee (gilwoo@uw.edu)

See https://en.wikipedia.org/wiki/Dubins_path for Dubins path

"""
import math
import IPython
import numpy as np
import matplotlib.pyplot as plt

show_animation = True


def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)


def pi_2_pi(angle):
    while (angle >= math.pi):
        angle = angle - 2.0 * math.pi

    while (angle <= -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def LSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d*d) - (2*c_ab) + (2 * d * (sa-sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(beta - tmp1)

    return t, p, q, mode


def RSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d*d) - (2*c_ab) + (2 * d * (sb-sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode


def LSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d*d) + (2*c_ab) + (2 * d * (sa+sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode


def RSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d*d) - 2 + (2*c_ab) - (2 * d * (sa+sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def RLR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d*d + 2.0*c_ab + 2.0 * d * (sa-sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def LRL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6.0 - d*d + 2.0*c_ab + 2.0 * d * (-sa + sb)) / 8.0
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p/2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(dx, dy, eyaw, curvature):
    # nomalize
    D = math.sqrt(dx**2.0 + dy**2.0)
    d = D * curvature
    #  print(dx, dy, D, d)

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(-theta)
    beta = mod2pi(eyaw - theta)
    #  print(theta, alpha, beta, d)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]

    bcost = float("inf")
    bt, bp, bq, bmode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if bcost > cost:
            bt, bp, bq, bmode = t, p, q, mode
            bcost = cost

    px, py, pyaw = generate_course([bt, bp, bq], bmode, curvature)
    return px, py, pyaw, bmode, bcost


def dubins_path_planning(start, end, curvature):
    """
    Dubins path plannner

    input:
        start: sx, xy, syaw
            sx x position of start point [m]
            sy y position of start point [m]
            syaw yaw angle of start point [rad]
        end: ex, ey, eyaw
            ex x position of end point [m]
            ey y position of end point [m]
            eyaw yaw angle of end point [rad]
        curvature curvature [1/m]

    output:
        px
        py
        pyaw
        length

    """
    sx, sy, syaw = start[0], start[1], start[2]
    ex, ey, eyaw = end[0], end[1], end[2]

    ex = ex - sx
    ey = ey - sy

    lex = math.cos(syaw) * ex + math.sin(syaw) * ey
    ley = -math.sin(syaw) * ex + math.cos(syaw) * ey
    leyaw = eyaw - syaw

    lpx, lpy, lpyaw, mode, clen = dubins_path_planning_from_origin(lex, ley, leyaw, curvature)

    px = [math.cos(-syaw) * x + math.sin(-syaw) * y + sx for x, y in zip(lpx, lpy)]
    py = [-math.sin(-syaw) * x + math.cos(-syaw) * y + sy for x, y in zip(lpx, lpy)]
    pyaw = [pi_2_pi(iyaw + syaw) for iyaw in lpyaw]

    ppx, ppy, ppyaw, pclen = process_dubins(sx, sy, syaw, px, py, pyaw, clen)

    path = np.array([ppx, ppy]).transpose()
    diff = np.diff(path, axis=0)
    length = np.linalg.norm(diff, axis=1)
    path_length = np.sum(length)
    return ppx, ppy, ppyaw, path_length


def path_length(s, e, c):
    px, py, pyaw, cost = dubins_path_planning(s, e, c)
    return cost


def generate_course(length, mode, curvature):

    px = [0.0]
    py = [0.0]
    pyaw = [0.0]

    for m, l in zip(mode, length):
        pd = 0.0
        if m == "S":
            d = 0.3 * curvature
        else:  # turning couse
            d = math.radians(3.0)

        while pd < abs(l - d):
            #  print(pd, l)
            px.append(px[-1] + d / curvature * math.cos(pyaw[-1]))
            py.append(py[-1] + d / curvature * math.sin(pyaw[-1]))

            if m == "L":  # left turn
                pyaw.append(pyaw[-1] + d)
            elif m == "S":  # Straight
                pyaw.append(pyaw[-1])
            elif m == "R":  # right turn
                pyaw.append(pyaw[-1] - d)
            pd += d
        else:
            d = l - pd
            px.append(px[-1] + d / curvature * math.cos(pyaw[-1]))
            py.append(py[-1] + d / curvature * math.sin(pyaw[-1]))

            if m == "L":  # left turn
                pyaw.append(pyaw[-1] + d)
            elif m == "S":  # Straight
                pyaw.append(pyaw[-1])
            elif m == "R":  # right turn
                pyaw.append(pyaw[-1] - d)
            pd += d

    return px, py, pyaw


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
  Plot arrow
  """

    if isinstance(x, list) or isinstance(x, np.ndarray):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def process_dubins(startx, starty, enda, px, py, pa, cost):
    '''
    Naive processing of the output
    Ensuring there are no 2pi rotations due to numerical issues
    '''
    pcost = cost
    eps = 1e-6
    for i in range(1, len(px) - 1):
        check1 = abs(px[i] - startx) < eps
        check2 = abs(py[i] - starty) < eps
        check3 = abs(pa[i] - enda) < eps
        if check1 and check2 and check3:
            return px[i:], py[i:], pa[i:], cost - math.radians(360.0)

    return px, py, pa, cost


def main():
    print("Dubins path planner sample start!!")

    start = [50, 32, math.radians(0)]
    end = [55, 24, math.radians(90)]
    curvature = 0.5

    px, py, pyaw, clen = dubins_path_planning(start, end, curvature)

    if show_animation:
        plt.plot(px, py)

        # plotting
        plot_arrow(start[0], start[1], start[2], fc='r')
        plot_arrow(end[0], end[1], end[2], fc='b')

        plt.grid(True)
        plt.axis("equal")
        plt.show()

        IPython.embed()


if __name__ == '__main__':
    main()
