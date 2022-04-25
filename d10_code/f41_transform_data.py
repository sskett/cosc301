import math
import numpy as np
import pandas as pd


def calc_dir_vector(angle):
    rads = {
        0: 0,
        90: math.pi / 2,
        180: math.pi,
        270: 3 * math.pi / 2,
        360: 2 * math.pi
    }
    # TODO: Adjust this to positive to left and neg to right (heading right)
    # Determine quadrant 0 degrees is pos-y axis, rotating clockwise
    if rads[270] < angle <= rads[360]:
        sign = (-1, 1)
        angle = angle - rads[270]
    elif rads[180] < angle <= rads[270]:
        sign = (-1, -1)
        angle = rads[270] - angle
    elif rads[90] <= angle <= rads[180]:
        sign = (1, -1)
        angle = angle - rads[90]
    else:
        sign = (1, 1)
        angle = rads[90] - angle

    dx = math.cos(angle) * sign[0]
    dy = math.sin(angle) * sign[1]

    return np.array([dx, dy])


def convert_to_radians(theta):
    return theta * math.pi / 180


def flip_position(row):
    if row.playDirection == 'left':
        pos = np.array([120 - row.pos[0], 53.3 - row.pos[1]])
    else:
        pos = row.pos
    return pos


def flip_heading(vector):
    return -vector


def flip_orientation(row):
    if row.playDirection == 'left':
        return flip_heading(row.o_vec)
    else:
        return row.o_vec


def flip_direction(row):
    if row.playDirection == 'left':
        return flip_heading(row.dir_vec)
    else:
        return row.dir_vec


def transform_tracking_data(df):
    # Convert angles to radians
    df['o'] = df['o'].apply(convert_to_radians)
    df['dir'] = df['dir'].apply(convert_to_radians)

    # Determine heading vectors
    df['o_vec'] = df['o'].apply(calc_dir_vector)
    df['dir_vec'] = df['dir'].apply(calc_dir_vector)
    df.drop(['o', 'dir'], axis=1, inplace=True)

    # Convert positions to heading right
    df['pos'] = df.apply(flip_position, axis=1)
    df['o_vec'] = df.apply(flip_orientation, axis=1)
    df['dir_vec'] = df.apply(flip_direction, axis=1)

    return df

