# -*- coding: utf-8 -*-


def preprocessing_image_rgb(x):
    # define mean and std values
    mean = [87.845, 96.965, 103.947]
    std = [23.657, 16.474, 13.793]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


def preprocessing_image_ms(x):
    # define mean and std values
    mean = [
        1353.036,
        1116.468,
        1041.475,
        945.344,
        1198.498,
        2004.878,
        2376.699,
        2303.738,
        732.957,
        12.092,
        1818.820,
        1116.271,
        2602.579,
    ]
    std = [
        65.479,
        154.008,
        187.997,
        278.508,
        228.122,
        356.598,
        456.035,
        531.570,
        98.947,
        1.188,
        378.993,
        303.851,
        503.181,
    ]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x
