import copy
import torch
import random
try:
    import pydiffvg
except:
    import diffvg as pydiffvg


def build_random_path(num_segments, canvas_width, canvas_height, stroke_width=1.0):
    num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
    points = []
    p0 = (random.random(), random.random())
    points.append(p0)
    for j in range(num_segments):
        radius = 0.1
        p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
        p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
        p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
        points.append(p1)
        points.append(p2)
        points.append(p3)
        p0 = p3
    points = torch.tensor(points)
    points[:, 0] *= canvas_width
    points[:, 1] *= canvas_height
    path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                         stroke_width=torch.tensor(stroke_width), is_closed=False)
    return path


def build_translated_path(path, dx, dy):
    pts = copy.deepcopy(path.points)
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pydiffvg.Path(num_control_points=path.num_control_points, points=pts,
                         stroke_width=path.stroke_width, is_closed=False)
