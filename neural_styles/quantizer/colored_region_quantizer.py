import pydiffvg
import bezier
import shapely
from shapely.geometry import Polygon, Point, LineString
from typing import NamedTuple, Tuple, List
from shapely.validation import make_valid
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
import numpy as np
from copy import deepcopy
from shapely.geometry import MultiPolygon
import argparse


def my_render(shapes, shape_groups, size=1024):
    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        1 * size, 1 * size, shapes, shape_groups)
    img = render(1 * size, 1 * size, 2, 2, 0,
                 torch.ones((1 * size, 1 * size, 4)), *scene_args)
    plt.imshow(img.cpu().detach().numpy())


def alpha_color_over(a, b):
    alpha_a, alpha_b = a[3], b[3]
    alpha_z = alpha_a + alpha_b * (1.0 - alpha_a)
    c_a, c_b = a[:3], b[:3]
    c_z = (c_a * alpha_a + c_b * alpha_b * (1.0 - alpha_a)) / alpha_z
    return torch.tensor((*c_z.numpy().tolist(), alpha_z))


class Geometry(NamedTuple):
    index: int
    poly: List[bezier.curve.Curve]
    color: Tuple[float, float, float, float]

    @classmethod
    def from_diffvg(cls, s, sg):
        curves = list()
        index = 0
        ctrl_pts = s.num_control_points.numpy().tolist()
        shape_pts = s.points.numpy().tolist()
        shape_pts = shape_pts + shape_pts[:1]
        for k in ctrl_pts:
            pts = shape_pts[index: index + k + 2]
            xs, ys = zip(*pts)
            curves.append(bezier.curve.Curve([xs, ys], k + 1))
            index += k + 1
        return cls(sg.shape_ids, curves, sg.fill_color)

    def plot(self):
        print([s.nodes for s in self.poly])
        bezier.curved_polygon.CurvedPolygon(*self.poly).plot()

    def to_polygon(self):
        n_ctrl_pts = len(self.poly)
        pts = list()
        for i in range(n_ctrl_pts):
            xs, ys = self.poly[i].evaluate_multi(np.linspace(0.0, 1.0, 30))
            pts.extend(zip(xs, ys))
        return Polygon(pts)

    def to_valid_polygons(self):
        poly = self.to_polygon()
        geom = make_valid(poly)

        try:
            return list(geom.geoms)
        except AttributeError:
            return [geom]

    def to_diffvg(self):
        n_ctrl_pts = len(self.poly)
        pts = list()
        for i in range(n_ctrl_pts):
            xs, ys = self.poly[i].nodes
            pts.extend(list(zip(xs, ys))[:-1])
        return pydiffvg.Path(num_control_points=torch.zeros(n_ctrl_pts) + 2,
                             points=torch.tensor(pts, dtype=torch.float32),
                             stroke_width=torch.tensor(0.0), is_closed=True), \
               pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.index]), fill_color=torch.tensor(self.color),
                                   stroke_color=self.color, use_even_odd_rule=False)

    @classmethod
    def render(cls, geoms):
        shapes, shape_groups = list(), list()
        for g in geoms:
            s, sg = g.to_diffvg()
            shapes.append(s)
            shape_groups.append(sg)
        my_render(shapes, shape_groups)

    @classmethod
    def poly_render(cls, geoms):
        shapes, shape_groups = list(), list()
        index = 0
        for g in geoms:
            _, sg = g.to_diffvg()
            for p in g.to_valid_polygons():
                s = pydiffvg.Polygon(points=torch.tensor(p.exterior.coords),
                                     stroke_width=torch.tensor(0.0), is_closed=True)
                shapes.append(s)
                sg = pydiffvg.ShapeGroup(shape_ids=torch.tensor([index]), fill_color=sg.fill_color,
                                         stroke_color=sg.fill_color, use_even_odd_rule=False)
                shape_groups.append(sg)
                index += 1

        print(len(shapes), len(shape_groups))

        my_render(shapes, shape_groups)

    @classmethod
    def to_color_poly(cls, geoms):
        color_polys = list()
        for g in geoms:
            for po in g.to_valid_polygons():
                color_polys.append(ColorPoly(g.index, po, g.color))
        return color_polys


class ColorPoly(NamedTuple):
    index: int
    shape: Polygon
    color: Tuple[int, int, int, int]

    @classmethod
    def from_collection_file(cls, file_path):
        chunks = list()
        with open(file_path, "r") as f:
            for line in f:
                chunks.append(cls(*line.strip().split(";")))
        return chunks

    @classmethod
    def to_chunks(cls, color_polygons):
        chunks = list()

        for i in range(len(color_polygons)):
            a = color_polygons[i].shape
            current_intersections = list()

            # a - compute intersections with other polys
            for j in range(i + 1, len(color_polygons)):

                b = color_polygons[j].shape

                c = a.intersection(b)
                if c.is_empty or type(c) in [LineString, Point]:
                    continue
                blend = alpha_color_over(color_polygons[i].color, color_polygons[j].color)
                try:
                    for element in c.geoms:
                        current_intersections.append(ColorPoly(j + 1, element, blend))
                except AttributeError:
                    current_intersections.append(ColorPoly(j + 1, c, blend))
            chunks.extend(current_intersections)

            # b - add the main shape minus intersections
            remaining = a
            for intersection in current_intersections:
                try:
                    cleaned_shape = geom_cleaning(intersection.shape)
                    remaining = remaining.difference(cleaned_shape)
                    remaining = geom_cleaning(remaining)
                except shapely.errors.TopologicalError as e:
                    print(remaining, "<>", cleaned_shape)
                    raise e

            try:
                for element in remaining.geoms:
                    if element.is_empty or type(element) in [LineString, Point]:
                        continue
                    chunks.append(ColorPoly(i, element, color_polygons[i].color))
            except AttributeError:
                chunks.append(ColorPoly(i, remaining, color_polygons[i].color))
        return chunks

    @classmethod
    def render(cls, color_polys, canvas_width=500, canvas_height=500, save_to_svg=False):
        shapes, shape_groups = list(), list()
        index = 0
        for cp in color_polys:
            if cp.shape.is_empty or type(cp.shape) == Point:
                continue
            s = pydiffvg.Polygon(points=torch.tensor(cp.shape.exterior.coords),
                                 stroke_width=torch.tensor(1.0), is_closed=True)
            shapes.append(s)
            sg = pydiffvg.ShapeGroup(shape_ids=torch.tensor([index]), fill_color=cp.color,
                                     stroke_color=cp.color, use_even_odd_rule=False)
            shape_groups.append(sg)
            index += 1

        if not save_to_svg:
            my_render(shapes, shape_groups)
        else:
            pydiffvg.save_svg("my_quantized_rez__.svg", canvas_width, canvas_height, shapes, shape_groups)

    def serialize(self):
        poly_repr = self.shape.wkt
        line = ";".join(map(str, [self.index, poly_repr, self.color.numpy().tolist()]))
        return line

    @classmethod
    def serialize_collection(cls, color_polys, file_path):
        with open(file_path, "w") as f:
            for p in color_polys:
                f.write(p.serialize() + "\n")


def is_valid(poly):
    c1 = poly.area > 100
    c2 = (max(poly.bounds) > 0) and (min(poly.bounds) < 499)
    return all([c1, c2])


def unroll_poly(s):
    polygons = list()
    try:
        for element in s.geoms:
            if element.is_empty or type(element) in [LineString, Point]:
                continue
            polygons.extend(unroll_poly(element))
    except AttributeError:
        polygons.append(s)
    return polygons


def geom_cleaning(shape):
    if type(shape) in [Point, LineString]:
        return Polygon()
    valid_shape = make_valid(shape)
    shapes = unroll_poly(valid_shape)
    return MultiPolygon(shapes)


def layer_carving(chunks):
    new_chunks = deepcopy(chunks)

    for i in reversed(range(len(new_chunks))):
        layer = make_valid(new_chunks[i].shape)
        for j in range(len(new_chunks)):
            if j < i:
                current_shape = make_valid(new_chunks[j].shape)
                try:
                    new_chunks[j] = ColorPoly(new_chunks[j].index,
                                              geom_cleaning(current_shape.difference(layer)),
                                              new_chunks[j].color)
                except Exception as e:
                    print(layer, new_chunks[j].shape)
                    raise e

    new_new_chunks = list()
    for chunk in new_chunks:
        new_polys = unroll_poly(chunk.shape)
        new_new_chunks.extend([ColorPoly(chunk.index, s, chunk.color) for s in new_polys])

    return new_new_chunks


def main(f_args, *args, **kwargs):
    print(f_args)

    svg = f_args.path
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(svg)
    my_geometries = list()

    for i, (s, sg) in enumerate(zip(shapes, shape_groups)):
        geo = Geometry.from_diffvg(s, sg)
        my_geometries.append(geo)

    color_polygons = Geometry.to_color_poly(my_geometries)

    chunks = ColorPoly.to_chunks(color_polygons)

    X = [rgb2lab(
        c.color.numpy()[:3] * c.color.numpy()[3] + (0.8 * np.array([1.0, 1.0, 1.0]) * (1.0 - c.color.numpy()[3])))
        for c in chunks]
    W = [c.shape.area for c in chunks]

    kmeans = KMeans(n_clusters=f_args.n_colors, random_state=0, max_iter=1000)
    kmeans.fit(X, sample_weight=W)
    assigned_colors = kmeans.predict(X)

    new_chunks = [ColorPoly(
        c.index, c.shape,
        torch.tensor((*lab2rgb(kmeans.cluster_centers_[assigned_colors[i]]).tolist(), 1.0))) for i, c in
        enumerate(chunks)]

    final_chunks = layer_carving(new_chunks)
    ColorPoly.serialize_collection(final_chunks, f_args.file_name)
    ColorPoly.render(final_chunks, canvas_width, canvas_height, save_to_svg=True)


p = argparse.ArgumentParser()
p.add_argument("--path", type=str)
p.add_argument("--n_colors", default=6, type=int)
p.add_argument("--file_name", default="serialized_rez.txt", type=str)
args = p.parse_known_args()


if __name__ == "__main__":
    main(*args)
