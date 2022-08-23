import pydiffvg
import bezier
import copy
import shapely
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, MultiLineString, MultiPolygon
from typing import NamedTuple, Tuple, List
from shapely.validation import make_valid
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
import numpy as np
from rtree import index
from copy import deepcopy
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
    depth: int = 0

    @classmethod
    def from_collection_file(cls, file_path):
        chunks = list()
        with open(file_path, "r") as f:
            for line in f:
                chunks.append(cls(*line.strip().split(";")))
        return chunks

    @classmethod
    def from_diffvg(cls, shapes, shape_groups):
        array = list()
        for s, sg in zip(shapes, shape_groups):
            pts = s.points.numpy().tolist()
            polys = geom_cleaning(make_valid(Polygon(pts + [pts[0]])))
            for p in unroll_poly(polys):
                array.append(cls(sg.shape_ids, p, sg.fill_color))
        return array

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
    def to_chunks_v2(cls, my_color_polys, max_depth=10):
        debug_index = -1
        done_index = set()

        def shape_update(poly, new_shape):
            if new_shape.area < 0.5:
                new_shape = Polygon()
            else:
                new_shape = geom_cleaning(new_shape)
            return cls(poly.index, new_shape, poly.color, poly.depth)

        my_chunks = deepcopy(my_color_polys)
        priority_queue = sorted(list(range(len(my_color_polys))), key=lambda i: my_color_polys[i].index)
        iter_ = 0
        print("total_blocks : ", len(my_chunks))

        while len(priority_queue) > 0:
            iter_ += 1
            # print("Queue size : ", len(priority_queue))
            i = priority_queue.pop()
            a = my_chunks[i].shape
            current_intersections = list()
            if my_chunks[i].depth > max_depth:  # first order intersection only
                continue
            if a.is_empty:
                continue
            # if my_chunks[i].color[3] < 0.01: # Layer is transparent
            #    continue

            if debug_index == my_chunks[i].index.numpy():
                print(f"i is {debug_index} : ", my_chunks[i].shape.area)

            current_intersections = list()
            for j in range(len(my_chunks)):
                if i == j:
                    continue
                if debug_index == my_chunks[j].index.numpy()[0]:
                    print(f"j is {debug_index} : ", my_chunks[j].shape.area)
                if (i, j) in done_index or (j, i) in done_index:  # Intersection already processed
                    continue
                if j in done_index:  # Higher up layer should been 100% done
                    continue
                if a.is_empty:
                    break
                cp = my_chunks[j]
                if cp.shape.is_empty:
                    continue
                # if my_chunks[j].color[3] < 0.01:
                #    continue
                try:
                    intersection = a.intersection(cp.shape)
                except Exception as e:
                    print("---> ", i, " ", j, " : ", my_chunks[i].shape, " ", my_chunks[j].shape)
                    raise e
                if intersection.is_empty or intersection.area < 0.5:
                    continue
                else:
                    if debug_index == my_chunks[j].index.numpy()[0]:
                        print(f"j intersection  is {debug_index} : ", intersection.area)
                        print(my_chunks[i].shape)
                        print(my_chunks[j].shape)
                    if debug_index == my_chunks[i].index.numpy()[0]:
                        print(f"i intersection  is {debug_index} : ", intersection.area)
                        print(my_chunks[i].shape)
                        print(my_chunks[j].shape)
                        print("\n")

                    current_intersections.append(intersection)
                    # print(" ", intersection.area, " ----> ", intersection)
                    # Remove intersection from secondary poly
                    my_chunks[j] = shape_update(my_chunks[j], my_chunks[j].shape.difference(my_chunks[i].shape))
                    # Add intersection as a new chunk
                    new_index = len(my_chunks)
                    blend = alpha_color_over(my_chunks[i].color, my_chunks[j].color)
                    my_chunks.append(cls(my_chunks[i].index, geom_cleaning(intersection),
                                         blend, my_chunks[i].depth + 1))
                    # This intersection will be the next items processed after the end of this loop
                    priority_queue.append(new_index)
                    done_index.add((new_index, j))

            # Remove intersection from main poly
            if len(current_intersections) > 0:
                global_intersection = current_intersections[0]
                for intersection in current_intersections[1:]:
                    global_intersection = global_intersection.union(intersection)
                shape_minus_intersection = my_chunks[i].shape.difference(global_intersection)
                if debug_index == my_chunks[i].index.numpy()[0]:
                    print(f"global intersection of {i} : {global_intersection.wkt} \n\n")
                    print(f"remains of {i} : {shape_minus_intersection.wkt} \n\n")
                    print(f"final : {shape_update(my_chunks[i], shape_minus_intersection).shape.wkt}")
                my_chunks[i] = shape_update(my_chunks[i], shape_minus_intersection)
            done_index.add(i)

        return my_chunks

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


def unroll_poly(s):
    polygons = list()
    try:
        for element in s.geoms:
            if element.is_empty or type(element) in [LineString, Point, MultiLineString] or element.area < 1.0:
                continue
            polygons.extend(unroll_poly(element))
    except AttributeError:
        simplified = s.buffer(-0.1).buffer(0.1)
        if type(simplified) in [MultiPolygon, GeometryCollection]:
            polygons.extend(unroll_poly(simplified))
        else:
            polygons.append(simplified)
    return polygons


def geom_cleaning(shape):
    if type(shape) in [Point, LineString, MultiLineString] or shape.area < 1:
        return Polygon()
    if type(shape) == GeometryCollection:
        polys = list(filter(lambda x: type(x) in [Polygon, MultiPolygon] and x.area > 0.5, shape.geoms))
        return GeometryCollection(polys)
    if not shape.is_valid:
        shape = make_valid(shape)
    shapes = list(filter(lambda x: type(x) == Polygon and x.area > 0.5, unroll_poly(shape)))
    if len(shapes) > 1:
        return MultiPolygon(shapes)
    elif len(shapes) == 0:
        return Polygon()
    else:
        return shapes[0]


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


def layer_carving_v2(new_chunks):
    elements = sorted(copy.deepcopy(new_chunks), key=lambda x: x.index.numpy()[0])

    my_index = index.Index()
    for i, e in enumerate(elements):
        if e.shape.is_empty:
            continue
        my_index.insert(e.index.numpy()[0], e.shape.bounds)

    for i in reversed(range(len(elements))):
        if elements[i].shape.is_empty:
            continue
        for j in my_index.intersection(elements[i].shape.bounds):
            if i == j:
                continue
            if elements[j].index < elements[i].index:
                current_shape = elements[j].shape
                elements[j] = ColorPoly(elements[j].index,
                                        geom_cleaning(elements[j].shape.difference(elements[i].shape)),
                                        elements[j].color)

    new_new_chunks = list()
    for chunk in elements:
        new_polys = unroll_poly(chunk.shape)
        new_new_chunks.extend([ColorPoly(chunk.index, s, chunk.color) for s in new_polys])

    return new_new_chunks


def main(f_args, *args, **kwargs):
    print(f_args)

    svg = f_args.path
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(svg)

    if type(shapes[0]) == pydiffvg.Path:
        my_geometries = list()

        for i, (s, sg) in enumerate(zip(shapes, shape_groups)):
            geo = Geometry.from_diffvg(s, sg)
            my_geometries.append(geo)
        color_polygons = Geometry.to_color_poly(my_geometries)
    else:
        color_polygons = ColorPoly.from_diffvg(shapes, shape_groups)

    chunks = ColorPoly.to_chunks_v2(color_polygons)

    if bool(f_args.quantize):
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
    else:
        new_chunks = chunks

    final_chunks = layer_carving_v2(new_chunks)
    ColorPoly.serialize_collection(final_chunks, f_args.file_name)
    ColorPoly.render(final_chunks, canvas_width, canvas_height, save_to_svg=True)


p = argparse.ArgumentParser()
p.add_argument("--path", type=str)
p.add_argument("--n_colors", default=6, type=int)
p.add_argument("--file_name", default="serialized_rez.txt", type=str)
p.add_argument("--quantize", default=1, type=int)
args = p.parse_known_args()


if __name__ == "__main__":
    main(*args)
