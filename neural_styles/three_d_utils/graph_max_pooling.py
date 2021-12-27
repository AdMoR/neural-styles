
from collections import defaultdict, namedtuple


Vertex = namedtuple("Vertex", ["index", "is_merged", "new_index", "level"])


class GraclusPooling:

	@classmethod
	def run(cls, edges):
		"""
		We expect to get a dict with the following format
		0: [1, 3, 4]
		1: [0]
		2: [4]
		3: [0]
		4: [0, 2]
		"""
		vertices = {i: Vertex(i, False, None, 0) for i in range(max(edges.keys()))}
		weights = {i: {j: 1 for j in edges[i]} for i in edges.keys()}

		while len(vertices) > 1:
			vertices, edges = cls.one_round(vertices, weights)

	@classmethod
	def one_round(cls, vertices, weights):
		"""
		In this function, we build the next level of pooling

		vertices: Dict[int, Vertex]
		edges: Dict[int, Dict[int, Edges]]

		"""
		if len(vertices) <= 1:
			return None, None
		next_vertices = dict()
		next_weights = defaultdict(lambda : defaultdict(lambda: 0))

		vertex_level = vertices[0].level

		for v in vertices.values():
			if v.is_merged:
				continue
			best_score = 1000000
			best_neighbour = None
			for neighboor_index in weights[v.index]:
				if vertices[neighboor_index].is_merged:
					continue

				# Compute w_ij * (d_i^-1 + d_j^-1) 
				score = weights[v.index][neighboor_index] * (1. / (len(weights[v.index]) + 1) + 1. / (len(weights[neighboor_index]) + 1))

				if score < best_score:
					best_neighbour = neighboor_index
					best_score = score

			# 2. Update the next graph
			# 2.a : find the new_index (size of new graph)
			new_index = len(next_vertices)
			# and update the current vertices
			if best_neighbour:
				vertex[best_neighbour].is_merged = True
				vertex[best_neighbour].new_index = new_index
			v.is_merged = True
			v.new_index = new_index

			# 2.b : Add the new struct in the next graph
			new_vertices.append(Vertex(new_index, False, None, vertex_level + 1))
			# Now the edges
			# if the current neighbours 
			cls.update_weights_for_vertex(v, vertices, weights, next_weights)
			cls.update_weights_for_vertex(vertex[best_neighbour], vertices, weights, next_weights)

		return next_vertices, next_weights

	@classmethod
	def update_weights_for_vertex(cls, current_vertex, vertices, weights, next_weights):
		for edge_index in weights[current_vertex.index]:
			neighbour = weights[edge_index]
			if neighbour.is_merged:
				#if neighbour.new_index not in next_weights:
				#	next_weights[neighbour.new_index] = dict()
				#if current_vertex not in next_weights[neighbour.new_index]:
				#	next_weights[neighbour.new_index][current_vertex.new_index] = 0
				next_weights[neighbour.new_index][current_vertex.new_index] += weights[current_vertex.index][edge_index]
				next_weights[current_vertex.new_index][neighbour.new_index] += weights[edge_index][current_vertex.index]

