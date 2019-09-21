from unittest import TestCase


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
		vertices = {i: Vertex(i, False, None, 0, list()) for i in range(max(edges.keys()) + 1)}
		weights = {i: {j: 1 for j in edges[i]} for i in edges.keys()}

		print(vertices)
		round_ = 0

		while len(vertices) > 1:
			print(round_)
			print(weights)
			vertices, weights = cls.one_round(vertices, weights)
			round_ += 1

		return weights

	###########################
	#        Main steps
	###########################

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
			children = [v.index]
			# and update the current vertices
			if best_neighbour:
				cls.update_vertex_merge_ststaus(best_neighbour, new_index, vertex_level, vertices)
				children.append(best_neighbour)
				#vertices[best_neighbour] = Vertex(best_neighbour, True, new_index, )
				#vertices[best_neighbour].is_merged = True
				#vertices[best_neighbour].new_index = new_index
			cls.update_vertex_merge_ststaus(v.index, new_index, vertex_level, vertices)
			#v.is_merged = True
			#v.new_index = new_index

			# 2.b : Add the new struct in the next graph
			next_vertices[new_index] = Vertex(new_index, False, None, vertex_level + 1)
			print("-->", next_vertices)
			# Now the edges
			# if the current neighbours 
			assert(vertices[best_neighbour].is_merged)
			cls.update_weights_for_vertex(vertices[v.index], vertices, weights, next_weights)
			cls.update_weights_for_vertex(vertices[best_neighbour], vertices, weights, next_weights)

		return next_vertices, next_weights

	@classmethod
	def tree_build(cls, depth, i, all_graphs, binary_tree):
		index = 2 ** depth - 1 + i
		children = cls.get_children(all_graphs, depth + 1, searched_new_index=i)

		print(depth, i, children)
		if children is None:
			return

		left = children[0] if len(children) > 0 else cls.get_max_index(all_graphs, depth + 1)
		binary_tree[2 * index + 1] = left
		cls.tree_build(depth + 1, left, all_graphs, binary_tree)
		right = children[1] if len(children) > 1 else cls.get_max_index(all_graphs, depth + 1)
		binary_tree[2 * (index + 1)] = right
		cls.tree_build(depth + 1, right, all_graphs, binary_tree)


	@classmethod
	def get_max_index(cls, all_graphs, depth):
		if depth >= len(all_graphs):
			return None  # We already reached the leaves
		max_index = len(all_graphs[depth])
		all_graphs[depth].append(Vertex(max_index, False, None, depth))
		return max_index

	@classmethod
	def get_children(cls, all_graphs, depth, searched_new_index):
		if depth >= len(all_graphs):
			return None  # We already reached the leaves
		return [v.index for v in all_graphs[depth] if v.new_index == searched_new_index]






	###########################
	#        Helpers
	###########################

	@classmethod
	def build_vertices_and_weights_from_edges(cls, edges):
		vertices = {i: Vertex(i, False, None, 0) for i in range(max(edges.keys()) + 1)}
		weights = {i: {j: 1 for j in edges[i]} for i in edges.keys()}
		return vertices, weights		

	@classmethod
	def update_weights_for_vertex(cls, current_vertex, vertices, weights, next_weights):
		for edge_index in weights[current_vertex.index]:
			neighbour = vertices[edge_index]
			if neighbour.is_merged and neighbour.new_index != current_vertex.new_index:
				#if neighbour.new_index not in next_weights:
				#	next_weights[neighbour.new_index] = dict()
				#if current_vertex not in next_weights[neighbour.new_index]:
				#	next_weights[neighbour.new_index][current_vertex.new_index] = 0
				next_weights[neighbour.new_index][current_vertex.new_index] += weights[current_vertex.index][edge_index]
				next_weights[current_vertex.new_index][neighbour.new_index] += weights[edge_index][current_vertex.index]

	@classmethod
	def update_vertex_merge_ststaus(cls, vertex_index, new_index, level, vertices):
		vertices[vertex_index] = Vertex(vertex_index, True, new_index, level)



class TestGraclusPooling(TestCase):

	@classmethod
	def setUpClass(cls):
		cls.cube_edges = {0: [1, 2, 7], 1: [0, 3, 6], 2: [0, 3, 5], 3: [1, 2, 4], 
						  4: [3, 5, 6], 5: [2, 4, 7], 6: [1, 4, 7], 7: [0, 5, 6]}

	def test_graph_building(self):
		vertices, weights = GraclusPooling.build_vertices_and_weights_from_edges(self.cube_edges)
		self.assertEqual(len(vertices), len(self.cube_edges))
		self.assertTrue(all([len(v) == len(w) for v, w in zip(weights.values(), self.cube_edges.values())]))

	def test_run_once(self):
		vertices, weights = GraclusPooling.build_vertices_and_weights_from_edges(self.cube_edges)

		new_vertices, new_weights = GraclusPooling.one_round(vertices, weights)
		print(dict(new_weights))
		#self.assertEqual(len(weights), 1)
		assert(all([w <= 2 for l in dict(new_weights).values() for w in l.values() ]))

	def test_tree_building(self):
		all_graphs = [[Vertex(0, False, None, 2)],
					  [Vertex(0, True, 0, 1), Vertex(1, True, 0, 1)],
					  [Vertex(1, True, 1, 0), Vertex(0, True, 0, 0), Vertex(2, True, 0, 0)]]

		tree = [0 for _ in range(2 ** len(all_graphs) - 1)]
		GraclusPooling.tree_build(0, 0, all_graphs, tree)

		print(tree[2 ** (len(all_graphs) - 1) - 1:])
		self.assertListEqual(tree[2 ** (len(all_graphs) - 1) - 1:], [0, 2, 1, 3])
		self.assertEqual(all_graphs[-1][-1].is_merged, False)





