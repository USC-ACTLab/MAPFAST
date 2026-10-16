'''
Graph neural network selector: picks the fastest solver of a portfolio from the graph of an instance built by
tools/build_graphs.py (the coarsened grid of its MovingAI map, with the agents' starts, goals and shortest paths as node
features). Plain PyTorch, no graph library: a batch is the union of its graphs, with `batch` giving each node's graph.

	net = SolverGNN(node_dim, global_dim, solvers, kind='sage', hidden=128, layers=4)
	x, edge_index, batch, glob = collate([graphs[i] for i in names], maps, side, device)
	logits = net(x, edge_index, batch, glob, len(names))
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ('gcn', 'sage', 'gin')

def grid_edges(coords, batch, side):
	'''
	Edges between the nodes of a batch whose blocks share a side, in both directions.

	The arguments are:
		1. coords -> LongTensor (nodes, 2) of each node's [row, column] in the coarsened grid of its map
		2. batch -> LongTensor (nodes,) of each node's graph in the batch
		3. side -> Size of the coarsened grids (no row or column reaches it)

	Returns: LongTensor (2, edges) of the [source, target] of each edge
	'''
	graphs = int(batch.max()) + 1 if len(batch) else 0
	lookup = torch.full((graphs * side * side,), -1, dtype=torch.long, device=coords.device)
	cell = (batch * side + coords[:, 0]) * side + coords[:, 1]
	lookup[cell] = torch.arange(len(coords), device=coords.device)
	sources, targets = [], []
	for dr, dc in ((1, 0), (0, 1)):
		r, c = coords[:, 0] + dr, coords[:, 1] + dc
		inside = (r < side) & (c < side)
		neighbour = torch.full_like(cell, -1)
		neighbour[inside] = lookup[(batch[inside] * side + r[inside]) * side + c[inside]]
		node = torch.nonzero(neighbour >= 0).squeeze(1)
		sources += [node, neighbour[node]]
		targets += [neighbour[node], node]
	return torch.stack([torch.cat(sources), torch.cat(targets)])

class GraphLayer(nn.Module):
	'''
	One message-passing layer:
		gcn -> W . sum over j in N(i) and i of h_j / sqrt(deg_i deg_j)             (Kipf and Welling, 2017)
		sage -> W1 . h_i + W2 . mean over j in N(i) of h_j                         (Hamilton et al., 2017)
		gin -> MLP((1 + eps) h_i + sum over j in N(i) of h_j)                       (Xu et al., 2019)
	'''
	def __init__(self, kind, dim):
		super().__init__()
		if kind not in KINDS:
			raise ValueError('unknown layer {} (one of {})'.format(kind, ', '.join(KINDS)))
		self.kind = kind
		if kind == 'gcn':
			self.linear = nn.Linear(dim, dim)
		elif kind == 'sage':
			self.self_linear = nn.Linear(dim, dim)
			self.neighbour_linear = nn.Linear(dim, dim, bias=False)
		else:
			self.eps = nn.Parameter(torch.zeros(1))
			self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

	def forward(self, h, edge_index, degree):
		source, target = edge_index
		if self.kind == 'gcn':
			norm = (degree + 1).rsqrt()
			messages = h[source] * (norm[source] * norm[target]).unsqueeze(1)
			out = torch.zeros_like(h).index_add_(0, target, messages) + h * (norm * norm).unsqueeze(1)
			return self.linear(out)
		neighbours = torch.zeros_like(h).index_add_(0, target, h[source])
		if self.kind == 'sage':
			return self.self_linear(h) + self.neighbour_linear(neighbours / degree.clamp(min=1).unsqueeze(1))
		return self.mlp((1 + self.eps) * h + neighbours)

class SolverGNN(nn.Module):
	'''
	Node features -> `layers` residual message-passing layers -> mean and max over each graph's nodes, joined with the
	graph's global features -> MLP -> one logit per solver.

	The arguments are:
		1. node_dim, global_dim -> Number of node and global features
		2. solvers -> Number of solvers to pick from
		3. kind -> Message-passing layer: gcn, sage or gin (see GraphLayer)
		4. hidden -> Width of the node embeddings and of the MLP
		5. layers -> Number of message-passing layers
		6. dropout -> Dropout after each layer and in the MLP
	'''
	def __init__(self, node_dim, global_dim, solvers, kind='sage', hidden=128, layers=4, dropout=0.1):
		super().__init__()
		self.encode = nn.Linear(node_dim, hidden)
		self.layers = nn.ModuleList(GraphLayer(kind, hidden) for _ in range(layers))
		self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
		self.dropout = nn.Dropout(dropout)
		self.head = nn.Sequential(
			nn.Linear(2 * hidden + global_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, solvers))

	def forward(self, x, edge_index, batch, glob, graphs):
		'''
		Returns: Tensor (graphs, solvers) of the logits of each graph
		'''
		degree = torch.zeros(len(x), device=x.device).index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=x.device))
		h = F.relu(self.encode(x))
		for layer, norm in zip(self.layers, self.norms):
			h = h + self.dropout(F.relu(norm(layer(h, edge_index, degree))))

		counts = torch.zeros(graphs, device=x.device).index_add_(0, batch, torch.ones(len(batch), device=x.device))
		mean = torch.zeros(graphs, h.shape[1], device=x.device).index_add_(0, batch, h) / counts.clamp(min=1).unsqueeze(1)
		index = batch.unsqueeze(1).expand_as(h)
		maximum = torch.full((graphs, h.shape[1]), float('-inf'), device=x.device).scatter_reduce(0, index, h, 'amax')
		return self.head(torch.cat([mean, maximum, glob], dim=1))

def collate(graphs, maps, side, device, node_stats=None, global_stats=None):
	'''
	Joins graphs into one batch.

	The arguments are:
		1. graphs -> List of graphs as built by tools/build_graphs.py: {'map', 'x', 'glob'}
		2. maps -> Dictionary map -> numpy array of its node coordinates
		3. side -> Size of the coarsened grids
		4. device -> Device to put the batch on
		5. node_stats, global_stats -> Optional (mean, std) numpy arrays to standardise the features with

	Returns: Tuple of the node features, edges (see grid_edges), each node's graph and the global features, as tensors
	'''
	x = np.concatenate([g['x'] for g in graphs]).astype(np.float32)
	glob = np.stack([g['glob'] for g in graphs]).astype(np.float32)
	if node_stats is not None:
		x = (x - node_stats[0]) / node_stats[1]
	if global_stats is not None:
		glob = (glob - global_stats[0]) / global_stats[1]
	coords = torch.from_numpy(np.concatenate([maps[g['map']] for g in graphs]).astype(np.int64)).to(device)
	batch = torch.from_numpy(np.repeat(np.arange(len(graphs)), [len(g['x']) for g in graphs])).to(device)
	return torch.from_numpy(x).to(device), grid_edges(coords, batch, side), batch, torch.from_numpy(glob).to(device)
