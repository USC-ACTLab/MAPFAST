'''
Graph neural network selector: picks the fastest solver of a portfolio from the graph of an instance built by
tools/build_graphs.py: a node per free cell of its MovingAI map, joined to its 4-connected neighbours, and a node per
agent, joined to its start and goal cells and, with paths=True, to every cell of its shortest path, so that agents
whose paths meet are 2 hops apart however far apart they are on the map. Each edge has a relation (RELATIONS), with its
own weights. Plain PyTorch, no graph library: a batch is the union of its graphs, with `batch` giving each node's graph
and `kind` whether it is a cell (0) or an agent (1).

	net = SolverGNN(node_dim, global_dim, solvers, kind='sage', hidden=128, layers=4)
	x, edge_index, edge_type, batch, kind, glob = collate(names, graphs, device, paths=True)
	outputs = net(x, edge_index, edge_type, batch, kind, glob, len(names))
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ('sage', 'gin')
# the relation of each edge type, from its source to its target
RELATIONS = ('cell-cell', 'start-agent', 'agent-start', 'goal-agent', 'agent-goal', 'path-agent', 'agent-path')

class GraphLayer(nn.Module):
	'''
	One message-passing layer, with a weight matrix W_r per relation r (Schlichtkrull et al., 2018):
		sage -> W . h_i + sum over r of W_r . mean over j in N_r(i) of h_j        (Hamilton et al., 2017)
		gin -> MLP((1 + eps) h_i + sum over r of W_r . sum over j in N_r(i) of h_j)   (Xu et al., 2019)
	'''
	def __init__(self, kind, dim):
		super().__init__()
		if kind not in KINDS:
			raise ValueError('unknown layer {} (one of {})'.format(kind, ', '.join(KINDS)))
		self.kind = kind
		# the W_r side by side, applied to the relations' aggregates side by side
		self.neighbour_linear = nn.Linear(len(RELATIONS) * dim, dim, bias=False)
		if kind == 'sage':
			self.self_linear = nn.Linear(dim, dim)
		else:
			self.eps = nn.Parameter(torch.zeros(1))
			self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

	def forward(self, h, edge_index, slot, degree):
		'''
		`slot` is each edge's target * len(RELATIONS) + relation, `degree` the number of edges of each slot
		'''
		neighbours = torch.zeros(len(h) * len(RELATIONS), h.shape[1], device=h.device).index_add_(0, slot, h[edge_index[0]])
		if self.kind == 'sage':
			neighbours = neighbours / degree.clamp(min=1).unsqueeze(1)
		neighbours = self.neighbour_linear(neighbours.view(len(h), -1))
		if self.kind == 'sage':
			return self.self_linear(h) + neighbours
		return self.mlp((1 + self.eps) * h + neighbours)

class SolverGNN(nn.Module):
	'''
	Node features -> `layers` residual message-passing layers -> mean and max over each graph's cells and over its
	agents, joined with the graph's global features -> MLP -> one output per solver.

	The arguments are:
		1. node_dim, global_dim -> Number of node and global features
		2. solvers -> Number of solvers to pick from
		3. kind -> Message-passing layer: sage or gin (see GraphLayer)
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
			nn.Linear(4 * hidden + global_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, solvers))

	def forward(self, x, edge_index, edge_type, batch, kind, glob, graphs):
		'''
		Returns: Tensor (graphs, solvers) of the outputs of each graph
		'''
		slot = edge_index[1] * len(RELATIONS) + edge_type
		degree = torch.zeros(len(x) * len(RELATIONS), device=x.device).index_add_(0, slot, torch.ones(len(slot), device=x.device))
		h = F.relu(self.encode(x))
		for layer, norm in zip(self.layers, self.norms):
			h = h + self.dropout(F.relu(norm(layer(h, edge_index, slot, degree))))

		# pool cells and agents apart: group 2g holds graph g's cells, 2g + 1 its agents
		group = 2 * batch + kind
		counts = torch.zeros(2 * graphs, device=x.device).index_add_(0, group, torch.ones(len(group), device=x.device))
		mean = torch.zeros(2 * graphs, h.shape[1], device=x.device).index_add_(0, group, h) / counts.clamp(min=1).unsqueeze(1)
		index = group.unsqueeze(1).expand_as(h)
		maximum = torch.full((2 * graphs, h.shape[1]), float('-inf'), device=x.device).scatter_reduce(0, index, h, 'amax')
		maximum = maximum.masked_fill(counts.unsqueeze(1) == 0, 0)
		return self.head(torch.cat([mean.view(graphs, -1), maximum.view(graphs, -1), glob], dim=1))

def graph_size(name, graphs):
	'''
	Returns: Number of nodes of an instance's graph
	'''
	g = graphs['instances'][name]
	return graphs['maps'][graphs['scenarios'][g['scenario']]['map']]['cells'] + g['agents']

def collate(names, graphs, device, global_stats=None, paths=False):
	'''
	Joins the graphs of instances into one batch, computing their node features (see tools/build_graphs.py).

	The arguments are:
		1. names -> List of instances
		2. graphs -> Graph file of tools/build_graphs.py: {'maps', 'scenarios', 'instances'}
		3. device -> Device to put the batch on
		4. global_stats -> Optional (mean, std) numpy arrays to standardise the global features with
		5. paths -> Whether to join each agent to every cell of its path, besides its start and goal

	Returns: Tuple of the node features, edges, each edge's relation (index of RELATIONS), each node's graph, each
	node's kind (0 cell, 1 agent) and the global features, as tensors
	'''
	xs, edges, types, batch, kind = [], [], [], [], []
	base = 0
	for i, name in enumerate(names):
		g = graphs['instances'][name]
		s = graphs['scenarios'][g['scenario']]
		m = graphs['maps'][s['map']]
		n, cells = g['agents'], m['cells']
		path_cells = s['paths'][:s['offsets'][n]]
		visits = np.bincount(path_cells, minlength=cells)
		# visits of each agent's path cells, less its own one per cell
		on_path = np.concatenate([[0], np.cumsum(visits[path_cells])])
		cells_on_path = np.diff(s['offsets'][:n + 1])
		overlap = on_path[s['offsets'][1:n + 1]] - on_path[s['offsets'][:n]] - cells_on_path

		x = np.zeros((cells + n, 7), dtype=np.float32)
		x[:cells, 1] = np.bincount(s['start'][:n], minlength=cells)
		x[:cells, 2] = np.bincount(s['goal'][:n], minlength=cells)
		x[:cells, 3] = np.log1p(visits)
		x[cells:, 0] = 1
		x[cells:, 4] = np.log1p(np.maximum(cells_on_path - 1, 0))
		x[cells:, 5] = s['detour'][:n]
		x[cells:, 6] = np.log1p(overlap)
		agents = np.arange(cells, cells + n)
		links = [(s['start'][:n], agents, 1), (s['goal'][:n], agents, 3)]
		if paths:
			links.append((path_cells, np.repeat(agents, cells_on_path), 5))
		edges.append(m['edges'] + base)
		types.append(np.zeros(m['edges'].shape[1], dtype=np.int64))
		for cell, agent, relation in links:
			edges += [np.stack([cell, agent]) + base, np.stack([agent, cell]) + base]
			types += [np.full(len(cell), relation), np.full(len(cell), relation + 1)]
		xs.append(x)
		batch.append(np.full(cells + n, i))
		kind.append(np.repeat([0, 1], [cells, n]))
		base += cells + n

	glob = np.stack([graphs['instances'][n]['glob'] for n in names]).astype(np.float32)
	if global_stats is not None:
		glob = (glob - global_stats[0]) / global_stats[1]
	tensor = lambda a, dtype: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
	return (tensor(np.concatenate(xs), torch.float32), tensor(np.concatenate(edges, axis=1), torch.long),
			tensor(np.concatenate(types), torch.long), tensor(np.concatenate(batch), torch.long), tensor(np.concatenate(kind), torch.long), tensor(glob, torch.float32))
