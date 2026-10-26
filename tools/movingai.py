'''MovingAI map, scenario and instance names (e.g. Berlin_1_256-random-1_agents10.yaml), in [row, column] cells.'''
import functools
import heapq
import re

import numpy as np

INSTANCE = re.compile(r'^((.+)-(random|even)-\d+)_agents(\d+)\.yaml$')

def map_name(instance):
	return INSTANCE.match(instance).group(2)

def scen_name(instance):
	return INSTANCE.match(instance).group(1)

def scen_type(instance):
	return INSTANCE.match(instance).group(3)

def agent_count(instance):
	return int(INSTANCE.match(instance).group(4))

@functools.lru_cache(maxsize=4)
def load_map(map_file):
	'''(height, width) boolean array, True for obstacles.'''
	with open(map_file) as f:
		lines = f.read().splitlines()
	height = int(lines[1].split()[1])
	width = int(lines[2].split()[1])
	return np.array([[c in '@TO' for c in row[:width]] for row in lines[4:4 + height]])

def load_scenario(scen_file):
	'''(start, goal) of every agent, in file order.'''
	agents = []
	with open(scen_file) as f:
		if 'version 1' not in f.readline():
			raise ValueError(scen_file + ' is not a version 1 scenario file')
		for line in f:
			fields = line.rstrip('\r\n').split('\t')
			if len(fields) >= 8:
				sx, sy, gx, gy = (int(v) for v in fields[4:8])
				agents.append(([sy, sx], [gy, gx]))
	return agents

def shortest_path(obstacles, start, goal):
	'''4-connected A*; [] if unreachable.'''
	height, width = obstacles.shape
	blocked = obstacles.ravel()
	s = start[0] * width + start[1]
	g = goal[0] * width + goal[1]
	gr, gc = goal

	parent = {s: s}
	cost = {s: 0}
	heap = [(abs(start[0] - gr) + abs(start[1] - gc), 0, s)]
	while heap:
		_, d, cell = heapq.heappop(heap)
		d = -d
		if cell == g:
			break
		if d > cost[cell]:
			continue
		r, c = divmod(cell, width)
		for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
			if 0 <= nr < height and 0 <= nc < width:
				n = nr * width + nc
				if not blocked[n] and (n not in cost or d + 1 < cost[n]):
					cost[n] = d + 1
					parent[n] = cell
					# ties go to the deeper node
					heapq.heappush(heap, (d + 1 + abs(nr - gr) + abs(nc - gc), -(d + 1), n))
	else:
		return []

	path = [g]
	while path[-1] != s:
		path.append(parent[path[-1]])
	return [list(divmod(cell, width)) for cell in reversed(path)]
