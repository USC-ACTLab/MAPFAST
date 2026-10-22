'''
Writes map.tex and graph.tex for gnn_architecture.tex: a 12 x 12 toy map with 3 agents, coarsened to 2 x 2 blocks as
in tools/build_graphs.py (node fill: path cells, ring: cells shared by two paths).

	python gen.py && pdflatex gnn_architecture.tex && pdftoppm -png -r 200 -singlefile gnn_architecture.pdf gnn_architecture
'''
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tools'))
from benchmark_dataset import shortest_path
rows = ["............",
        "..##....##..",
        "..##....##..",
        "......#.....",
        "##....#..###",
        "##.......###",
        "....##......",
        "....##..#...",
        ".#......#...",
        ".#..##......",
        "....##..##..",
        "........##.."]
obs = np.array([[ch == '#' for ch in r] for r in rows])
H = len(rows); C = ['2a78d6', 'eb6834', '1baf7a']
agents = [((0, 0), (11, 11)), ((11, 0), (0, 11)), ((5, 2), (3, 10))]
m = []
for r in range(H):
    for c in range(H):
        if obs[r, c]:
            m.append(r'\fill[black] (%d,%d) rectangle ++(1,1);' % (c, H - 1 - r))
visits = np.zeros((H, H), int)
for i, (s, g) in enumerate(agents):
    p = shortest_path(obs, list(s), list(g))
    for r, c in p: visits[r, c] += 1
    m.append(r'\draw[c%d, line width=1.6pt, rounded corners=2pt] %s;' % (i, ' -- '.join('(%.1f,%.1f)' % (c + .5, H - .5 - r) for r, c in p)))
    m.append(r'\fill[c%d] (%.2f,%.2f) rectangle ++(.8,.8);' % (i, s[1] + .1, H - .9 - s[0]))
    m.append(r'\draw[c%d, fill=white, line width=1.6pt] (%.1f,%.1f) circle (.35);' % (i, g[1] + .5, H - .5 - g[0]))
open('map.tex', 'w').write('\n'.join(m))
k = 2; S = H // k
blk = lambda a: a.reshape(S, k, S, k).sum(axis=(1, 3))
free = blk(~obs) / 4; paths = blk(visits > 0); shared = blk(visits > 1)
g = []
node = lambda r, c: (c * 2 + 1, (S - 1 - r) * 2 + 1)
for r in range(S):
    for c in range(S):
        if free[r, c] == 0: continue
        for dr, dc in ((1, 0), (0, 1)):
            if r + dr < S and c + dc < S and free[r + dr, c + dc] > 0:
                g.append(r'\draw[gray!70, line width=.9pt] (%d,%d) -- (%d,%d);' % (*node(r, c), *node(r + dr, c + dc)))
for r in range(S):
    for c in range(S):
        if free[r, c] == 0: continue
        x, y = node(r, c); pct = int(100 * paths[r, c] / paths.max())
        g.append(r'\draw[line width=.7pt, fill=pathc!%d!white] (%d,%d) circle (.55);' % (pct, x, y))
        if shared[r, c]: g.append(r'\draw[c1, line width=1.2pt] (%d,%d) circle (.75);' % (x, y))
open('graph.tex', 'w').write('\n'.join(g))
