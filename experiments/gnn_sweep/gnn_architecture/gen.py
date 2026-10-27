'''
Writes map.tex and graph.tex for gnn_architecture_paths.tex: a 12 x 12 toy map with 3 agents as MAPFAST-GNN's `paths` graph.

	python gen.py && pdflatex gnn_architecture_paths.tex && pdftoppm -png -r 200 -singlefile gnn_architecture_paths.pdf gnn_architecture_paths
'''
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../tools'))
from movingai import shortest_path
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
H = len(rows)
agents = [((0, 0), (11, 11)), ((11, 0), (0, 11)), ((5, 2), (3, 10))]
paths = [shortest_path(obs, list(s), list(g)) for s, g in agents]
xy = lambda r, c: (c + .5, H - .5 - r)

m = [r'\fill[black] (%d,%d) rectangle ++(1,1);' % (c, H - 1 - r) for r in range(H) for c in range(H) if obs[r, c]]
for i, ((s, g), p) in enumerate(zip(agents, paths)):
    m.append(r'\draw[c%d, line width=1.6pt, rounded corners=2pt] %s;' % (i, ' -- '.join('(%.1f,%.1f)' % xy(r, c) for r, c in p)))
    m.append(r'\fill[c%d] (%.2f,%.2f) rectangle ++(.8,.8);' % (i, s[1] + .1, H - .9 - s[0]))
    m.append(r'\draw[c%d, fill=white, line width=1.6pt] (%.1f,%.1f) circle (.35);' % (i, *xy(*g)))
open('map.tex', 'w').write('\n'.join(m))

# a node per free cell (k = 1), shaded by how many paths visit it; an agent node per agent, right of the grid
visits = np.zeros((H, H), int)
for p in paths:
    for r, c in p:
        visits[r, c] += 1
g = []
for r in range(H):
    for c in range(H):
        for dr, dc in ((1, 0), (0, 1)):
            if not obs[r, c] and r + dr < H and c + dc < H and not obs[r + dr, c + dc]:
                g.append(r'\draw[gray!55, line width=.5pt] (%.1f,%.1f) -- (%.1f,%.1f);' % (*xy(r, c), *xy(r + dr, c + dc)))
agent_xy = [(15.8, 9.5), (15.8, 6), (15.8, 2.5)]
for i, ((s, goal), p) in enumerate(zip(agents, paths)):
    ax, ay = agent_xy[i]
    for r, c in p[1:-1]:
        g.append(r'\draw[c%d, opacity=.28, line width=.45pt] (%.1f,%.1f) -- (%.1f,%.1f);' % (i, ax, ay, *xy(r, c)))
    g.append(r'\draw[c%d, line width=1.3pt] (%.1f,%.1f) -- (%.1f,%.1f);' % (i, ax, ay, *xy(*s)))
    g.append(r'\draw[c%d, line width=1.3pt, dashed] (%.1f,%.1f) -- (%.1f,%.1f);' % (i, ax, ay, *xy(*goal)))
for r in range(H):
    for c in range(H):
        if not obs[r, c]:
            g.append(r'\draw[line width=.45pt, fill=pathc!%d!white] (%.1f,%.1f) circle (.3);' % (min(100, 45 * visits[r, c]), *xy(r, c)))
for i, (ax, ay) in enumerate(agent_xy):
    g.append(r'\node[draw, fill=c%d, text=white, minimum size=.42cm, inner sep=0pt, font=\tiny\bfseries] at (%.1f,%.1f) {a%d};' % (i, ax, ay, i + 1))
open('graph.tex', 'w').write('\n'.join(g))
