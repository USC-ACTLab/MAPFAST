'''
General-purpose classifier baseline: asks TypeSafe AI's Jev (https://docs.typesafe.ai/), a hosted model that answers
typed questions about a text state, which solver of a dataset's portfolio is fastest on each test instance, and writes
its picks in the format of main.py -T 0, so that plots.py treats it like a trained model.

Jev only reads text, so each instance is described by its map, the number of agents, start-goal distances and the
agents' start and goal cells (--no-coords leaves those out). With --examples K, the state also holds the K training
instances of the same map and scenario type with the closest number of agents, with every solver's time, so that
Jev gets some of the training data a trained model has.

It is only used as a baseline: scripts/compare.sh runs it on the test instances of the models it trains. By hand, e.g. to
check the response format on a few instances before a round (--limit):

	export TYPESAFE_API_KEY=...
	python tools/derive_dataset.py datasets/movingai_10-500-agents_300s/four-solvers --name movingai_10-500-agents_300s/four-solvers/jev20
	python models/jev_baseline.py datasets/movingai_10-500-agents_300s/four-solvers/jev20 --test-from datasets/movingai_10-500-agents_300s/four-solvers/y_pred_true_movingai_10-500-agents_300s_four-solvers.json --examples 20 --limit 5 --yes

The test instances are those of --test-from, the predictions of a model trained on the same instances, so that both
are evaluated on exactly the same problems; every other instance of the dataset counts as training data for --examples.
Without --yes, it only prints the estimated cost. It stops before a request that would take the spending past --budget.
'''
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from utils import read_config, read_json

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
# USD per million input tokens (output tokens are free), as listed at launch
PRICE = 0.042

SOLVERS = {
	'bcp2': 'Branch-and-cut-and-price (BCP), an optimal MAPF solver built on integer programming and column generation',
	'cbs': 'Conflict-Based Search, the basic optimal two-level search: a tree of constraints over single-agent A* paths',
	'ecbs': 'Enhanced CBS, a bounded-suboptimal variant of CBS with focal search',
	'mapf_prioritized_sipp': 'Prioritized planning with Safe Interval Path Planning: agents planned one by one, fast but neither optimal nor complete',
	'cbsh2_rtc': 'CBSH2-RTC, optimal CBS with admissible heuristics and rectangle, target and corridor reasoning',
	'cbsh2_rtc_ch': 'CBSH2-RTC with conflict-based heuristics',
	'cbsh2_rtc_bp': 'CBSH2-RTC with bypassing of conflicts',
	'cbsh2_rtc_chbp': 'CBSH2-RTC with conflict-based heuristics and bypassing',
	'reloc': 'Optimal MAPF by reduction to a constraint or satisfiability problem',
}

def map_and_type(instance):
	'''
	Returns: Tuple of the MovingAI map and the scenario type of an instance, e.g. ('Berlin_1_256', 'random') for Berlin_1_256-random-1_agents10.yaml
	'''
	m = re.match(r'(.*)-(random|even)-\d+_agents\d+\.yaml$', instance)
	return (m.group(1), m.group(2)) if m else (instance, None)

def features(instance, map_details, agent_details, coords=True):
	'''
	Returns: Json object describing an instance for Jev
	'''
	md = map_details[instance]
	height, width = md['mp_dim']
	starts, goals = agent_details[instance]['starts'], agent_details[instance]['goals']
	distances = [abs(s[0] - g[0]) + abs(s[1] - g[1]) for s, g in zip(starts, goals)]
	free = height * width - md['no_obs']
	f = {
		'map': map_and_type(instance)[0],
		'scenario_type': map_and_type(instance)[1],
		'height': height, 'width': width, 'obstacle_cells': md['no_obs'], 'free_cells': free,
		'agents': md['no_agents'], 'agents_per_free_cell': round(md['no_agents'] / max(free, 1), 4),
		'mean_start_goal_manhattan_distance': round(sum(distances) / len(distances), 1),
		'max_start_goal_manhattan_distance': max(distances),
	}
	if coords:
		f['starts_row_col'] = starts
		f['goals_row_col'] = goals
	return f

def examples_for(instance, train, map_details, agent_details, yaml_details, solvers, k):
	'''
	Returns: List of the k training instances of the same map and scenario type with the closest number of agents,
	each with its features (without coordinates) and every solver's time
	'''
	same = [t for t in train if map_and_type(t) == map_and_type(instance)]
	agents = map_details[instance]['no_agents']
	same.sort(key=lambda t: abs(map_details[t]['no_agents'] - agents))
	return [{**features(t, map_details, agent_details, coords=False),
			 'seconds': {s: (yaml_details[t][s] if yaml_details[t][s] != -1 else 'failed') for s in solvers},
			 'fastest': yaml_details[t]['SOLVER']} for t in same[:k]]

def request(state, solvers, timeout, model, key):
	body = {
		'model': model,
		'state': state,
		'questions': {
			'fastest_solver': {
				'type': 'choice',
				'instructions': 'Which solver finds a solution to this multi-agent path finding instance in the least time '
								'(a solver that does not finish within {:g} seconds counts as {:g} seconds)'.format(timeout, timeout),
				'criteria': {s: SOLVERS.get(s) for s in solvers},
			},
		},
	}
	data = json.dumps(body).encode()
	for attempt in range(6):
		req = urllib.request.Request(ENDPOINT, data=data, headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
		try:
			start = time.perf_counter()
			with urllib.request.urlopen(req, timeout=60) as r:
				response = json.load(r)
			return response, time.perf_counter() - start
		except urllib.error.HTTPError as e:
			if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
				raise SystemExit('Jev request failed: HTTP {} {}'.format(e.code, e.read().decode(errors='replace')[:500]))
		except (urllib.error.URLError, TimeoutError):
			if attempt == 5:
				raise
		time.sleep(2 ** attempt)

def input_tokens(response, state):
	'''
	Returns: Input tokens of a response, from its usage if reported, else estimated at 4 characters per token
	'''
	usage = response.get('usage') or {}
	counted = [v for k, v in usage.items() if 'input' in k and isinstance(v, (int, float))]
	return counted[0] if counted else len(state) / 4

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pick solvers with Jev, as a general-purpose classifier baseline')
	parser.add_argument('config', help='Dataset folder (or its config.json) to write the predictions for, e.g. datasets/movingai_10-500-agents_300s/four-solvers/jev20')
	parser.add_argument('--test-from', required=True, help='Predictions of a model trained on the same dataset (main.py -T 0), whose instances are the test set')
	parser.add_argument('--examples', type=int, default=0, help='Training instances of the same map to put in each state (default 0: none)')
	parser.add_argument('--no-coords', action='store_true', help='Leave the agents\' start and goal cells out of the state')
	parser.add_argument('--limit', type=int, default=None, help='Only ask about the first n test instances and print the first response, without writing the predictions (a pilot)')
	parser.add_argument('--budget', type=float, default=3.0, help='Most USD to spend in this run (default 3)')
	parser.add_argument('--model', default='jev-latest')
	parser.add_argument('--jobs', type=int, default=8, help='Requests in flight at the same time')
	parser.add_argument('--yes', action='store_true', help='Send the requests (otherwise only print the estimated cost)')
	args = parser.parse_args()

	config = read_config(args.config)
	yaml_details = read_json(config['yaml_details'])
	map_details = read_json(config['map_details'])
	agent_details = read_json(config['agent_details'])
	mapping = config['mapping']
	solvers = sorted(mapping, key=mapping.get)
	timeout = config['timeout']

	test_keys = list(read_json(args.test_from))[:args.limit]
	test = set(k[:k.rindex('_')] for k in test_keys)
	missing = [k for k in test if k not in yaml_details]
	if missing:
		raise SystemExit('{} test instances of {} are not in {}'.format(len(missing), args.test_from, config['yaml_details']))
	train = [k for k in yaml_details if k not in test]

	states = {}
	for key in test_keys:
		instance = key[:key.rindex('_')]
		state = {'instance': features(instance, map_details, agent_details, coords=not args.no_coords)}
		if args.examples:
			state['training_examples_same_map'] = examples_for(instance, train, map_details, agent_details, yaml_details, solvers, args.examples)
		states[key] = json.dumps(state, separators=(',', ':'))
	estimate = sum(len(s) for s in states.values()) / 4 / 1e6 * PRICE
	print('{} test instances, about {:.1f}M input tokens, about ${:.2f} at ${}/M'.format(
		len(states), sum(len(s) for s in states.values()) / 4 / 1e6, estimate, PRICE))
	if not args.yes:
		print('Add --yes to send them')
		sys.exit(0)
	if estimate > args.budget:
		raise SystemExit('estimated cost is over the --budget of ${:.2f}'.format(args.budget))
	key = os.environ.get('TYPESAFE_API_KEY') or sys.exit('set TYPESAFE_API_KEY')

	predictions = {}
	latencies = []
	spent = 0.
	with ThreadPoolExecutor(args.jobs) as pool:
		futures = {k: pool.submit(request, s, solvers, timeout, args.model, key) for k, s in states.items()}
		for i, (k, future) in enumerate(futures.items()):
			response, latency = future.result()
			latencies.append(latency)
			if args.limit and i == 0:
				print(json.dumps(response, indent=1))
			spent += input_tokens(response, states[k]) / 1e6 * PRICE
			answer = response['answers']['fastest_solver']
			instance = k[:k.rindex('_')]
			predictions[k] = {'best': [mapping[answer['choice']], mapping[yaml_details[instance]['SOLVER']]],
							  'probabilities': answer.get('probabilities'), 'confidence': answer.get('confidence'),
							  'latency_s': latency}
			if spent > args.budget:
				pool.shutdown(wait=False, cancel_futures=True)
				raise SystemExit('stopped after {} instances: spent ${:.2f}, over the --budget'.format(i + 1, spent))
			if (i + 1) % 100 == 0:
				print('{}/{} instances, ${:.3f} spent'.format(i + 1, len(states), spent), flush=True)

	if args.limit:
		correct = sum(p['best'][0] == p['best'][1] for p in predictions.values())
		print('pilot: {} of {} picks are the fastest solver, ${:.3f} spent; predictions not written'.format(correct, len(predictions), spent))
		sys.exit(0)
	with open(config['prediction_output'], 'w') as f:
		json.dump(predictions, f)
	print('wrote {} ({} instances, ${:.3f} spent)'.format(config['prediction_output'], len(predictions), spent))
	# the selector's own time is spent before any solver runs, so it adds to the total runtime
	print('Jev latency: mean {:.3f} s, median {:.3f} s per instance; {:.1f} min over the test set'.format(
		sum(latencies) / len(latencies), sorted(latencies)[len(latencies) // 2], sum(latencies) / 60))
