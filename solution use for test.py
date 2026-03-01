from game_env import GameEnv
from game_state import GameState
import heapq
import time

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 1 "Cheese Hunter" Support Code, 2025
"""


class Solver:

    def __init__(self, game_env):
        # set the game environment
        self.game_env = game_env
        #
        #
        # TODO: Define any class instance variables you require here (avoid performing any computationally expensive
        #  heuristic preprocessing operations here - use the preprocess_heuristic method below for this purpose).
        #
        #
        # set the tie
        self.tie = 0
        # set the h cache
        self.h_cache = {}
        # set the preprocessed flag
        self.preprocessed = False

    @staticmethod
    def get_testcases():
        """
        Select which testcases you wish the autograder to test you on.
        The autograder will not run any excluded testcases.
        e.g. [1, 4, 6] will only run testcases 1, 4, and 6, excluding, 2, 3, and 5.
        :return: a list containing which testcase numbers to run (testcases in 1-6).
        """
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def get_search():
        """
        Select which search you wish the autograder to run.
        The autograder will only run the specified search methods.
        e.g. "both" will run both UCS and A*, but "a_star" will only run A* and exclude UCS.
        :return: a string containing which search methods to run ("ucs" to only run UCS, "a_star" to only run A*,
        and "both" to run both).
        """
        return "both"

    # === Uniform Cost Search ==========================================================================================
    

    def search_ucs(self):
        env = self.game_env
        start = env.get_init_state()

        frontier = []
        self.tie = 0
        heapq.heappush(frontier, (0.0, self.tie, start))

        best_g = {start: 0.0}
        parent = {start: (None, None)}

        # --- counters for Q3 ---
        reached = set([start])        # states ever pushed
        expanded = 0
        t0 = time.time()

        while frontier:
            g, _, state = heapq.heappop(frontier)
            if g != best_g.get(state, float('inf')):
                continue  # stale
            expanded += 1

            if env.is_solved(state):
                # reconstruct actions
                actions = []
                s = state
                while True:
                    prev, act = parent[s]
                    if prev is None: break
                    actions.append(act); s = prev
                actions.reverse()

                # publish stats for tester.py
                self.last_stats = {
                    "visited": len(reached),
                    "expanded": expanded,
                    "frontier": len(frontier),   # size at termination
                    "runtime": time.time() - t0,
                }
                return actions

            # expand successors
            for a in env.ACTIONS:
                success, next_state = env.perform_action(state, a)
                if not success:
                    continue
                new_g = g + env.ACTION_COST[a]
                if new_g < best_g.get(next_state, float('inf')):
                    best_g[next_state] = new_g
                    parent[next_state] = (state, a)
                    self.tie += 1
                    heapq.heappush(frontier, (new_g, self.tie, next_state))
                    reached.add(next_state)

        # no solution case (still publish stats)
        self.last_stats = {
            "visited": len(reached),
            "expanded": expanded,
            "frontier": len(frontier),
            "runtime": time.time() - t0,
        }
        return None

        

    # === A* Search ====================================================================================================
    # ------------------------------------------------------------
    # helper: directed lower-bound distance between two cells
    # movement costs different for horizontal and vertical movements
    # ------------------------------------------------------------

    def _directed_cost(self, r1, c1, r2, c2):
        # calculate the horizontal distance
        dx = abs(c1 - c2)
        # calculate the vertical distance
        dy = r1 - r2  # +dy means we must go UP; -dy means we go DOWN
        # calculate the up and down costs
        up = max(0, dy)
        down = max(0, -dy)
        # return the total cost
        return (
            dx * self._horiz_per_cell
            + up * self._up_per_cell
            + down * self._down_per_cell
        )

    # ------------------------------------------------------------
    # (1) pre-process once: read env constants, build pairwise dists, set up MST cache
    # build pairwise distance between every POI(levers and goal)
    # ------------------------------------------------------------
    def preprocess_heuristic(self):
        # get the environment
        env = self.game_env
        # ----- goal / levers -----
        # Try goal_row/goal_col first; fall back to a tuple attribute if needed.
        if hasattr(env, "goal_row") and hasattr(env, "goal_col"):
            # if the goal row and goal column are attributes of the environment, set the goal position
            self.goal_pos = (env.goal_row, env.goal_col)
        # if the goal is an attribute of the environment, set the goal position
        elif hasattr(env, "goal"):
            self.goal_pos = tuple(env.goal)  # expect (r, c)
        # if the goal position is an attribute of the environment, set the goal position
        elif hasattr(env, "goal_pos"):
            self.goal_pos = tuple(env.goal_pos)
        # if the goal position is not an attribute of the environment, raise an error
        else:
            raise AttributeError("GameEnv must expose goal_row/goal_col or goal/goal_pos tuple")

        # index i of trap_status corresponds to lever_positions[i]
        if not hasattr(env, "lever_positions"):
            # if the lever positions are not an attribute of the environment, raise an error
            raise AttributeError("GameEnv must expose lever_positions (list of (r,c) for each lever).")
        # set the lever positions
        self.lever_positions = list(env.lever_positions)
        # set the number of levers
        self.L = len(self.lever_positions)
        # set the goal index
        self.GOAL_IDX = self.L
        # set the pois
        self._pois = self.lever_positions + [self.goal_pos]  # L levers + goal
        # ----- action costs -> per-cell directional costs -----
        # get the action costs
        AC = env.ACTION_COST
        wl = AC.get(getattr(env, "WALK_LEFT", "wl"), float("inf"))
        wr = AC.get(getattr(env, "WALK_RIGHT", "wr"), float("inf"))
        sl = AC.get(getattr(env, "SPRINT_LEFT", "sl"), float("inf"))
        sr = AC.get(getattr(env, "SPRINT_RIGHT", "sr"), float("inf"))
        climb = AC.get(getattr(env, "CLIMB", "c"), float("inf"))
        jump  = AC.get(getattr(env, "JUMP", "j"), float("inf"))
        drop  = AC.get(getattr(env, "DROP", "d"), float("inf"))
        act   = AC.get(getattr(env, "ACTIVATE", "a"), 1.0)
        # set the horizontal per cell cost
        # sprint covers 2 columns per action → per-cell is cost/2
        self._horiz_per_cell = min(
            wl, wr,
            sl / 2.0 if sl < float("inf") else float("inf"),
            sr / 2.0 if sr < float("inf") else float("inf"),
        )
        # set the up per cell cost
        self._up_per_cell = min(climb, jump)
        # set the down per cell cost
        self._down_per_cell = drop
        # set the activate cost
        self._activate_cost = act

        # fallbacks (shouldn't trigger in the template, but keep admissible if they do)
        if not (self._horiz_per_cell < float("inf")): self._horiz_per_cell = 1.0
        if not (self._up_per_cell    < float("inf")): self._up_per_cell    = 2.0
        if not (self._down_per_cell  < float("inf")): self._down_per_cell  = 0.5

        # ----- build symmetric pairwise LB between every POI -----
        # set the number of pois
        N = self.L + 1
        # set the pairwise lower bound
        self._pair_sym = [[0.0]*N for _ in range(N)]
        # for each poi
        for i in range(N):
            r1, c1 = self._pois[i]
            # for each poi
            for j in range(i+1, N):
                # get the position of the poi
                r2, c2 = self._pois[j]
                # calculate the directed cost
                dij = self._directed_cost(r1, c1, r2, c2)
                dji = self._directed_cost(r2, c2, r1, c1)
                w = min(dij, dji)  # symmetric lower bound
                self._pair_sym[i][j] = w
                self._pair_sym[j][i] = w
        # ----- fast arrays and caches -----
        # set the mst cache
        self._mst_cache = {}
        # split lever (r,c) into separate arrays for hot loops
        # set the lever r
        self._lever_r = [rc[0] for rc in self.lever_positions]
        # set the lever c
        self._lever_c = [rc[1] for rc in self.lever_positions]
        # set the mask cache
        # cache: trap_status tuple -> (mask, remaining_count)
        # (far fewer unique trap_status than states)
        self._mask_cache = {}
        
        # set the whole-heuristic cache
        # (re-init here is safe; pre-processing is per-env)
        if not hasattr(self, "h_cache"):
            self.h_cache = {}
        # set the preprocessed flag
        self.preprocessed = True

    # ------------------------------------------------------------
    # helper: MST cost over remaining levers ∪ {goal} using Prim’s (O(k^2)), cached by mask
    # mask: bit i = 1 if lever i is still UNACTIVATED
    # computes a minimum spanning tree over the remaining levers and the goal
    # ------------------------------------------------------------
    def _mst_cost(self, mask: int) -> float:
        if mask == 0:
            # no levers left → MST over {goal} is 0
            return 0.0
        if mask in self._mst_cache:
            return self._mst_cache[mask]

        # Build the list of node indices (remaining levers + goal_idx)
        nodes = [i for i in range(self.L) if (mask >> i) & 1]
        # add the goal index to the nodes
        nodes.append(self.GOAL_IDX)
        # set the number of nodes
        k = len(nodes)
        # if the number of nodes is 1, return 0
        if k <= 1:
            self._mst_cache[mask] = 0.0
            return 0.0

        # set the in tree
        in_tree = [False]*k
        # Prim’s: keep current best edge into the partial tree
        best = [float("inf")]*k
        best[0] = 0.0  # start at nodes[0]
        total = 0.0

        # for each node
        for _ in range(k):
            # pick u not in tree with minimal best[u]
            u = -1
            # set the best u
            bu = float("inf")
            # for each node
            for i in range(k):
                # if the node is not in the tree and the best value is less than the best u
                if not in_tree[i] and best[i] < bu:
                    bu = best[i]; u = i
            # set the node in the tree
            in_tree[u] = True
            # add the best u to the total
            total += bu
            # get the node u
            iu = nodes[u]
            # for each node
            for v in range(k):
                # if the node is in the tree, continue
                if in_tree[v]:
                    continue
                # get the node v
                iv = nodes[v]
                # get the weight
                w = self._pair_sym[iu][iv]
                # if the weight is less than the best v, update the best v
                if w < best[v]:
                    best[v] = w

        # set the mst cache
        self._mst_cache[mask] = total
        return total

    # ------------------------------------------------------------
    # (2) admissible heuristic
    #   - if no levers left: directed distance to goal
    #   - else: min dist to any {lever or goal} + MST over {remaining levers ∪ goal} + (#levers)*activate_cost
    # ------------------------------------------------------------
    def compute_heuristic(self, state) -> float:
        # if the preprocessed flag is not set or the lever r is not an attribute of the solver, preprocess the heuristic
        if (not getattr(self, "preprocessed", False)) or (not hasattr(self, "_lever_r")):
            self.preprocess_heuristic()

        # local bindings (avoid attribute lookup in hot path)
        # set the directed cost
        dcost = self._directed_cost
        # set the lever r
        lr = self._lever_r
        # set the lever c
        lc = self._lever_c
        # set the goal row and goal column
        goal_r, goal_c = self.goal_pos
        # set the activate cost
        act_cost = self._activate_cost
        # set the row and column
        r = state.row
        c = state.col
        # 1) trap_status -> (mask, remaining), cached
        ts = state.trap_status
        # get the mask and remaining
        mr = self._mask_cache.get(ts)
        if mr is None:
            # set the mask
            mask = 0
            # set the remaining
            remaining = 0
            # build bitmask once for this trap_status tuple
            # build bitmask once for this trap_status tuple
            for i, flag in enumerate(ts):
                if flag == 0:
                    # set the mask
                    mask |= (1 << i)
                    # increment the remaining
                    remaining += 1
            # set the mask cache
            self._mask_cache[ts] = (mask, remaining)
        else:
            # get the mask and remaining
            mask, remaining = mr
        
        # 2) whole-heuristic cache by (r,c,mask)
        key = (r, c, mask)
        # get the heuristic cached
        h_cached = self.h_cache.get(key)
        if h_cached is not None:
            return h_cached

        # 3) no levers left -> direct to goal (directed LB)
        if mask == 0:
            h = dcost(r, c, goal_r, goal_c)
            self.h_cache[key] = h
            return h

        # 4) mindist to any remaining lever (iterate set bits only)
        mind = float("inf")
        m = mask
        while m:
            lsb = m & -m                 # lowest set bit
            i = (lsb.bit_length() - 1)   # index of that bit
            d = dcost(r, c, lr[i], lc[i])
            if d < mind:
                mind = d
            m ^= lsb                     # clear that bit

        # 5) structural LB: MST over {remaining levers ∪ goal}, cached by mask
        mst = self._mst_cost(mask)

        # 6) at least one activate per remaining lever
        h = mind + mst + remaining * act_cost
        self.h_cache[key] = h
        return h
    # ------------------------------------------------------------
    # (3) A* with goal-on-pop + stale-entry skipping
    # ------------------------------------------------------------
    def search_a_star(self):
        env = self.game_env
        start = env.get_init_state()
        if not getattr(self, "preprocessed", False):
            self.preprocess_heuristic()

        frontier = []
        self.tie = 0
        heapq.heappush(frontier, (self.compute_heuristic(start), self.tie, start))

        best_g = {start: 0.0}
        parent = {start: (None, None)}

        reached = set([start])
        expanded = 0
        t0 = time.time()

        while frontier:
            f, _, state = heapq.heappop(frontier)
            g = best_g.get(state, float('inf'))
            if f != g + self.compute_heuristic(state):
                # optional: skip if you store f differently; minimally ensure staleness is handled
                pass
            # standard stale-check:
            if g == float('inf'):
                continue
            expanded += 1

            if env.is_solved(state):
                actions = []
                s = state
                while True:
                    prev, act = parent[s]
                    if prev is None: break
                    actions.append(act); s = prev
                actions.reverse()
                self.last_stats = {
                    "visited": len(reached),
                    "expanded": expanded,
                    "frontier": len(frontier),
                    "runtime": time.time() - t0,
                }
                return actions

            for a in env.ACTIONS:
                success, next_state = env.perform_action(state, a)
                if not success:
                    continue
                g2 = g + env.ACTION_COST[a]
                if g2 < best_g.get(next_state, float('inf')):
                    best_g[next_state] = g2
                    parent[next_state] = (state, a)
                    self.tie += 1
                    heapq.heappush(frontier, (g2 + self.compute_heuristic(next_state), self.tie, next_state))
                    reached.add(next_state)

        self.last_stats = {
            "visited": len(reached),
            "expanded": expanded,
            "frontier": len(frontier),
            "runtime": time.time() - t0,
        }
        return None
