import heapq

from game_env import GameEnv
from game_state import GameState


class Solver:
    """
    Solver implementing Uniform Cost Search (UCS) and A* for the Cheese Hunter task.

    Same algorithms as before; micro-optimised for speed:
    - Heavier precomputation of heuristic constants
    - Aggressive local-binding in loops (reduces attribute lookups)
    - Cheap mask caching for trap_status → (mask, remaining)
    - Cached MST over remaining levers ∪ goal (by bitmask)
    - Flat dict caches for (r,c,mask) heuristic queries
    - Lightweight path reconstruction helper
    """

    # === Construction & Autograder selection ========================================================

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        self.tie = 0

        # heuristic caches
        self.h_cache = {}
        self._mst_cache = {}
        self._mask_cache = {}
        self._nodes_cache = {} 
        self.preprocessed = False

    @staticmethod
    def get_testcases():
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def get_search():
        return "both"

    # === Heuristic: preprocessing + helpers =========================================================

    def _directed_cost(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """
        Calculate the cost of moving from (r1, c1) to (r2, c2).
        """
        dx = abs(c1 - c2)
        dy = r1 - r2  # +dy means go UP; -dy means go DOWN
        up = dy if dy > 0 else 0
        down = -dy if dy < 0 else 0
        return (
            dx * self._horiz_per_cell
            + up * self._up_per_cell
            + down * self._down_per_cell
        )

    def preprocess_heuristic(self):
        """Precompute environment constants and caches used by the heuristic."""
        env = self.game_env

        # ----- goal -----
        if hasattr(env, "goal_row") and hasattr(env, "goal_col"):
            self.goal_pos = (env.goal_row, env.goal_col)
        elif hasattr(env, "goal"):
            self.goal_pos = tuple(env.goal)
        elif hasattr(env, "goal_pos"):
            self.goal_pos = tuple(env.goal_pos)
        else:
            self.goal_pos = (0, 0)  # safe fallback

        # ----- levers & goal -----
        self.lever_positions = list(getattr(env, "lever_positions", []))
        self.L = len(self.lever_positions)
        self.GOAL_IDX = self.L
        # add goal_pos to lever_positions as it also needs to be activated
        self._pois = self.lever_positions + [self.goal_pos]

        # ----- costs -----
        AC = env.ACTION_COST
        wl = AC.get(getattr(env, "WALK_LEFT", "wl"), float("inf"))
        wr = AC.get(getattr(env, "WALK_RIGHT", "wr"), float("inf"))
        sl = AC.get(getattr(env, "SPRINT_LEFT", "sl"), float("inf"))
        sr = AC.get(getattr(env, "SPRINT_RIGHT", "sr"), float("inf"))
        climb = AC.get(getattr(env, "CLIMB", "c"), float("inf"))
        jump = AC.get(getattr(env, "JUMP", "j"), float("inf"))
        drop = AC.get(getattr(env, "DROP", "d"), float("inf"))
        act = AC.get(getattr(env, "ACTIVATE", "a"), 1.0)

        # per-cell lower bounds (asymmetric actions collapsed to symmetric LB)
        # consider sprinting more than walking
        horiz_candidates = [wl, wr]
        if sl < float("inf"):
            horiz_candidates.append(sl / 2.0)
        if sr < float("inf"):
            horiz_candidates.append(sr / 2.0)
        self._horiz_per_cell = min(horiz_candidates) if horiz_candidates else 1.0
        self._up_per_cell = min(climb, jump) if (climb < float("inf") or jump < float("inf")) else 2.0
        self._down_per_cell = drop if drop < float("inf") else 0.5
        self._activate_cost = act

        # ----- pairwise symmetric LBs between POIs -----
        # N is the number of POIs (levers + goal)
        N = self.L + 1
        # pair is a NxN matrix where pair[i][j] is the cost of going from POI i to POI j
        # pair[i][j] = pair[j][i] because the cost is symmetric
        pair = [[0.0] * N for _ in range(N)]
        for i in range(N):
            r1, c1 = self._pois[i]
            for j in range(i + 1, N):
                r2, c2 = self._pois[j]
                dij = self._directed_cost(r1, c1, r2, c2)
                dji = self._directed_cost(r2, c2, r1, c1)
                # take the minimum of the two costs
                w = dij if dij < dji else dji
                # set the cost for both directions
                pair[i][j] = w
                pair[j][i] = w
        self._pair_sym = pair

        # vectors for fast lever coordinate access
        self._lever_r = [rc[0] for rc in self.lever_positions]
        self._lever_c = [rc[1] for rc in self.lever_positions]

        # reset caches
        self._mst_cache.clear()
        self._mask_cache.clear()
        self._nodes_cache.clear()
        self.h_cache = self.h_cache if hasattr(self, "h_cache") else {}
        self.preprocessed = True
    
    def _mst_cost(self, mask: int) -> float:
        """MST over remaining levers ∪ {goal} using Prim’s; cached by mask."""
        # if there are no remaining levers, the MST cost is 0
        if mask == 0:
            return 0.0
        # check if the MST cost is already cached
        cached = self._mst_cache.get(mask)
        # if it is, return the cached value
        if cached is not None:
            return cached

        # nodes for this mask: cache once
        nodes = self._nodes_cache.get(mask)
        if nodes is None:
            # collect remaining lever indices + goal once per mask
            # (tuple is faster to reuse than list)
            tmp = [i for i in range(self.L) if (mask >> i) & 1]
            tmp.append(self.GOAL_IDX)
            nodes = tuple(tmp)
            self._nodes_cache[mask] = nodes
        # k is the number of nodes in the MST
        k = len(nodes)
        # if there is only one node, the MST cost is 0
        if k <= 1:
            self._mst_cache[mask] = 0.0
            return 0.0

        # Prim’s algorithm with minimal Python overhead
        INF = float("inf")
        # in_tree is a list of k booleans, where in_tree[i] is True if node i is in the tree
        in_tree = [False] * k
        # best is a list of k floats, where best[i] is the minimum cost of an edge from node i to the tree
        best = [INF] * k
        # the first node has a cost of 0
        best[0] = 0.0

        total = 0.0
        # pair is the pairwise cost matrix
        pair = self._pair_sym  # local bind

        for _ in range(k):
            # select u, the node with the minimum cost to the tree
            u = -1
            bu = INF
            # manual index loop tends to be fastest in CPython
            # i is the index of the node
            for i in range(k):
                if not in_tree[i]:
                    bi = best[i]
                    # if the cost to the tree is less than the current minimum, update the minimum
                    if bi < bu:
                        bu = bi
                        u = i

            # add the node to the tree
            in_tree[u] = True
            total += bu
            iu = nodes[u]

            # relax edges
            for v in range(k):
                # if the node is not in the tree, update the minimum cost to the tree
                if not in_tree[v]:
                    iv = nodes[v]
                    w = pair[iu][iv]
                    # if the cost to the tree is less than the current minimum, update the minimum
                    if w < best[v]:
                        best[v] = w

        # cache the MST cost
        self._mst_cache[mask] = total
        return total

    def compute_heuristic(self, state: GameState) -> float:
        """
        Admissible heuristic:
          - If no levers left: directed distance to goal
          - Else: min dist to any remaining lever + MST over (remaining levers ∪ goal)
                  + (#remaining levers)*activate_cost
        """
        if not self.preprocessed:
            self.preprocess_heuristic()

        # local binds (hot path)
        dcost = self._directed_cost
        lr = self._lever_r
        lc = self._lever_c
        goal_r, goal_c = self.goal_pos
        act_cost = self._activate_cost
        r = state.row
        c = state.col

        # trap_status -> (mask, remaining), cached
        ts = state.trap_status
        # get the mask and remaining from the cache
        mr = self._mask_cache.get(ts)
        # if it is not cached, compute the mask and remaining
        if mr is None:
            # mask is a bitmask of the remaining levers
            # remaining is the number of remaining levers
            mask = 0
            remaining = 0
            # iterate over the trap_status
            for i, flag in enumerate(ts):
                # if the trap is not triggered, set the bit in the mask
                if flag == 0:
                    mask |= (1 << i)
                    remaining += 1
            # cache the mask and remaining
            self._mask_cache[ts] = (mask, remaining)
        else:
            mask, remaining = mr

        # whole-heuristic cache by (r,c,mask)
        key = (r, c, mask)
        # check if the heuristic value is already cached
        h_val = self.h_cache.get(key)
        # if it is, return the cached value
        if h_val is not None:
            return h_val

        # if there are no remaining levers, return the distance to the goal
        if mask == 0:
            h = dcost(r, c, goal_r, goal_c)
            self.h_cache[key] = h
            return h

        # min distance to any remaining lever (iterate set bits only)
        # mind is the minimum distance to any remaining lever
        mind = float("inf")
        m = mask
        # iterate over the mask
        while m:
            # get the least significant bit
            lsb = m & -m
            # get the index of the least significant bit
            i = (lsb.bit_length() - 1)
            # get the distance to the lever
            d = dcost(r, c, lr[i], lc[i])
            # if the distance is less than the current minimum, update the minimum
            if d < mind:
                mind = d
            # remove the least significant bit from the mask
            m ^= lsb

        # get the MST cost
        mst = self._mst_cost(mask)
        # compute the heuristic value
        h = mind + mst + remaining * act_cost
        # cache the heuristic value
        self.h_cache[key] = h
        return h

    # === Shared helpers ============================================================================

    @staticmethod
    def _reconstruct(parent, goal_state):
        """Reconstruct action path using a (state -> (prev_state, action)) map."""
        actions = []
        s = goal_state
        while True:
            prev, act = parent[s]
            if prev is None:
                break
            actions.append(act)
            s = prev
        actions.reverse()
        return actions

    # === Search algorithms =========================================================================

    def search_ucs(self):
        """Uniform Cost Search (UCS)."""
        env = self.game_env
        start = env.get_init_state()

        # Local binds (tight loop performance)
        ACTIONS = env.ACTIONS
        ACTION_COST = env.ACTION_COST
        perform_action = env.perform_action
        is_solved = env.is_solved

        # Frontier: (g, tie, state)
        self.tie = 0
        frontier = [(0.0, self.tie, start)]
        best_g = {start: 0.0}
        parent = {start: (None, None)}

        heappop = heapq.heappop
        heappush = heapq.heappush
        INF = float("inf")

        while frontier:
            g, _, s = heappop(frontier)
            if g != best_g.get(s, INF):
                continue
            if is_solved(s):
                return self._reconstruct(parent, s)

            for a in ACTIONS:
                ok, ns = perform_action(s, a)
                if not ok:
                    continue
                new_g = g + ACTION_COST[a]
                if new_g < best_g.get(ns, INF):
                    best_g[ns] = new_g
                    parent[ns] = (s, a)
                    self.tie += 1
                    heappush(frontier, (new_g, self.tie, ns))
        return None

    def search_a_star(self):
        """A* search (admissible heuristic)."""
        env = self.game_env
        if not self.preprocessed:
            self.preprocess_heuristic()

        start = env.get_init_state()

        # Local binds
        ACTIONS = env.ACTIONS
        ACTION_COST = env.ACTION_COST
        perform_action = env.perform_action
        is_solved = env.is_solved
        h = self.compute_heuristic

        parent = {start: (None, None)}
        best_g = {start: 0.0}
        self.tie = 0

        g0 = 0.0
        f0 = g0 + h(start)
        frontier = [(f0, self.tie, g0, start)]

        heappop = heapq.heappop
        heappush = heapq.heappush
        INF = float("inf")

        while frontier:
            f, _, g, s = heappop(frontier)
            if g != best_g.get(s, INF):
                continue
            if is_solved(s):
                return self._reconstruct(parent, s)

            for a in ACTIONS:
                ok, ns = perform_action(s, a)
                if not ok:
                    continue
                new_g = g + ACTION_COST[a]
                if new_g < best_g.get(ns, INF):
                    best_g[ns] = new_g
                    parent[ns] = (s, a)
                    self.tie += 1
                    heappush(frontier, (new_g + h(ns), self.tie, new_g, ns))
        return None
