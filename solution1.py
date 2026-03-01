from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import heapq

from game_env import GameEnv
from game_state import GameState

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 1 "Cheese Hunter" Support Code, 2025
"""


class Solver:
    """
    Solver implementing Uniform Cost Search (UCS) and A* for the Cheese Hunter task.

    Notes:
    - Code path and behaviour preserved from your original version.
    - Only ordering, docstrings, type hints, and comments improved for readability.
    """

    # === Construction & Autograder selection ========================================================

    def __init__(self, game_env: GameEnv) -> None:
        """
        Create a Solver.

        Avoid computationally expensive heuristic preprocessing here. Use
        preprocess_heuristic() for that purpose.
        """
        # set the game environment
        self.game_env: GameEnv = game_env

        # tie-breaker for the heap
        self.tie: int = 0

        # heuristic caches
        self.h_cache: Dict[Tuple[int, int, int], float] = {}
        self.preprocessed: bool = False

    @staticmethod
    def get_testcases() -> List[int]:
        """
        Select which testcases you wish the autograder to test you on.
        The autograder will not run any excluded testcases.
        e.g. [1, 4, 6] will only run testcases 1, 4, and 6, excluding 2, 3, and 5.
        :return: a list containing which testcase numbers to run (testcases in 1-6).
        """
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def get_search() -> str:
        """
        Select which search you wish the autograder to run.
        The autograder will only run the specified search methods.
        e.g. "both" will run both UCS and A*, but "a_star" will only run A* and exclude UCS.
        :return: a string containing which search methods to run ("ucs" to only run UCS, "a_star" to only run A*,
        and "both" to run both).
        """
        return "both"

    # === Heuristic: preprocessing + helpers =========================================================

    def _directed_cost(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """
        Lower-bound directed cost to move from (r1,c1) to (r2,c2) under asymmetric movement:
        - Horizontal cost uses the best per-cell (walk vs sprint/2).
        - Vertical up uses min(climb, jump).
        - Vertical down uses drop.
        """
        # calculate the horizontal distance
        dx = abs(c1 - c2)
        # calculate the vertical distance (positive means we must go up)
        dy = r1 - r2  # +dy means we must go UP; -dy means we go DOWN
        up = max(0, dy)
        down = max(0, -dy)
        # return the total cost
        return (
            dx * self._horiz_per_cell
            + up * self._up_per_cell
            + down * self._down_per_cell
        )

    def preprocess_heuristic(self) -> None:
        """
        Precompute environment constants and caches used by the heuristic.
        This sets goal position, lever positions, per-cell costs, pairwise lower bounds,
        and MST cache. If some attributes are missing, safe defaults are used instead
        of raising errors (to allow search to proceed).
        """
        env = self.game_env

        # ----- goal -----
        if hasattr(env, "goal_row") and hasattr(env, "goal_col"):
            self.goal_pos = (env.goal_row, env.goal_col)
        elif hasattr(env, "goal"):
            self.goal_pos = tuple(env.goal)
        elif hasattr(env, "goal_pos"):
            self.goal_pos = tuple(env.goal_pos)
        else:
            # Fallback: default to (0,0) if not defined
            print("[Warning] No goal attribute in GameEnv, defaulting to (0,0).")
            self.goal_pos = (0, 0)

        # ----- levers -----
        if hasattr(env, "lever_positions"):
            self.lever_positions = list(env.lever_positions)
        else:
            # Fallback: no levers
            print("[Warning] No lever_positions attribute in GameEnv, assuming none.")
            self.lever_positions = []

        self.L = len(self.lever_positions)
        self.GOAL_IDX = self.L
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

        self._horiz_per_cell = min(
            wl, wr,
            sl / 2.0 if sl < float("inf") else float("inf"),
            sr / 2.0 if sr < float("inf") else float("inf"),
        )
        self._up_per_cell = min(climb, jump)
        self._down_per_cell = drop
        self._activate_cost = act

        # Safe fallbacks
        if not (self._horiz_per_cell < float("inf")): self._horiz_per_cell = 1.0
        if not (self._up_per_cell < float("inf")): self._up_per_cell = 2.0
        if not (self._down_per_cell < float("inf")): self._down_per_cell = 0.5

        # ----- pairwise distances -----
        N = self.L + 1
        self._pair_sym = [[0.0] * N for _ in range(N)]
        for i in range(N):
            r1, c1 = self._pois[i]
            for j in range(i + 1, N):
                r2, c2 = self._pois[j]
                dij = self._directed_cost(r1, c1, r2, c2)
                dji = self._directed_cost(r2, c2, r1, c1)
                w = min(dij, dji)
                self._pair_sym[i][j] = w
                self._pair_sym[j][i] = w

        # ----- caches -----
        self._mst_cache = {}
        self._lever_r = [rc[0] for rc in self.lever_positions]
        self._lever_c = [rc[1] for rc in self.lever_positions]
        self._mask_cache = {}
        if not hasattr(self, "h_cache"):
            self.h_cache = {}
        self.preprocessed = True


    def _mst_cost(self, mask: int) -> float:
        """
        MST cost over remaining levers ∪ {goal} using Prim’s (O(k^2)), cached by mask.
        mask: bit i = 1 if lever i is still UNACTIVATED.
        """
        if mask == 0:
            # no levers left → MST over {goal} is 0
            return 0.0
        if mask in self._mst_cache:
            return self._mst_cache[mask]

        # Build the list of node indices (remaining levers + goal_idx)
        nodes: List[int] = [i for i in range(self.L) if (mask >> i) & 1]
        nodes.append(self.GOAL_IDX)
        k = len(nodes)
        if k <= 1:
            self._mst_cache[mask] = 0.0
            return 0.0

        in_tree = [False]*k
        # Prim’s: keep current best edge into the partial tree
        best = [float("inf")]*k
        best[0] = 0.0  # start at nodes[0]
        total = 0.0

        for _ in range(k):
            # pick u not in tree with minimal best[u]
            u = -1
            bu = float("inf")
            for i in range(k):
                if not in_tree[i] and best[i] < bu:
                    bu = best[i]; u = i
            in_tree[u] = True
            total += bu
            iu = nodes[u]
            for v in range(k):
                if in_tree[v]:
                    continue
                iv = nodes[v]
                w = self._pair_sym[iu][iv]
                if w < best[v]:
                    best[v] = w

        self._mst_cache[mask] = total
        return total

    def compute_heuristic(self, state: GameState) -> float:
        """
        Admissible heuristic:
          - If no levers left: directed distance to goal
          - Else: min dist to any remaining lever + MST over (remaining levers ∪ goal) + (#levers)*activate_cost
        """
        # ensure preprocessing
        if (not getattr(self, "preprocessed", False)) or (not hasattr(self, "_lever_r")):
            self.preprocess_heuristic()

        # local bindings (avoid attribute lookup in hot path)
        dcost = self._directed_cost
        lr = self._lever_r
        lc = self._lever_c
        goal_r, goal_c = self.goal_pos
        act_cost = self._activate_cost
        r = state.row
        c = state.col

        # 1) trap_status -> (mask, remaining), cached
        ts: Tuple[int, ...] = state.trap_status
        mr = self._mask_cache.get(ts)
        if mr is None:
            mask = 0
            remaining = 0
            # build bitmask once for this trap_status tuple
            for i, flag in enumerate(ts):
                if flag == 0:
                    mask |= (1 << i)
                    remaining += 1
            self._mask_cache[ts] = (mask, remaining)
        else:
            mask, remaining = mr

        # 2) whole-heuristic cache by (r,c,mask)
        key = (r, c, mask)
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

    # === Search algorithms =========================================================================

    def search_ucs(self) -> Optional[List[str]]:
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        :return: path (list of actions, where each action is an element of GameEnv.ACTIONS) or None.
        """
        # get the environment
        env = self.game_env
        start: GameState = env.get_init_state()

        # initialize frontier
        frontier: List[Tuple[float, int, GameState]] = []
        # initialize tie
        self.tie = 0
        # push the start state to the frontier
        heapq.heappush(frontier, (0.0, self.tie, start))
        # initialize best g and parent
        best_g: Dict[GameState, float] = {start: 0.0}
        parent: Dict[GameState, Tuple[Optional[GameState], Optional[str]]] = {start: (None, None)}

        # while the frontier is not empty
        while frontier:
            # pop the frontier
            g, _, state = heapq.heappop(frontier)
            # skip if not the best g value
            if g != best_g.get(state, float('inf')):
                continue
            # goal test
            if env.is_solved(state):
                # reconstruct actions
                actions: List[str] = []
                s = state
                while True:
                    # get the parent and action
                    prev, act = parent[s]
                    if prev is None:
                        break
                    # add the action to the actions
                    actions.append(act)  # type: ignore[arg-type]
                    # update the state
                    s = prev
                # reverse the actions for the correct order
                actions.reverse()
                return actions

            # expand
            for a in env.ACTIONS:
                # perform the action
                success, next_state = env.perform_action(state, a)
                # if the action is not successful, skip
                if not success:
                    continue
                # calculate the new g value (g + cost of the action)
                new_g = g + env.ACTION_COST[a]
                # if the new g value is better than the best g value, update the best g value and parent
                if new_g < best_g.get(next_state, float('inf')):
                    best_g[next_state] = new_g
                    # update the parent
                    parent[next_state] = (state, a)
                    # update the tie
                    self.tie += 1
                    # push the new state to the frontier
                    heapq.heappush(frontier, (new_g, self.tie, next_state))
        # if no solution is found, return None
        return None

    def search_a_star(self) -> Optional[List[str]]:
        """
        Find a path which solves the environment using A* search.
        Uses compute_heuristic() as the admissible heuristic.
        :return: path (list of actions, where each action is an element of GameEnv.ACTIONS) or None.
        """
        # get the environment
        env = self.game_env
        # if the preprocessed flag is not set, preprocess the heuristic
        if not getattr(self, "preprocessed", False):
            self.preprocess_heuristic()

        # get the start state
        start: GameState = env.get_init_state()
        # set the parent
        parent: Dict[GameState, Tuple[Optional[GameState], Optional[str]]] = {start: (None, None)}
        # set the best g
        best_g: Dict[GameState, float] = {start: 0.0}
        # set the tie
        self.tie = 0

        # set the g0
        g0: float = 0.0
        # set the f0
        f0: float = g0 + self.compute_heuristic(start)
        # set the frontier
        frontier: List[Tuple[float, int, float, GameState]] = [(f0, self.tie, g0, start)]

        while frontier:
            # pop the frontier
            f, _, g, s = heapq.heappop(frontier)
            # if the g is not the best g, continue
            if g != best_g.get(s, float("inf")):
                continue

            # if the state is solved, reconstruct the actions
            if env.is_solved(s):
                # reconstruct actions
                actions: List[str] = []
                # set the current state
                cur = s
                # while the current state is not None
                while True:
                    # get the parent and action
                    prev, act = parent[cur]
                    # if the parent is None, break
                    if prev is None:
                        break
                    # add the action to the actions
                    actions.append(act)  # type: ignore[arg-type]
                    # update the current state
                    cur = prev
                # reverse the actions for the correct order
                actions.reverse()
                return actions

            # expand
            for a in env.ACTIONS:
                # perform the action
                ok, ns = env.perform_action(s, a)
                # if the action is not successful, skip
                if not ok:
                    continue
                # calculate the new g value
                new_g = g + env.ACTION_COST[a]
                # if the new g value is better than the best g value, update the best g value and parent
                if new_g < best_g.get(ns, float("inf")):
                    # update the best g value
                    best_g[ns] = new_g
                    # update the parent
                    parent[ns] = (s, a)
                    # increment the tie
                    self.tie += 1
                    # calculate the new f value
                    new_f = new_g + self.compute_heuristic(ns)
                    # push the new state to the frontier
                    heapq.heappush(frontier, (new_f, self.tie, new_g, ns))
        # if no solution is found, return None
        return None
