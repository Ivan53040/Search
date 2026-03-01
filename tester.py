import sys
import time

from game_env import GameEnv
from control.game_env import GameEnv as ControlEnv
from solution import Solver

"""
Tester script.

Use this script to debug and/or evaluate your solution. You may modify this file if desired.

COMP3702 Assignment 1 "Cheese Hunter" Support Code, 2025
"""

VISUALISE_TIME_PER_STEP = 0.7


def print_usage():
    print("Usage: python tester.py [search_type] [testcase_file] [-v (optional)]")
    print("    search_type = 'ucs' or 'a_star'")
    print("    testcase_file = filename of a valid testcase file (e.g. level_1.txt)")
    print("    if -v is specified, the solver's trajectory will be visualised")


def run_once_and_collect(search_type: str, game_env: GameEnv):
    """
    Run one search to obtain: actions and (if provided by solver) stats:
      - visited/reached
      - expanded
      - frontier size at termination
      - runtime (seconds)
    Returns: (actions, stats_dict)
    """
    solver = Solver(game_env)

    t0 = time.time()
    if search_type == "ucs":
        actions = solver.search_ucs()
    else:
        # A* may rely on preprocessing
        if hasattr(solver, "preprocess_heuristic"):
            solver.preprocess_heuristic()
        actions = solver.search_a_star()
    one_run_time = time.time() - t0

    stats = getattr(solver, "last_stats", None)
    if not isinstance(stats, dict):
        stats = {}

    # Fallbacks if solver didn't set last_stats
    stats.setdefault("visited", None)
    stats.setdefault("expanded", None)
    stats.setdefault("frontier", None)
    stats.setdefault("runtime", one_run_time)

    return actions, stats


def run_for_average_time(search_type: str, game_env: GameEnv):
    """
    Run multiple times to compute a smoother average runtime only.
    Returns: avg_runtime_seconds
    """
    # For small environments, take average time over multiple trials
    if game_env.ucs_time_min_tgt < 0.01:
        trials = 50
    elif game_env.ucs_time_min_tgt < 0.1:
        trials = 5
    else:
        trials = 1

    t0 = time.time()
    for _ in range(trials):
        solver = Solver(game_env)
        if search_type == "ucs":
            _ = solver.search_ucs()
        else:
            if hasattr(solver, "preprocess_heuristic"):
                solver.preprocess_heuristic()
            _ = solver.search_a_star()
    avg_run_time = (time.time() - t0) / max(trials, 1)
    return avg_run_time, trials


def main(arglist):
    if len(arglist) != 2 and len(arglist) != 3:
        print_usage()
        return

    search_type = arglist[0]
    if search_type not in ["ucs", "a_star"]:
        print("/!\\ ERROR: Invalid search_type given")
        print_usage()
        return

    # Load environment
    testcase_file = arglist[1]
    game_env = GameEnv(testcase_file)

    if len(arglist) == 3:
        if arglist[2] == "-v":
            visualise = True
        else:
            print(f"/!\\ ERROR: Invalid option given: {arglist[2]}")
            print_usage()
            return
    else:
        visualise = False

    # 1) Average runtime (optional but nice for reporting)
    avg_run_time, trials = run_for_average_time(search_type, game_env)

    # 2) One run to collect actions + stats (visited/expanded/frontier/runtime)
    actions, stats = run_once_and_collect(search_type, game_env)

    # Evaluate solution by replaying actions on a fresh control env
    control_env = ControlEnv(testcase_file)
    persistent_state = control_env.get_init_state()
    total_cost = 0.0
    error_occurred = False

    if visualise:
        try:
            from gui import GUI
            gui = GUI(game_env)
        except ModuleNotFoundError:
            gui = None
            control_env.render(persistent_state)
            time.sleep(VISUALISE_TIME_PER_STEP)
    else:
        gui = None

    for i, a in enumerate(actions or []):
        try:
            total_cost += game_env.ACTION_COST[a]
            success, persistent_state = game_env.perform_action(persistent_state, a)
            if not success:
                print("/!\\ ERROR: Action resulting in Collision performed at step " + str(i))
                error_occurred = True
            if visualise:
                if gui is not None:
                    gui.update_state(persistent_state)
                else:
                    control_env.render(persistent_state)
                    time.sleep(VISUALISE_TIME_PER_STEP)
        except KeyError:
            print("/!\\ ERROR: Unrecognised action performed at step " + str(i))
            error_occurred = True

    if error_occurred:
        print("/!\\ ERROR: Collision, Game Over or Unrecognised Action Occurred")

    # Report stats block for Q3
    print("\n=== Search Statistics (for Q3) ===")
    print(f"Algorithm:          {search_type.upper()}")
    print(f"Testcase:           {testcase_file}")
    print(f"Visited/Reached:    {stats.get('visited', 'N/A')}")
    print(f"Expanded:           {stats.get('expanded', 'N/A')}")
    print(f"Frontier (end):     {stats.get('frontier', 'N/A')}")
    print(f"Runtime (s):        {stats.get('runtime', float('nan')):.10f}")
    print(f"Avg Runtime (s):    {avg_run_time:.10f} (over {trials} trial{'s' if trials != 1 else ''})")

    if game_env.is_solved(persistent_state):
        print(f"Level completed! Solution cost = {round(total_cost, 1)}")
    else:
        print("/!\\ ERROR: Level not completed after all actions performed.")
        return


if __name__ == "__main__":
    main(sys.argv[1:])
