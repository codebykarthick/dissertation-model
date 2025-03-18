import curses
from constants import CONSTANTS
from logger import setup_logger
import os
import sys

log = setup_logger()
weights_dir = os.path.join(os.getcwd(), CONSTANTS["weights_path"])


def get_weight_for_model(model_name):
    """
    Get the model name and then list all the weights that are
    previously saved and then return the full path of the weight the
    user selects.
    """
    if not os.path.exists(weights_dir):
        log.fatal(
            "Weights path does not exist (probably no training done before), exiting")
        sys.exit(1)

    model_weights_dir = os.path.join(weights_dir, model_name)

    if not os.path.exists(model_weights_dir):
        log.fatal(
            "Model weights directory does not exist (probably no training done before), exiting")
        sys.exit(1)

    weights_files = os.listdir(model_weights_dir)
    if not weights_files:
        log.fatal("No weights found for the model. Exiting.")
        sys.exit(1)

    selected_weight = curses.wrapper(_weight_selection_menu, weights_files)
    selected_weight = os.path.join(model_weights_dir, selected_weight)
    return selected_weight


def _weight_selection_menu(stdscr, weights):
    curses.curs_set(0)  # Hide the cursor
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr(
            0, 0, "Select a weight file using arrow keys and press Enter")
        for idx, weight in enumerate(weights):
            y = idx + 1
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, 0, weight)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, 0, weight)
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(weights) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return weights[current_row]
