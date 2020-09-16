"""tables Algorithm:

This Algorithm is a version of an greedy algorithm called: 'The Farthest Neighbors Algorithm'
like in the original version, every step the algorithm tries to place another table,
and in addition place it at the farest point from the other tables that have already putted.

in contrast to the original algorithm, here we consider every table as an 2D object instead of a point,
and that what make this algorithm suitable for the "Tables Arrangement" problem.
"""

import matplotlib.pyplot as plt
import numpy as np

"""tables_arrangement_algorithm is tha main function of the algorithm.
this function tries to find the best table arrangement that possible.
because this is an approximation algorithm, the main loop of the function is executed few times.
every iteration this function calls place_tabels, which is the actual function that try add all tables.

Parameters
----------
tables_list : list
    a list of all tables entered by the user
restaurant_matrix : np,matrix
    representation of the restaurant according what the user chose

Returns
-------
best_distance: int
    the minimal distance in meters between the closest tables in the restaurant
"""


def tables_arrangement_algorithm(tables_list, restaurant_matrix):
    init_board = create_init_board(restaurant_matrix, tables_list)  # initialize board
    best_board = init_board.copy()
    global_min_distance: int = -1  # represent the minimal distance between some tables on the board

    for i in range(decide_num_of_iter(len(tables_list), len(best_board),
                                      len(best_board[0]))):  # main loop of the function

        init_tables(tables_list)  # zero the location of all tables
        temp_board = init_board.copy()  # create a new empty board
        curr_min_distance = add_tables(temp_board, global_min_distance, tables_list, i)

        # in case the last run gave a better solution
        if curr_min_distance > global_min_distance:
            global_min_distance = curr_min_distance
            best_board = temp_board

    # in case the algorithm succeed in placing all tables
    if global_min_distance > 0:

        # for every table update his actual location according to the best board
        update_tables_location(best_board, tables_list)

        # fixing the solution by trying moving tables to better spot
        for _ in range(2):
            for i in range(0, len(tables_list)):
                tables_list[i].min_distance = 0
                fix(tables_list[i], best_board, tables_list[i].location[0], tables_list[i].location[1], tables_list)

        # update global distances
        global_min_distance = update_global_distance(tables_list)

        # create an image based on the solution matrix
        create_image(best_board, global_min_distance / 5)

        return global_min_distance / 5
    # if the algorithm failed to place all table
    else:
        return -1


"""add tables_algorithm is the function that in charge of adding all tables.
every iteration of the main algorithm (tables_arrangement_algorithm) this function is called.
every step the function tries to add the next table

Parameters
----------
board : matrix
    a representation of the restaurant
best_distance : int
    the best score so far in all iterations
tables: list
    list of all tables
rem_pts: list
    represent all points on the board the still empty

Returns
-------
min_distance: int
    the minimal distance in meters between the closest tables in the restaurant in the current iteration
"""


def add_tables(board, best_distance, tables, try_num):
    min_distance = np.inf

    for i in range(len(tables)):

        # find all possible points in the board for placing next table
        possible_spots = find_potential_spots(tables[i], board)

        # in case there are no spots to put the next table or the curr distance is lower than previous iterations
        if len(possible_spots) == 0 or best_distance > min_distance:
            return -1

        # finding the best spot to locate next table:
        best_spot_distance, best_spot_coordinate = find_best_spot(possible_spots, tables[i], tables, try_num)

        # update curr minimal distance between all tables:
        min_distance = min(best_spot_distance, min_distance)

        # place table in the best spot and check for unwanted collisions
        collision = locate_table(tables[i], board, best_spot_coordinate[0], best_spot_coordinate[1])

        # validation of the board, assert no collision happened when last table placed
        if collision:
            return -1
    return min_distance


"""put table locate a specific table in x,y coordinates given by the main function

Parameters
----------
table : Table
    a table to locate
board: matrix
x,y : int
    Represents the coordinates in which to place the table

Returns
-------
bull
    indicates if collisions with other tables occurred
"""


def locate_table(table, board, x, y):
    table.location = (x, y)

    # locate table
    for i in range(x, x + table.length):
        for j in range(y, y + table.width):

            # assure there is no collision with other tables
            if board[i][j] == 0:
                board[i][j] = table.table_number
            else:
                return True
    return False


"""find_potential_spots finds all spot available for place a specific table
every spot that will be return represents a point for placeing the top left corner of a table.
we assure spot is available for placing the op left corner by check if all other corners also empty.

Parameters
----------
table : Table
    A table for which we will look for potential spots
board : matrix

Returns
-------
possible_spots : list
    contains all potential spots for placing the curr table.
"""


def find_potential_spots(table, board):
    possible_spots = []

    # for every empty spot on the board, we check if by placing there the top left corner of -->
    # --> the curr table and then checks if we can also place all other corner
    for i in range(len(board)):
        for j in range(len(board[0])):
            if is_empty(table, board, i, j):
                possible_spots.append((i, j))
    return possible_spots


"""find_best_spot:
iter over the potential spots for placing a table, and for each one calculate the the distance
between this available spot to all other tables  that already putted.
we want to take the spot that placing there the curr table will be the best in terms of distances 
from all other tables.

* the first table is located randomly

Parameters
----------
table : Table
    a specific table we plan to locate
potential_spots : list
    list of potential points for placing this table
tables_list: list
    contains all tables
try_num: int
    indicates the iteration number of the main loop

Returns
-------
best_spot_dis_coordinate: tuple

best_spot_dis: int
    the minimal distance between the best spot to all other tables
"""


def find_best_spot(potential_spots, table, tables_list, try_num):
    # special case for the first table: return a random available spot
    if table.table_number == 1:

        # assure at least once we put the first table at the corner
        if try_num == 0:
            return np.inf, (potential_spots[0][0], potential_spots[0][1])
        n = np.random.randint(0, len(potential_spots) - 1)
        return np.inf, (potential_spots[n][0], potential_spots[n][1])

    best_spot_dis = 0
    best_spot_dis_coordinate = (-1, -1)

    # for every potential spot, we calculate the distance to all other tables that have already located.
    for (i, j) in potential_spots:
        curr_dis = dis_from_other_tables(table, i, j, tables_list, table.table_number - 1)

        # in case we found a better spot
        if curr_dis > best_spot_dis:
            best_spot_dis = curr_dis
            best_spot_dis_coordinate = (i, j)
    return best_spot_dis, best_spot_dis_coordinate


# simple function for calculate oclidis distance between to pairs of coordinates
def distance(j, i, r, k):
    return np.math.sqrt(((i - k) ** 2) + ((j - r) ** 2))


# function for calculating distance between two rectangles
def rect_distance(i1a, j1a, i2a, j2a, i1b, j1b, i2b, j2b):
    left = j2b < j1a
    right = j2a < j1b
    bottom = i1b > i2a
    top = i1a > i2b
    if top and left:
        return distance(i1a, j1a, i2b, j2b)
    elif left and bottom:
        return distance(i2a, j1a, i1b, j2b)
    elif bottom and right:
        return distance(i2a, j2a, i1b, j1b)
    elif right and top:
        return distance(i1a, j2a, i2b, j1b)
    elif left:
        return j1a - j2b
    elif right:
        return j1b - j2a
    elif bottom:
        return i1b - i2a
    elif top:
        return i1a - i2b
    else:  # rectangles intersect
        return 0.


"""create_init_board: 

create all parameters for the main function of the algorithm. get a matrix form the user,
and create the board which represents the restaurant.
we assume that tha max length and width of the restaurant is 10m.
we indeed create from 10X10 matrix a new board up to 50X50 ,
because the algorithm works with squares of 0.2 X 0.2 

Parameters
----------
initial_matrix : matrix
    matrix got from user's choices about his restaurant dimensions
tables : list

Returns
-------
board : matrix
    representation of the restaurant according to the user input

"""


def create_init_board(initial_matrix, tables):
    restaurant_length = 0
    restaurant_width = 0

    # finding the dimensions of the restaurant
    for i in range(10):
        for j in range(10):

            # light indicate a 1X1m spot in the user's restaurant ( a square he marked as chosen)
            if initial_matrix[i][j] == "light":
                if (i + 1) * 5 > restaurant_length:
                    restaurant_length = (i + 1) * 5
                if restaurant_width < (j + 1) * 5:
                    restaurant_width = (j + 1) * 5

    board = np.zeros((restaurant_length, restaurant_width))

    # update spots that the user excluded
    for i in range(10):
        for j in range(10):
            if i * 5 < restaurant_length and j * 5 < restaurant_width and initial_matrix[i][j] == "dark":
                for k in range(5):
                    for m in range(5):
                        board[i * 5 + k][j * 5 + m] = len(tables) + 1

    return board


# a useful function the returns num of iteration recommended for the main function
# the choice of how many depends on the number of tables and the size of the restaurant
def decide_num_of_iter(num_of_tables, res_length, res_width):
    if num_of_tables == 2:
        return min(40, 2 * res_width * res_length)
    if num_of_tables == 3:
        return min(30, 2 * res_width * res_length)
    elif num_of_tables == 4:
        return min(20, 2 * res_width * res_length)
    elif num_of_tables == 5:
        return min(15, 2 * res_width * res_length)
    if num_of_tables == 6:
        return min(10, 2 * res_width * res_length)
    else:
        return 5


# in charge of create a image of the restaurant and save it
def create_image(best_board, global_min):
    # create a new image
    plt.figure(2)

    plt.text(1, len(best_board) + 2, 'the distance between the closest tables is ' + str(global_min) + 'm')

    # remove boundaries
    plt.box(on=None)

    # set colour map
    c_map = plt.cm.gnuplot2

    # create image
    plt.imshow(best_board, interpolation='nearest', cmap=c_map, extent=[0, len(best_board[0]), 0, len(best_board)])

    # set axis labels
    plt.xlabel('m')
    plt.ylabel('m')

    # save and show picture
    plt.savefig('static/new_result.jpg')
    plt.show()


"""fix:
a recursive function that takes the best board returned from the algorithm
 and try to improve it by moving tables one spot every step.

Parameters
----------
table : Table
    a specific table we want to improve its location
board : matrix
    current board
tables_list: list
    contains all tables
i,j: int
    indicates the table's curr location
"""


def fix(table, board, i, j, tables_list):
    next_coor = (-1, -1)
    min_dis = table.min_distance

    # check if we can move the table one step in some direction
    empties_squares = [is_empty(table, board, i - 1, j), is_empty(table, board, i, j + 1),
                       is_empty(table, board, i + 1, j),
                       is_empty(table, board, i, j - 1)]

    # for every possible direction for moving table, calculate the distance from other tables
    if empties_squares[0]:
        up_dis = dis_from_other_tables(table, i - 1, j, tables_list, len(tables_list))
        if up_dis > min_dis:
            min_dis = up_dis
            next_coor = (i - 1, j)

    if empties_squares[1]:
        right_dis = dis_from_other_tables(table, i, j + 1, tables_list, len(tables_list))
        if right_dis > min_dis:
            min_dis = right_dis
            next_coor = (i, j + 1)

    if empties_squares[2]:
        down_dis = dis_from_other_tables(table, i + 1, j, tables_list, len(tables_list))
        if down_dis > min_dis:
            min_dis = down_dis
            next_coor = (i + 1, j)

    if empties_squares[3]:
        left_dis = dis_from_other_tables(table, i, j - 1, tables_list, len(tables_list))
        if left_dis > min_dis:
            min_dis = left_dis
            next_coor = (i, j - 1)

    # in case we found a direction which will improve curr table minimal distance
    # we locate it at the new spot ant try to continue to find a new direction
    if next_coor != (-1, -1):
        table.min_distance = min_dis
        remove_table(board, table.table_number)
        locate_table(table, board, next_coor[0], next_coor[1])
        fix(table, board, next_coor[0], next_coor[1], tables_list)


# indicates if a specific squre in the table is empty for placing specific table
def is_empty(table, board, i, j):
    index = table.table_number
    if (i >= 2) and (j >= 2) and (i + table.length + 1 < len(board)) and (j + table.width + 1 < len(board[0])):
        if ((board[i - 2, j - 2] == 0 or board[i - 2, j - 2] == index) and
                (board[i + table.length + 1][j - 2] == 0 or board[i + table.length + 1][j - 2] == index) and
                (board[i + table.length + 1][j + table.width + 1] == 0 or board[i + table.length + 1][
                    j + table.width + 1] == index) and
                (board[i - 2][j + table.width + 1] == 0 or board[i - 2][j + table.width + 1] == index) and

                (board[i + int(table.length / 2)][j + int(table.width / 2)] == 0 or board[i + int(table.length / 2)][
                    j + int(table.width / 2)] == index) and

                (board[i + int(table.length / 2)][j] == 0 or board[i + int(table.length / 2)][j] == index) and
                (board[i][j + int(table.width / 2)] == 0 or board[i][j + int(table.width / 2)] == index) and
                (board[i + int(table.length / 2)][j + table.width - 1] == 0 or board[i + int(table.length / 2)][
                    j + table.width - 1] == index) and
                (board[i + table.length - 1][j + int(table.width / 2)] == 0 or board[i + table.length - 1][
                    j + int(table.width / 2)] == index)):
            return True

    return False


# this function iterate over all tables requested and calculate the distance to them
def dis_from_other_tables(table, i, j, tables_list, check_until):
    dis_from_closest_table = np.inf
    for p in range(check_until):
        if p + 1 != table.table_number:

            curr_dis = rect_distance(i, j, i + table.length - 1, j + table.length - 1, tables_list[p].location[0],
                                     tables_list[p].location[1], tables_list[p].location[0] + tables_list[p].length - 1,
                                     tables_list[p].location[1] + tables_list[p].width - 1)

            if curr_dis < dis_from_closest_table:
                dis_from_closest_table = curr_dis
    return dis_from_closest_table


# remove specific table from the board
def remove_table(board, table_number):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i, j] == table_number:
                board[i][j] = 0


# every step we try to locate the tables in different posture (vertical/horizon)
def shuffle_tables_dimensions(tables_list):
    for i in range(len(tables_list)):
        tables_list[i].shuffle()

    pass


# find for every table its location on the board
def update_tables_location(board, tables_list):
    for i in range(len(tables_list)):

        # find top left corner of the table
        found = False
        for j in range(len(board)):
            if not found:
                for k in range(len(board[0])):
                    if board[j][k] == i + 1:
                        tables_list[i].location = (j, k)
                        found = True
                        break

        # find distance from the closest table

        tables_list[i].min_distance = 0


# avital procedure before every iteration of the main loop
def init_tables(tables_list):
    for i in range(len(tables_list)):
        tables_list[i].location = (-1, -1)
        tables_list[i].min_distance = np.inf
        tables_list[i].mass_center = (0, 0)
    pass


def update_global_distance(tables_list):
    min_dis = np.inf
    for table in tables_list:
        temp = dis_from_other_tables(table, table.location[0], table.location[1], tables_list,
                                     len(tables_list))
        min_dis = np.min((min_dis, temp))

    return min_dis
