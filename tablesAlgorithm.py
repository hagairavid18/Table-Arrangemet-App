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


def shuffle_tables_dimensions(tables_list):
    for i in range(len(tables_list)):
        tables_list[i].shuffle()

    pass


def tables_arrangement_algorithm(tables_list, restaurant_matrix):
    # initialize essential parameters
    init_avail_pts, res_length, res_width = create_all_points(restaurant_matrix)
    best_distance = -1
    best_board = np.zeros((res_length , res_length ))

    # main loop of the function
    for i in range(decide_num_of_iter(len(tables_list), res_length, res_width)):
        shuffle_tables_dimensions(tables_list)
        board = np.zeros((res_length , res_width ))
        remain_pts = init_avail_pts.copy()
        curr_distance = add_tables(board, best_distance, tables_list, remain_pts, i)

        # in case the last run gave a better solution
        if curr_distance > best_distance:
            best_distance = curr_distance
            best_board = board
    # in case the algorithm succeed in placing all tables
    if best_distance > 0:
        # change colour of points that shouldn't be in the restaurant
        add_non_res_points(init_avail_pts, res_length, res_width, best_board, len(tables_list) + 1)

        # create an image based on the solution matrix
        create_image(best_board, best_distance / 5)
        return best_distance / 5
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


def add_tables(board, best_distance, tables, rem_pts, try_num):
    min_distance = np.inf

    for i in range(len(tables)):

        # find all possible points in the board for putting the next table
        possibleSpots = find_potential_spots(tables[i], rem_pts)

        # in case there are no spots to put the next table or the curr distance is lower than previous iterations
        if len(possibleSpots) == 0 or best_distance > min_distance:
            return -1

        # finding the best spot to locate next table:
        best_spot_distance, best_spot_coordinate = calculate_distances_center(possibleSpots, tables[i], tables, try_num)

        # update curr minimal distance between all tables:
        min_distance = min(best_spot_distance, min_distance)

        # place table in the best spot ant check for unwanted collisions
        collision = locate_table(tables[i], rem_pts, board, best_spot_coordinate[0], best_spot_coordinate[1])

        # validation of the board, assert no collision happened when last table placed
        if collision:
            return -1
    return min_distance


"""put table locate a specific table in x,y coordiante given by the main function

Parameters
----------
table : Table
    a table to locate
rem_points : list
    list of empty points (each points is tuple of 3) in the board
board: matrix
x,y : int
    Represents the coordinates in which to place the table

Returns
-------
bull
    indicates if collisions with other tables occurred
"""


def locate_table(table, rem_points, board, x, y):
    # 'mass center' is the center of the table, needed for future calculations
    table.mass_center = ((x + table.length) / 2, (y + table.width) / 2)

    # locate table
    for i in range(x, x + table.length):
        for j in range(y, y + table.width):
            if (i, j, 0) in rem_points:
                next_point = rem_points.pop(rem_points.index((i, j, 0)))

                # assure this spot is indeed available
                if board[next_point[0]][next_point[1]] == 0:
                    board[next_point[0]][next_point[1]] = table.table_number
                else:
                    return True
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
rem_points : list
    list of empty points in the board

Returns
-------
possible_spots : list
    contains all potential spots for placing the curr table.
"""


def find_potential_spots(table, rem_pts):
    possible_spots = []

    # for every empty spot on the board, we check if by placing there the top left corner of -->
    # --> the curr table and then checks if we can also place all other corner
    for i in range(len(rem_pts)):
        if ((rem_pts[i][0], rem_pts[i][1], 0) in rem_pts) and (
                (rem_pts[i][0] + table.length - 1, rem_pts[i][1], 0) in rem_pts) and (
                (rem_pts[i][0], rem_pts[i][1] + table.width - 1, 0) in rem_pts) and (
                (rem_pts[i][0] + table.length - 1, rem_pts[i][1] + table.width - 1, 0) in rem_pts) and (
                (rem_pts[i][0] + int(table.length / 2), rem_pts[i][1] + int(table.width / 2), 0) in rem_pts) and(
                (rem_pts[i][0] + int(table.length / 2), rem_pts[i][1] - 2, 0) in rem_pts) and (
                (rem_pts[i][0] + int(table.length / 2), rem_pts[i][1] + table.width + 1, 0) in rem_pts) and (
                (rem_pts[i][0] - 2, rem_pts[i][1] + int(table.width / 2), 0) in rem_pts) and (
                (rem_pts[i][0] + table.length + 1 , rem_pts[i][1] + int(table.width / 2), 0) in rem_pts) and (
                (rem_pts[i][0] - 2, rem_pts[i][1] - 2, 0) in rem_pts) and (
                (rem_pts[i][0] + table.length + 1, rem_pts[i][1] + table.width + 1, 0) in rem_pts) and (
                (rem_pts[i][0] + table.length + 1, rem_pts[i][1] - 2, 0) in rem_pts)and (
                (rem_pts[i][0] - 2, rem_pts[i][1] + table.width + 1, 0) in rem_pts):

            possible_spots.append(rem_pts[i])
    return possible_spots


"""calculate_distances_center:
iter over the potential spots for placing a table, and for each one calculate the the distance
between this available spot to all others tables's centers that already putted
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

Returns
-------
bull
    indicates if collisions with other tables occurred
"""


def calculate_distances_center(potential_spots, table, tables_list, try_num):
    # special case for the first table: return a random available spot
    if table.table_number == 1:
        if try_num == 0:
            return np.inf, (potential_spots[0][0], potential_spots[0][1])
        n = np.random.randint(0, len(potential_spots) - 1)
        return np.inf, (potential_spots[n][0], potential_spots[n][1])

    best_spot_dis = 0
    best_spot_dis_coordinate = (-1, -1)

    # for every potential spot, we calculate the distance to all other tables that have already located.
    for (i, j, k) in potential_spots:
        dis_from_closest_table = np.inf
        for p in range(table.table_number - 1):
            curr_dis = distance((i + table.length) / 2, (j + table.width) / 2, tables_list[p].mass_center[0],
                                tables_list[p].mass_center[1])
            if curr_dis < dis_from_closest_table:
                dis_from_closest_table = curr_dis

        # in case we found a better spot
        if dis_from_closest_table > best_spot_dis:
            best_spot_dis = dis_from_closest_table
            best_spot_dis_coordinate = (i, j)

    return best_spot_dis, best_spot_dis_coordinate


# a non useful function right now, more precise than the previews one but takes too long to calculate
def calculate_distances(possible_spots, selected_points, table, try_num):
    global_min_distance = 0
    global_min_distance_coordinate = (-1, -1)

    if table.table_number == 1:
        if try_num == 0 :
            return np.inf, (possible_spots[0][0], possible_spots[0][1])
        n = np.random.randint(0, len(possible_spots) - 1)

        return np.inf, (possible_spots[n][0], possible_spots[n][1])

    for (i, j, k) in possible_spots:
        min_distance = np.inf
        for r in range(0, table.length):
            for m in range(0, table.width):
                for (o, p, q) in selected_points:
                    if distance(i + r, j + m, o, p) < min_distance:
                        min_distance = distance(i + r, j + m, o, p)
        if min_distance > global_min_distance:
            global_min_distance = min_distance
            global_min_distance_coordinate = (i, j)

    return global_min_distance, global_min_distance_coordinate


# simple function for calculate oclidis distance between to pairs of coordinates
def distance(i, j, k, r):
    return np.math.sqrt(((i - k) ** 2) + ((j - r) ** 2))


"""create_all_points: 

create all parameters for the main function of the algorithm. get a matrix form the user,
and create a list of spots.
we assume that tha max length and width of the restaurant is 10m.
we indeed create from 10X10 matrix a new list that has up to 50X50 tuples,
because the algorithm works with squares of 0.2 X 0.2 

Parameters
----------
initial_matrix : matrix
    matrix got from user's choices about his restaurant dimensions

Returns
-------
init_points : array
    contain all spots in the restaurant according for the data came from the user
restaurant_length,restaurant_width : int
    help future function to make a sketch of the restaurant
"""


def create_all_points(initial_matrix):
    restaurant_length = 0
    restaurant_width = 0
    init_points = []
    for i in range(10):
        for j in range(10):

            # light indicate a 1X1m spot in the user's restaurant ( a square he marked as chosen)
            if initial_matrix[i][j] == "light":
                if (i + 1) * 5 > restaurant_length:
                    restaurant_length = (i + 1) * 5
                if restaurant_width < (j + 1) * 5:
                    restaurant_width = (j + 1) * 5
                for k in range(5):
                    for m in range(5):
                        init_points.append((int(((float(i) * 5) + k)), int(((float(j) * 5) + m)), 0))

    return init_points, restaurant_length, restaurant_width


# a useful function the returns num of iteration recommended for the main function
# the choice of how many depends on the number of tables and the size of the restaurant
def decide_num_of_iter(num_of_tables, res_length, res_width):
    if num_of_tables == 3:
        return min(250, 2 * res_width * res_length)
    elif num_of_tables == 4:
        return min(80, 2 * res_width * res_length)
    elif num_of_tables == 5:
        return min(50, 2 * res_width * res_length)
    if num_of_tables == 6:
        return min(30, 2 * res_width * res_length)
    else:
        return 50


# change colour of points that shouldn't be in the restaurant
def add_non_res_points(init_avail_pts, res_length, res_width, best_board, num):
    for i in range(res_length):
        for j in range(res_width):
            if (i, j, 0) not in init_avail_pts:
                best_board[i][j] = num


# in charge of create a image of the restaurant and save it
def create_image(best_board, best_distance):
    # create a new image
    plt.figure(2)

    #plt.text(1, len(best_board) + 2, 'the distance between the closest tables is ' + str(best_distance) + 'm')

    # remove boundaris
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
