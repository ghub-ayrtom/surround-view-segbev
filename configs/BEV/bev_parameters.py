# vehicle_width = 200
# vehicle_height = 660

near_shift_width = 200
near_shift_height = 200

# square_width = 115
# square_height = 90

# circle_square_width = 2 * square_width
# circle_square_height = 2 * square_height

# corner_boards_width = 6 * square_width
# corner_boards_height = 5 * square_height

# pattern_width = vehicle_width + 2 * near_shift_width + 2 * corner_boards_width
# pattern_height = vehicle_height + 2 * near_shift_height + 2 * corner_boards_height

far_shift_width = 100
far_shift_height = 100

total_width = 585 + 2 * far_shift_width
total_height = 677 + 2 * far_shift_height

# total_width = pattern_width + 2 * far_shift_width
# total_height = pattern_height + 2 * far_shift_height

vehicle_leftside_edges_x = far_shift_width + 50 + near_shift_width
vehicle_rightside_edges_x = total_width - vehicle_leftside_edges_x

vehicle_topside_edges_y = far_shift_height + 50 + near_shift_height
vehicle_bottomside_edges_y = total_height - vehicle_topside_edges_y

# vehicle_leftside_edges_x = far_shift_width + corner_boards_width + near_shift_width
# vehicle_topside_edges_y = far_shift_height + corner_boards_height + near_shift_height

projection_shapes = {
    'camera_front_left': (total_height, vehicle_leftside_edges_x), 
    'camera_front': (total_width, vehicle_topside_edges_y), 
    'camera_front_blind': (total_width, vehicle_topside_edges_y), 
    'camera_front_right': (total_height, vehicle_leftside_edges_x), 
    'camera_rear': (total_width, vehicle_topside_edges_y), 
}

projection_src_points = {
    'camera_front_left': [
        (377, 61), 
        (933, 60), 
        (170, 208), 
        (1141, 208), 
    ], 

    'camera_front': [
        (373, 410), 
        (937, 410), 
        (0, 531), 
        (1304, 531), 
    ], 

    'camera_front_blind': [
        (272, 25), 
        (1035, 25), 
        (0, 89), 
        (1304, 89), 
    ], 

    'camera_front_right': [
        (377, 61), 
        (933, 60), 
        (170, 208), 
        (1141, 208), 
    ], 

    'camera_rear': [
        (307, 60), 
        (999, 60), 
        (0, 236), 
        (1304, 236), 
    ], 
}

projection_dst_points = {
    'camera_front_left': [
        (far_shift_height + 204, far_shift_width), 
        (far_shift_height + 475, far_shift_width), 
        (far_shift_height + 204, far_shift_width + 145), 
        (far_shift_height + 475, far_shift_width + 145), 

        # (far_shift_width, far_shift_height + corner_boards_height + square_height), 
        # (far_shift_width + 5 * square_width, far_shift_height + corner_boards_height + square_height), 
        # (far_shift_width, far_shift_height + corner_boards_height + (5 * square_height + 2 * circle_square_height)), 
        # (far_shift_width + 5 * square_width, far_shift_height + corner_boards_height + (5 * square_height + 2 * circle_square_height)), 
    ], 

    'camera_front': [
        (far_shift_width + 109, far_shift_height), 
        (far_shift_width + 470, far_shift_height), 
        (far_shift_width + 109, far_shift_height + 155), 
        (far_shift_width + 470, far_shift_height + 155), 
    ], 

    'camera_front_blind': [
        (far_shift_width + 109, far_shift_height), 
        (far_shift_width + 470, far_shift_height), 
        (far_shift_width + 109, far_shift_height + 110.5), 
        (far_shift_width + 470, far_shift_height + 110.5), 
    ], 

    'camera_front_right': [
        (far_shift_height + 204, far_shift_width + 8), 
        (far_shift_height + 475, far_shift_width + 5), 
        (far_shift_height + 204, far_shift_width + 150), 
        (far_shift_height + 475, far_shift_width + 150), 
    ], 

    'camera_rear': [
        (far_shift_width + 115, far_shift_height), 
        (far_shift_width + 475, far_shift_height), 
        (far_shift_width + 115, far_shift_height + 150), 
        (far_shift_width + 475, far_shift_height + 150), 
    ], 
}
