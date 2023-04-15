import math
import numpy as np


def prepare_data(data):
    collapsed_data = []
    for thread_result in data:
        collapsed_data += thread_result

    inputs = []
    outputs = []

    for item in collapsed_data:
        coords = item[0]

        distance = math.sqrt(coords[0] * coords[0]
                             + coords[2] * coords[2])

        height_difference = coords[1]

        rad_angle = math.atan2(coords[1], coords[0]) / math.pi

        inputs.append((distance, height_difference, coords[0], coords[1], rad_angle))

        outputs.append(item[1])

    max_distance = max(inputs, key=lambda x: abs(x[0]))[0]
    max_height   = max(inputs, key=lambda x: abs(x[1]))[1]
    max_x        = max(inputs, key=lambda x: abs(x[2]))[2]
    max_y        = max(inputs, key=lambda x: abs(x[3]))[3]
    max_rad      = max(inputs, key=lambda x: abs(x[4]))[4]

    inputs = [(it[0]/max_distance,
               it[1]/max_height,
               it[2]/max_x,
               it[3]/max_y,
               it[4]/max_rad,
               ) for it in inputs]

    max_delta_t     = max(outputs, key=lambda x: abs(x[0]))[0]
    max_tried_pitch = max(outputs, key=lambda x: abs(x[1]))[1]
    max_airtime     = max(outputs, key=lambda x: abs(x[2]))[2]

    outputs = [(it[0]/max_delta_t,
                it[1]/max_tried_pitch,
                it[2]/max_airtime) for it in outputs]

    inputs  = np.array(inputs)
    outputs = np.array(outputs)

    return {
        "inputs":inputs,
        "outputs":outputs,

        "max_distance":max_distance,
        "max_height":max_height,
        "max_delta_t":max_delta_t,
        "max_tried_pitch":max_tried_pitch,
        "max_airtime":max_airtime
    }

if __name__ == "__main__":
    import pickle
    with open("data", mode="rb") as file:
        data = pickle.load(file)

    print(prepare_data(data))