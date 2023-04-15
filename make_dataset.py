from CannonBallisticFunctions import make_dataset, time_in_air, rough_pitch_estimation, fine_pitch_estimation, try_pitch
import pickle

if __name__ == "__main__":
    res = make_dataset(7, 32, 512, 4, True, 100000, 0.25)
    with open("data025", mode="wb") as file:
        pickle.dump(res, file)

import test