//
// Created by spaceeye on 14.04.23.
//

#ifndef CANNONBALLISTICCALCULATOR_BALLISTIC_FUNCTIONS_H
#define CANNONBALLISTICCALCULATOR_BALLISTIC_FUNCTIONS_H

#include <cstdint>
#include <vector>
#include <cmath>
#include <array>
#include <thread>
#include <future>
#include <chrono>

#include <iostream>

#include <pybind11/stl.h>

#include "linspace.h"

inline int64_t time_in_air(float y0, float y, float Vy, int32_t max_steps=100000) {
    int64_t t = 0;

    if (y0 < y) {
        while (t < max_steps) {
            y0 += Vy;
            Vy = 0.99 * Vy - 0.05;

            t+=1;

            if (y0>y) {break;}
        }
    }

    while (t < max_steps) {
        y0 += Vy;
        Vy = 0.99 * Vy - 0.05;

        t += 1;

        if (y0 <= y) {return t;}
    }

    return -1;
}

inline double rad(auto deg) {return deg * (M_PI / 180);}

inline std::vector<std::pair<float, float>>
rough_pitch_estimation(const std::array<float, 3> & cannon,
                       const std::array<float, 3> & target,
                       float distance,
                       int32_t initial_speed,
                       int32_t length,
                       int32_t max_steps) {
    std::vector<std::pair<float, float>> delta_times;
    for (int tried_pitch = 60; tried_pitch >= -30; tried_pitch--) {
        auto tp_rad = rad(tried_pitch);

        auto Vw = std::cos(tp_rad) * initial_speed;
        auto Vy = std::sin(tp_rad) * initial_speed;

        auto x_coord_2d = length * std::cos(tp_rad);

        if (Vw == 0) { continue;}
        auto part = 1 - (distance - x_coord_2d) / (100 * Vw);
        if (part <= 0) { continue;}
        auto time_to_target = std::abs(std::log(part) / (-0.010050335853501));

        auto y_coord_end_of_barrel = cannon[1] + std::sin(tp_rad) * length;

        auto time_air = time_in_air(y_coord_end_of_barrel, target[1], Vy, max_steps);
        if (time_air < 0) { continue;}

        auto delta_t = std::abs(time_to_target - time_air);

        delta_times.emplace_back(delta_t, tried_pitch);
    }
    return delta_times;
}

inline std::vector<std::pair<float, float>>
py_rough_pitch_estimation(
                       pybind11::tuple & cannon_t,
                       pybind11::tuple & target_t,
                       float distance,
                       int32_t initial_speed,
                       int32_t length,
                       int32_t max_steps) {
    std::array<float, 3> cannon{
            cannon_t[0].cast<float>(),
            cannon_t[1].cast<float>(),
            cannon_t[2].cast<float>(),
    };
    std::array<float, 3> target{
            target_t[0].cast<float>(),
            target_t[1].cast<float>(),
            target_t[2].cast<float>(),
    };

    return rough_pitch_estimation(cannon, target, distance, initial_speed, length, max_steps);
}

inline std::vector<std::array<float, 3>>
fine_pitch_estimation(const std::array<float, 3> & cannon,
                      const std::array<float, 3> & target,
                      float distance,
                      int32_t initial_speed,
                      int32_t length,
                      float pitch,
                      int32_t num_refined = 20,
                      int32_t max_steps = 100000) {
    std::vector<std::array<float, 3>> delta_times;
    auto pitches = linspace<float>(pitch-1, pitch+1, num_refined);
    for (auto & tried_pitch: pitches) {
        auto tp_rad = rad(tried_pitch);

        auto Vw = std::cos(tp_rad) * initial_speed;
        auto Vy = std::sin(tp_rad) * initial_speed;

        auto x_coord_2d = length * std::cos(tp_rad);

        if (Vw == 0) { continue;}
        auto part = 1 - (distance - x_coord_2d) / (100 * Vw);
        if (part <= 0) { continue;}
        auto time_to_target = std::abs(std::log(part) / (-0.010050335853501));

        auto y_coord_end_of_barrel = cannon[1] + std::sin(tp_rad) * length;

        auto time_air = time_in_air(y_coord_end_of_barrel, target[1], Vy, max_steps);
        if (time_air < 0) { continue;}

        auto delta_t = std::abs(time_to_target - time_air);

        delta_times.push_back({delta_t, tried_pitch, (float)time_air});
    }
    return delta_times;
}

inline std::vector<std::array<float, 3>>
py_fine_pitch_estimation(
                      pybind11::tuple & cannon_t,
                      pybind11::tuple & target_t,
                      float distance,
                      int32_t initial_speed,
                      int32_t length,
                      float pitch,
                      int32_t num_refined = 20,
                      int32_t max_steps = 100000) {
    std::array<float, 3> cannon{
            cannon_t[0].cast<float>(),
            cannon_t[1].cast<float>(),
            cannon_t[2].cast<float>(),
    };
    std::array<float, 3> target{
            target_t[0].cast<float>(),
            target_t[1].cast<float>(),
            target_t[2].cast<float>(),
    };

    return fine_pitch_estimation(cannon, target, distance, initial_speed, length, pitch, num_refined, max_steps);
}

inline std::array<float, 3>
            try_pitch(const std::array<float, 3> & cannon,
                      const std::array<float, 3> & target,
                      int32_t power, int32_t length,
                      int32_t max_steps) {
    auto Dx = cannon[0] - target[0];
    auto Dz = cannon[2] - target[2];
    auto distance = std::sqrt(Dx * Dx + Dz * Dz);
    auto initial_speed = power;

    auto delta_times1 = rough_pitch_estimation(cannon, target, distance, initial_speed, length, max_steps);

    if (delta_times1.empty()) {return std::array<float, 3>{-1, -1, -1};}

    auto min_pair = std::min_element(delta_times1.begin(), delta_times1.end(),
                                        [](const auto & a, const auto & b){ return a.first < b.first;})[0];
    auto delta_times2 = fine_pitch_estimation(cannon, target, distance, initial_speed, length, min_pair.second);
    if (delta_times2.empty()) {return std::array<float, 3>{-1, -1, -1};}

    auto min_arr = std::min_element(delta_times2.begin(), delta_times2.end(),
                                    [](const auto & a, const auto & b){return a[0] < b[0];})[0];
    return min_arr;
}

inline std::array<float, 3>
py_try_pitch(
          pybind11::tuple & cannon_t,
          pybind11::tuple & target_t,
          int32_t power, int32_t length, int32_t max_steps) {
    std::array<float, 3> cannon{
            cannon_t[0].cast<float>(),
            cannon_t[1].cast<float>(),
            cannon_t[2].cast<float>(),
    };
    std::array<float, 3> target{
            target_t[0].cast<float>(),
            target_t[1].cast<float>(),
            target_t[2].cast<float>(),
    };

    return try_pitch(cannon, target, power, length, max_steps);
}

auto threaded_make_dataset(std::vector<std::array<std::array<float, 3>, 2>> * dataset,
                           int32_t charges,
                           int length,
                           int max_height_difference,
                           int start_pos,
                           int num_threads,
                           float * x_done,
                           int * y_done,
                           int max_steps,

                           float step = 1,

                           uint8_t * done = nullptr
                           ) {
    dataset->reserve(100000);
    for (int y = start_pos; y < max_height_difference/2; y+=num_threads) {
        float x = 1;

        while (true) {
            auto res = try_pitch({0, 0, 0}, {x, y, 0}, charges, length, max_steps);
            if (res[0] < 0) {
                break;
            } else {
                dataset->push_back(std::array<std::array<float, 3>, 2>{
                        std::array<float, 3>{(float )x, (float)y, 0},
                        res
                });
                x+=step;
            }
            (*x_done)+=step;
        }
        (*y_done)++;
    }

    for (int y = -start_pos-1; y > -max_height_difference/2; y-=num_threads) {
        float x = 1;

        while (true) {
            auto res = try_pitch({0, 0, 0}, {x, y, 0}, charges, length, max_steps);
            if (res[0] < 0) {
                break;
            } else {
                dataset->push_back(std::array<std::array<float, 3>, 2>{
                        std::array<float, 3>{(float )x, (float)y, 0},
                        res
                });
                x+=step;
            }
        }
        (*y_done)++;
        (*x_done)+=x;
    }

    *done = true;
}

auto make_dataset(int32_t charges,
                  int length,
                  int max_height_difference = 256,
                  int num_threads=16,
                  bool verbose=true,
                  int max_steps=100000,
                  float step=1
                  ) {
    using namespace std::chrono_literals;

    std::vector<std::vector<std::array<std::array<float, 3>, 2>>> threads_result;
    std::vector<std::pair<float, int>> threads_progress;
    std::vector<std::thread> threads;
    std::vector<uint8_t> done;

    threads_progress.resize(num_threads);
    threads_result.resize(num_threads);
    threads.resize(num_threads);
    done.resize(num_threads, false);

    for (int i = 0; i < num_threads; i++) {
        threads[i] = std::thread(
                                 threaded_make_dataset,
                                 &threads_result[i],
                                 charges, length, max_height_difference, i, num_threads,
                                 &threads_progress[i].first,
                                 &threads_progress[i].second,
                                 max_steps,
                                 step,
                                 &done[i]
        );

        threads[i].detach();
    }

    bool not_ended = true;
    while (not_ended) {
        not_ended = false;
        std::this_thread::sleep_for(1s);

        for (auto & is_done: done) {
            if (!is_done) {
                not_ended += true;
                break;
            }
        }
        if (!verbose) { continue;}

        double avg_y_done = 0;
        double avg_x_done = 0;

        for (const auto & [x_done, y_done]: threads_progress) {
            avg_x_done += x_done;
            avg_y_done += y_done;
        }

        std::cout << "avg x done: " << avg_x_done << " avg y done: " << avg_y_done << "\n";
    }

    return threads_result;
}




#endif //CANNONBALLISTICCALCULATOR_BALLISTIC_FUNCTIONS_H
