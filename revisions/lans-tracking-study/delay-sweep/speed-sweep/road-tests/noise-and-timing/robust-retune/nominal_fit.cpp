#include <algorithm>
#include <cmath>

extern "C" void score(int episodes, int ticks, const double* a, const double* b,
                      const double* lqr, const double* initial, const double* disturbance,
                      const double* preview, const double* noise, const double* weights,
                      const double* hinges, const double* thresholds, int count, double* errors) {
    const double state_scale[] = {.1, .5, .05, .2};
    for (int episode = 0; episode < episodes; ++episode) {
        double state[4];
        std::copy_n(initial + 4 * episode, 4, state);
        double square_error = state[0] * state[0];
        int tick = 0;
        for (; tick < ticks; ++tick) {
            const int sample = episode * ticks + tick;
            double observed[4];
            double command = 0., baseline = 0.;
            for (int i = 0; i < 4; ++i) {
                observed[i] = state[i] + noise[4 * sample + i];
                command += weights[i] * observed[i];
                baseline -= lqr[i] * observed[i];
            }
            for (int j = 0; j < 3; ++j)
                command += weights[4 + j] * preview[3 * sample + j];
            for (int j = 0; j < count; ++j) {
                double projection = 0.;
                for (int i = 0; i < 4; ++i)
                    projection += observed[i] * hinges[4 * j + i] / state_scale[i];
                command += weights[7 + j] * (std::max(projection - thresholds[j], 0.) -
                                              std::max(-projection - thresholds[j], 0.));
            }
            command = std::clamp(baseline + std::clamp(command - baseline, -1.5, 1.5), -.5, .5);
            double next[4];
            for (int i = 0; i < 4; ++i) {
                next[i] = b[i] * command + disturbance[4 * sample + i];
                for (int j = 0; j < 4; ++j)
                    next[i] += a[4 * i + j] * state[j];
            }
            std::copy_n(next, 4, state);
            square_error += state[0] * state[0];
            if (!std::isfinite(square_error) || std::abs(state[0]) > 2.5 || std::abs(state[2]) > .5) {
                const double penalty = std::max(2.5, std::abs(state[0]));
                square_error += (ticks - tick - 1) * penalty * penalty;
                break;
            }
        }
        errors[episode] = std::sqrt(square_error / (ticks + 1));
    }
}
