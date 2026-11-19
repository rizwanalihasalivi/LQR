#include <algorithm>
#include <cmath>
#include <vector>

extern "C" void replay(int cases, int horizon, int steps, int lag, int hinges,
                         double authority, const double *p, const double *startup,
                         const double *A, const double *B, const double *gain,
                         const double *disturbance, const double *reference,
                         const double *initial, const double *first, const double *transform,
                         const double *directions, const double *thresholds, double *results,
                         const double *tail) {
    std::vector<double> queue(horizon);
    const double scale[] = {.1, .5, .05, .2};
    for (int j = 0; j < cases; ++j) {
        double state[4];
        std::copy_n(initial + 4*j, 4, state);
        double squared = state[0]*state[0];
        bool failed = false;
        int t = 0;
        for (; t < steps; ++t) {
            const int offset = horizon*j + t;
            const double *ref = reference + 7*offset;
            double u = first[j];
            if (t || !lag || startup) {
                const double *weights = t == 0 && lag ? startup : p;
                const int count = t == 0 && lag && startup ? 14 : hinges;
                const bool apply_tail = tail && (!lag || !t);
                double base = 0., command = 0., whitened[4], scaled[4], correction = 0.;
                const double lateral = ref[0] + state[0];
                const double heading = ref[1] + state[2];
                for (int i = 0; i < 4; ++i) {
                    base -= gain[i]*state[i];
                    command += weights[i]*state[i];
                    if (apply_tail) correction += tail[i]*state[i];
                    const double h = .01*(i+1);
                    const float preview = lateral + (ref[2]*heading)*h - ref[i+3];
                    whitened[i] = static_cast<float>((preview - state[0] - ref[2]*state[2]*h)/(10*h*h));
                    scaled[i] = static_cast<float>(state[i]/scale[i]);
                }
                for (int i = 0; i < 3; ++i) {
                    double v = 0.;
                    for (int k = 0; k < 4; ++k) v += whitened[k]*transform[3*k+i];
                    command += weights[i+4]*static_cast<float>(v);
                    if (apply_tail) correction += tail[i+4]*static_cast<float>(v);
                }
                for (int i = 0; i < count; ++i) {
                    double v = 0.;
                    for (int k = 0; k < 4; ++k) v += scaled[k]*directions[4*i+k];
                    const double hinge = std::max(v-thresholds[i], 0.) -
                                         std::max(-v-thresholds[i], 0.);
                    command += weights[i+7]*hinge;
                    if (apply_tail) correction += tail[i+7]*hinge;
                }
                if (apply_tail) {
                    if (tail[21] < 0.) {
                        const double threshold = -tail[21];
                        command += std::max(correction-threshold,0.) -
                                   std::max(-correction-threshold,0.);
                    } else {
                        const double gate = .25 * (1. - std::abs(state[0])/tail[21]);
                        command += 1.5 * (std::max(gate+correction/1.5,0.)-std::max(gate,0.));
                    }
                }
                if (t == 0 && lag) command = base + (authority/1.5)*(command-base);
                u = base + std::clamp(command-base, -authority, authority);
            }
            queue[t] = std::clamp(u, -.5, .5);
            const double applied = t >= lag ? queue[t-lag] : 0.;
            double next[4];
            for (int i = 0; i < 4; ++i) {
                next[i] = disturbance[4*offset+i] + B[i]*applied;
                for (int k = 0; k < 4; ++k) next[i] += A[4*i+k]*state[k];
            }
            std::copy_n(next, 4, state);
            squared += state[0]*state[0];
            if (std::abs(state[0]) > 2.5 || std::abs(state[2]) > .5 || !std::isfinite(squared)) {
                failed = true;
                ++t;
                break;
            }
        }
        const double penalty = failed ? std::max(2.5, std::abs(state[0])) : 0.;
        results[3*j] = std::sqrt((squared+(horizon-t)*penalty*penalty)/(horizon+1));
        results[3*j+1] = !failed;
        results[3*j+2] = t;
    }
}
