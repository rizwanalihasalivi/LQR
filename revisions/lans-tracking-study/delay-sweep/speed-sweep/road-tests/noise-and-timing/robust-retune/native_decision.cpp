#include <algorithm>
#include <array>
#include <cmath>

namespace {
constexpr int Width = 64;
constexpr int Inputs = 11;

// Match Torch CPU's float32 affine evaluation, including the bias addition.
// Some fitted latency gates subtract quantities large relative to the command.
using Sgemm = void (*)(const char*, const char*, const int*, const int*, const int*,
                       const float*, const float*, const int*, const float*, const int*,
                       const float*, float*, const int*);

template<int In, int Out>
class Layer {
    std::array<float, In * Out> weights;
    std::array<float, Out> bias;
    Sgemm multiply;

public:
    Layer(const float* w, const float* b, Sgemm gemm) : multiply(gemm) {
        std::copy_n(w, weights.size(), weights.begin());
        std::copy_n(b, bias.size(), bias.begin());
    }

    void apply(const std::array<float, In>& x, std::array<float, Out>& y, bool relu) const {
        const int one = 1, in = In, out = Out;
        const float unit = 1.f;
        y = bias;
        multiply("T", "N", &out, &one, &in, &unit, weights.data(), &in,
                 x.data(), &in, &unit, y.data(), &out);
        if (relu)
            for (int i = 0; i < out; ++i)
                y[i] = std::max(y[i], 0.f);
    }
};

struct Actor {
    Layer<Inputs, Width> first;
    Layer<Width, Width> second;
    Layer<Width, 1> action;
};

double lqr_command(const double* state, const double* gains) {
    double command = 0.;
    for (int i = 0; i < 4; ++i)
        command -= state[i] * gains[i];
    return command;
}

std::array<float, Inputs> observe(const double* state, const double* reference,
                                  int stride, int tick, double speed, double delay,
                                  double mean_delay, const double* transform) {
    std::array<float, Inputs> obs{static_cast<float>(state[0] / .1),
                                  static_cast<float>(state[1] / .5),
                                  static_cast<float>(state[2] / .05),
                                  static_cast<float>(state[3] / .2),
                                  static_cast<float>(speed / 25.)};
    float preview[4];
    const double position = reference[tick] + state[0];
    const double heading = reference[stride + tick] / speed + state[2];
    for (int i = 0; i < 4; ++i) {
        const double horizon = .01 * (i + 1);
        const float raw = position + speed * heading * .01 * (i + 1) - reference[tick + i + 1];
        preview[i] = (raw - state[0] - speed * state[2] * horizon) /
                     (10. * (horizon * horizon));
    }
    for (int i = 0; i < 4; ++i) {
        double sum = 0.;
        for (int j = 0; j < 4; ++j)
            sum += preview[j] * transform[j * 4 + i];
        obs[5 + i] = sum;
    }
    obs[9] = delay / 200.;
    obs[10] = mean_delay / 200.;
    return obs;
}
}

extern "C" {
Actor* create_actor(const float* w1, const float* b1, const float* w2,
                    const float* b2, const float* w3, const float* b3, Sgemm gemm) {
    return new Actor{{w1, b1, gemm}, {w2, b2, gemm}, {w3, b3, gemm}};
}

void destroy_actor(Actor* actor) {
    delete actor;
}

double steer_lqr(const double* state, const double* gains) {
    return std::clamp(lqr_command(state, gains), -.5, .5);
}

double steer_actor(const Actor* actor, const double* state, const double* gains,
                   const double* reference, const double* transform, int stride, int tick,
                   double speed, double previous_delay, double mean_delay,
                   double residual_scale, double decay, double delay, int hybrid) {
    const auto obs = observe(state, reference, stride, tick, speed, previous_delay, mean_delay, transform);
    std::array<float, Width> hidden, features;
    std::array<float, 1> action;
    actor->first.apply(obs, hidden, true);
    actor->second.apply(hidden, features, true);
    actor->action.apply(features, action, false);
    const double base = hybrid ? lqr_command(state, gains) : 0.;
    const double gain = hybrid ? std::exp(-decay * std::max(0., delay - 5.)) : 1.;
    const double residual = residual_scale * std::clamp(action[0], -1.f, 1.f);
    return std::clamp(base + gain * residual, -.5, .5);
}
}
