// Included by native_decision.cpp for the optional fixed-delay evaluator.
#pragma once

#include <chrono>
#include <cstdint>

namespace {
constexpr std::array<float, Inputs> ObservationLimit{32, 24, 16, 40, 0, 8, 25, 25, 32, 0, 0};
constexpr int PackedRows = 8;

struct Interval {
    double lo;
    double hi;
};

template<class Affine, std::size_t N>
Interval activation_range(const Affine& layer, int row, const std::array<Interval, N>& input) {
    double lo = layer.offset(row), hi = lo, magnitude = std::abs(lo);
    for (std::size_t j = 0; j < N; ++j) {
        const double w = layer.weight(row, j);
        const double a = w * input[j].lo;
        const double b = w * input[j].hi;
        lo += std::min(a, b);
        hi += std::max(a, b);
        magnitude += std::max(std::abs(a), std::abs(b));
    }
    // Float32 affine accumulation admits rounding in each product and sum.
    const double guard = 4 * (N + 1) * std::numeric_limits<float>::epsilon() * (1 + magnitude);
    return {std::max(0., lo - guard), std::max(0., hi + guard)};
}

class ActivationPlan {
    std::array<float, PackedRows * Width> weights{};
    std::array<float, PackedRows> bias{};
    std::array<int, PackedRows> original_row{};
    std::array<float, Width> fixed_hidden{};
    std::array<int, Width> dynamic_hidden{};
    int dynamic_size = 0;
    int size = 0;
    bool usable = true;

public:
    ActivationPlan(const Actor& actor, double speed, double previous_delay) {
        std::array<Interval, Inputs> obs;
        for (int i = 0; i < Inputs; ++i)
            obs[i] = {-ObservationLimit[i], ObservationLimit[i]};
        const float s = speed / 25.;
        const float d = previous_delay / 200.;
        obs[4] = {s, s};
        obs[9] = obs[10] = {d, d};
        std::array<Interval, Width> hidden;
        std::array<bool, Width> needed{};
        for (int i = 0; i < Width; ++i)
            hidden[i] = activation_range(actor.first, i, obs);
        for (int i = 0; i < Width; ++i) {
            if (actor.action.weight(0, i) == 0.f || activation_range(actor.second, i, hidden).hi == 0.)
                continue;
            if (size == PackedRows) {
                usable = false;
                return;
            }
            original_row[size] = i;
            bias[size] = actor.second.offset(i);
            for (int j = 0; j < Width; ++j) {
                const float w = actor.second.weight(i, j);
                weights[size * Width + j] = w;
                needed[j] |= w != 0.f;
            }
            ++size;
        }
        if (actor.first.can_use_sparse_rows()) {
            std::array<float, Inputs> constants{};
            constants[4] = s;
            constants[9] = constants[10] = d;
            for (int i = 0; i < Width; ++i) {
                if (!needed[i])
                    continue;
                bool varies = false;
                for (int j = 0; j < Inputs; ++j)
                    varies |= ObservationLimit[j] != 0 && actor.first.weight(i, j) != 0.f;
                if (varies)
                    dynamic_hidden[dynamic_size++] = i;
                else
                    fixed_hidden[i] = actor.first.activation(constants, i);
            }
        }
    }

    bool covers(const std::array<float, Inputs>& obs) const {
        if (!usable)
            return false;
        for (int i = 0; i < Inputs; ++i)
            if (ObservationLimit[i] != 0 && !(std::abs(obs[i]) <= ObservationLimit[i]))
                return false;
        return true;
    }

    float evaluate(const Actor& actor, const std::array<float, Inputs>& obs) const {
        auto hidden = fixed_hidden;
        if (actor.first.can_use_sparse_rows()) {
            for (int i = 0; i < dynamic_size; ++i) {
                const int row = dynamic_hidden[i];
                hidden[row] = actor.first.activation(obs, row);
            }
        } else {
            actor.first.apply(obs, hidden, true);
        }
        const int one = 1, in = Width, out = PackedRows;
        const float unit = 1.f;
        auto scores = bias;
        actor.second.gemm()("T", "N", &out, &one, &in, &unit, weights.data(), &in,
                            hidden.data(), &in, &unit, scores.data(), &out);
        float action = 0.f;
        for (int i = 0; i < size; ++i)
            action = std::fma(actor.action.weight(0, original_row[i]), std::max(scores[i], 0.f), action);
        return action + actor.action.offset(0);
    }

    unsigned rows() const { return usable ? size : Width; }
};

struct FixedActor {
    BoundActor base;
    double delay;
    ActivationPlan initial;
    ActivationPlan steady;

    FixedActor(const BoundActor& context, double fixed_delay)
        : base(context), delay(fixed_delay), initial(*context.actor, context.speed, 0.),
          steady(*context.actor, context.speed, fixed_delay) {}
};

double fixed_command(const FixedActor& fixed, const double* state, int tick) {
    const auto& context = fixed.base;
    if (tick < 0 || tick >= context.stride - 4)
        return std::numeric_limits<double>::quiet_NaN();
    const double observed_delay = tick == 0 ? 0. : fixed.delay;
    const auto obs = observe(state, context.reference, context.stride, tick, context.speed,
                             observed_delay, observed_delay, context.transform);
    const double base = context.hybrid ? lqr_command(state, context.gains) : 0.;
    const auto& plan = tick == 0 ? fixed.initial : fixed.steady;
    if (!plan.covers(obs))
        return steer(*context.actor, obs, base, context.residual_scale, context.gain);

    const float action = plan.evaluate(*context.actor, obs);
    const double residual = context.residual_scale * std::clamp(action, -1.f, 1.f);
    return std::clamp(base + context.gain * residual, -.5, .5);
}
}

extern "C" {
FixedActor* create_fixed_actor(const BoundActor* base, double delay) {
    return new FixedActor(*base, delay);
}

void destroy_fixed_actor(FixedActor* actor) {
    delete actor;
}

double steer_fixed_actor(const FixedActor* actor, const double* state, int tick) {
    return fixed_command(*actor, state, tick);
}

unsigned fixed_actor_rows(const FixedActor* actor, int phase) {
    return (phase ? actor->steady : actor->initial).rows();
}

// Slots share the same state buffer and timing loop. Slot 3 is the previous
// dense LANS core, measured with the same actor and fixed-delay observation.
void time_native_core(const FixedActor* lans, const FixedActor* ppo, const double* gains,
                      const double* states, int ticks, std::uint64_t* ns, double* commands) {
    using Clock = std::chrono::steady_clock;
    for (int tick = 0; tick < ticks; ++tick) {
        const double* state = states + 4 * tick;
        const double delay = tick == 0 ? 0. : lans->delay;
        for (int i = 0; i < 4; ++i) {
            const int slot = (i + tick) % 4;
            const int index = 4 * tick + slot;
            const auto start = Clock::now();
            switch (slot) {
            case 0: commands[index] = fixed_command(*lans, state, tick); break;
            case 1: commands[index] = fixed_command(*ppo, state, tick); break;
            case 2: commands[index] = steer_lqr(state, gains); break;
            default: commands[index] = steer_bound_actor(&lans->base, state, tick, delay, delay); break;
            }
            ns[index] = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
        }
    }
}
}
