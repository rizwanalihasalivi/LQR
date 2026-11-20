// This translation unit reuses the existing controller implementation with a
// scalar SGEMM backend that needs neither MKL nor CPU-specific instructions.
#define LANS_PACK_FIRST_LAYER
#define LANS_FIXED_DELAY_PLAN
#include "native/native_decision.cpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

extern "C" {
#include "osqp.h"
#include "mpc90_workspace.h"
#include "mpc150_workspace.h"
}

namespace {
constexpr int Horizon = 20;
constexpr int Extended = 4 * Horizon;
constexpr int Controllers = 4;
using State = std::array<double, 4>;

class Reader {
    std::ifstream stream;

public:
    explicit Reader(const char* filename) : stream(filename, std::ios::binary) {
        char magic[8];
        if (!stream.read(magic, sizeof(magic)) || std::memcmp(magic, "LQRCPU01", 8))
            throw std::runtime_error("invalid legacy timing tape");
    }

    template<class T> T read() {
        T data;
        if (!stream.read(reinterpret_cast<char*>(&data), sizeof(T)))
            throw std::runtime_error("truncated legacy timing tape");
        return data;
    }

    template<class T> std::vector<T> vector(std::size_t size) {
        std::vector<T> data(size);
        if (!stream.read(reinterpret_cast<char*>(data.data()), size * sizeof(T)))
            throw std::runtime_error("truncated legacy timing tape");
        return data;
    }

    void finish() {
        if (stream.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("unexpected data after legacy timing tape");
    }
};

void scalar_gemm(const char*, const char*, const int* out, const int*, const int* in,
                 const float*, const float* weights, const int*, const float* input,
                 const int*, const float*, float* output, const int*) {
    for (int i = 0; i < *out; ++i) {
        float sum = 0.f;
        for (int j = 0; j < *in; ++j)
            sum = static_cast<float>(double(weights[i * *in + j]) * input[j] + sum);
        output[i] += sum;
    }
}

using ActorOwner = std::unique_ptr<Actor, decltype(&destroy_actor)>;

ActorOwner load_actor(Reader& file) {
    const auto w1 = file.read<std::array<float, 64 * 11>>();
    const auto b1 = file.read<std::array<float, 64>>();
    const auto w2 = file.read<std::array<float, 64 * 64>>();
    const auto b2 = file.read<std::array<float, 64>>();
    const auto w3 = file.read<std::array<float, 64>>();
    const auto b3 = file.read<std::array<float, 1>>();
    return ActorOwner(create_actor(w1.data(), b1.data(), w2.data(), b2.data(),
                                  w3.data(), b3.data(), scalar_gemm), destroy_actor);
}

struct MatchedPlant {
    unsigned speed;
    std::array<double, 16> a;
    State g, h;
    std::array<double, Extended * 4> t;
    std::array<double, Horizon * Extended> sw;
    std::array<double, Horizon * Horizon> inv;

    explicit MatchedPlant(Reader& f)
        : speed(f.read<std::uint32_t>()), a(f.read<decltype(a)>()),
          g(f.read<State>()), h(f.read<State>()), t(f.read<decltype(t)>()),
          sw(f.read<decltype(sw)>()), inv(f.read<decltype(inv)>()) {}
};

class Mpc {
    const MatchedPlant& plant;
    OSQPSolver& solver;
    std::array<OSQPFloat, Horizon> last{};

public:
    explicit Mpc(const MatchedPlant& p) : plant(p), solver(p.speed == 90 ? mpc90_solver : mpc150_solver) {
        reset();
    }

    void reset() {
        osqp_cold_start(&solver);
        last.fill(0.);
    }

    double decide(const State& state, double previous, const double* road, int stride, int tick) {
        State forecast{};
        std::array<double, Extended> prediction;
        const double speed = plant.speed / 3.6;
        for (int i = 0; i < Horizon; ++i) {
            const int k = std::min(tick + i, stride - 1);
            State next{};
            for (int j = 0; j < 4; ++j) {
                for (int d = 0; d < 4; ++d)
                    next[j] += plant.a[4 * j + d] * forecast[d];
                next[j] += plant.g[j] * (road[2 * stride + k] / speed);
                next[j] += plant.h[j] * (road[3 * stride + k] / speed);
                const int row = 4 * i + j;
                prediction[row] = next[j];
                for (int d = 0; d < 4; ++d)
                    prediction[row] += plant.t[4 * row + d] * state[d];
            }
            forecast = next;
        }
        std::array<OSQPFloat, Horizon> gradient{};
        for (int i = 0; i < Horizon; ++i)
            for (int j = 0; j < Extended; ++j)
                gradient[i] += plant.sw[i * Extended + j] * prediction[j];
        gradient[0] -= .01 * previous;
        std::array<OSQPFloat, Horizon> command{};
        bool bounded = false;
        for (int i = 0; i < Horizon; ++i) {
            for (int j = 0; j < Horizon; ++j)
                command[i] -= plant.inv[i * Horizon + j] * gradient[j];
            bounded |= std::abs(command[i]) > .5;
        }
        if (bounded) {
            for (auto& value : gradient)
                value *= 2.;
            if (osqp_update_data_vec(&solver, gradient.data(), nullptr, nullptr) ||
                osqp_warm_start(&solver, last.data(), nullptr) || osqp_solve(&solver) ||
                solver.info->status_val != OSQP_SOLVED)
                throw std::runtime_error("matched MPC did not converge");
            std::copy_n(solver.solution->x, Horizon, command.begin());
        }
        for (int i = 0; i < Horizon - 1; ++i)
            last[i] = command[i + 1];
        last.back() = command.back();
        return std::clamp(command[0], -.5, .5);
    }
};

struct Frame {
    State state;
    double previous;
    std::array<double, Controllers> expected;
};

struct Case {
    unsigned speed, delay, ticks, stride;
    double lans_scale, ppo_scale, decay;
    State gains, lqr;
    std::vector<double> road;
    std::vector<Frame> frames;

    explicit Case(Reader& f)
        : speed(f.read<std::uint32_t>()), delay(f.read<std::uint32_t>()),
          ticks(f.read<std::uint32_t>()), stride(f.read<std::uint32_t>()),
          lans_scale(f.read<double>()), ppo_scale(f.read<double>()), decay(f.read<double>()),
          gains(f.read<State>()), lqr(f.read<State>()) {
        if ((speed != 90 && speed != 150) || (delay != 0 && delay != 200 && delay != 1000) ||
            !ticks || stride > 10006 || ticks > stride || stride - ticks < 4)
            throw std::runtime_error("unsupported case dimensions");
        road = f.vector<double>(4 * stride);
        frames.reserve(ticks);
        for (unsigned i = 0; i < ticks; ++i)
            frames.push_back({f.read<State>(), f.read<double>(), f.read<std::array<double, Controllers>>()});
    }
};

void pin_cpu() {
#ifdef __linux__
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
        throw std::runtime_error("could not read CPU affinity");
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &allowed))
            continue;
        cpu_set_t pinned;
        CPU_ZERO(&pinned);
        CPU_SET(i, &pinned);
        if (sched_setaffinity(0, sizeof(pinned), &pinned))
            throw std::runtime_error("could not pin the benchmark thread");
        std::cerr << "Pinned Linux CPU " << i << '\n';
        return;
    }
    throw std::runtime_error("no CPU available to benchmark");
#endif
}

void run_case(const Case& c, const Actor* lans, const Actor* ppo, const double* transform,
              const MatchedPlant& plant, int repeats, std::array<std::vector<double>, Controllers>& times,
              std::array<double, Controllers>& error) {
    const int stride = c.stride;
    const double speed = c.speed / 3.6;
    const double gain = std::exp(-c.decay * std::max(0., double(c.delay) - 5.));
    const BoundActor hybrid{lans, c.gains.data(), c.road.data(), transform, stride, 1,
                            speed, c.lans_scale, gain};
    const BoundActor neural{ppo, c.gains.data(), c.road.data(), transform, stride, 0,
                            speed, c.ppo_scale, 1.};
    const FixedActor fixed_lans(hybrid, c.delay), fixed_ppo(neural, c.delay);
    Mpc mpc(plant);
    const auto decide = [&](int slot, unsigned tick) {
        const Frame& f = c.frames[tick];
        switch (slot) {
        case 0: return fixed_command(fixed_lans, f.state.data(), tick);
        case 1: return fixed_command(fixed_ppo, f.state.data(), tick);
        case 2: return steer_lqr(f.state.data(), c.lqr.data());
        default: return mpc.decide(f.state, f.previous, c.road.data(), stride, tick);
        }
    };

    for (int i = 0; i < 40; ++i)
        for (int slot = 0; slot < Controllers; ++slot)
            decide(slot, 0);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        mpc.reset();
        for (unsigned tick = 0; tick < c.ticks; ++tick) {
            for (int i = 0; i < Controllers; ++i) {
                const int slot = (i + tick) % Controllers;
                const auto begin = std::chrono::steady_clock::now();
                const double command = decide(slot, tick);
                const auto end = std::chrono::steady_clock::now();
                if (!std::isfinite(command))
                    throw std::runtime_error("invalid steering command");
                times[slot].push_back(std::chrono::duration<double, std::micro>(end - begin).count());
                error[slot] = std::max(error[slot], std::abs(command - c.frames[tick].expected[slot]));
            }
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("usage: legacy-timing TAPE [REPEATS]");
        const int repeats = argc == 3 ? std::stoi(argv[2]) : 3;
        if (repeats < 1 || repeats > 100)
            throw std::runtime_error("repeat count must be between 1 and 100");
        pin_cpu();
        Reader tape(argv[1]);
        if (tape.read<std::uint32_t>() != 2 || tape.read<std::uint32_t>() != 24)
            throw std::runtime_error("unsupported legacy workload");
        const auto lans = load_actor(tape);
        const auto ppo = load_actor(tape);
        const auto transform = tape.read<std::array<double, 16>>();
        const MatchedPlant slow(tape), fast(tape);
        if (slow.speed != 90 || fast.speed != 150)
            throw std::runtime_error("unexpected MPC model speeds");
        std::array<std::vector<double>, Controllers> times;
        std::array<double, Controllers> error{};
        for (int i = 0; i < 24; ++i) {
            const Case c(tape);
            run_case(c, lans.get(), ppo.get(), transform.data(), c.speed == 90 ? slow : fast, repeats, times, error);
        }
        tape.finish();
        // Float32 accumulation order can differ on the scalar backend. The
        // reference limit is checked over every timed state and reported.
        const std::array<double, Controllers> limit{.003, 1e-5, 1e-12, 3e-5};
        const std::array<const char*, Controllers> names{"LANS", "PPO", "LQR", "MPC"};
        for (int i = 0; i < Controllers; ++i) {
            if (error[i] > limit[i]) {
                std::cerr << names[i] << " max reference steering difference " << error[i] << " rad\n";
                throw std::runtime_error("reference steering difference exceeded the limit");
            }
            std::sort(times[i].begin(), times[i].end());
        }
        const auto percentile = [](const std::vector<double>& values, double p) {
            const double rank = p * (values.size() - 1);
            const std::size_t lo = rank, hi = std::ceil(rank);
            return values[lo] + (rank - lo) * (values[hi] - values[lo]);
        };
        std::array<int, Controllers> rank{0, 1, 2, 3};
        std::sort(rank.begin(), rank.end(), [&](int a, int b) {
            return percentile(times[a], .5) < percentile(times[b], .5);
        });
        std::cout << "rank,controller,decisions,median_us,p95_us,mean_us,max_reference_error_rad\n";
        for (int i = 0; i < Controllers; ++i) {
            const int slot = rank[i];
            const auto& t = times[slot];
            std::cout << i + 1 << ',' << names[slot] << ',' << t.size() << ',' << std::setprecision(12)
                      << percentile(t, .5) << ',' << percentile(t, .95) << ','
                      << std::accumulate(t.begin(), t.end(), 0.) / t.size() << ',' << error[slot] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Legacy timing failed: " << error.what() << '\n';
        return 1;
    }
}
