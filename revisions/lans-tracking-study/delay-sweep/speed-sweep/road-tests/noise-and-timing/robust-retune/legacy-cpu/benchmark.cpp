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
#include <thread>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cpuid.h>
#elif defined(__linux__)
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
constexpr std::array<double, Controllers> ReferenceLimits{.003, 1e-5, 1e-12, 3e-5};
constexpr std::array<const char*, Controllers> ControllerNames{"LANS", "PPO", "LQR", "MPC"};
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
#if defined(_WIN32)
    if (__get_cpuid_max(0x80000000U, nullptr) >= 0x80000004U) {
        std::array<char, 49> brand{};
        for (unsigned i = 0; i < 3; ++i) {
            std::array<unsigned, 4> registers{};
            __cpuid(0x80000002U + i, registers[0], registers[1], registers[2], registers[3]);
            std::memcpy(brand.data() + 16 * i, registers.data(), 16);
        }
        std::cerr << "Processor: " << brand.data() << '\n';
    }
    DWORD_PTR allowed = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system) || !allowed)
        throw std::runtime_error("could not read Windows CPU affinity");
    DWORD_PTR bit = 1;
    unsigned cpu = 0;
    while (!(allowed & bit)) {
        bit <<= 1;
        ++cpu;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), bit))
        throw std::runtime_error("could not pin the benchmark thread");
    std::cerr << "Pinned Windows logical CPU " << cpu << '\n';
#elif defined(__linux__)
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

class PreparedCase {
    const Case& data;
    BoundActor hybrid, neural;
    FixedActor fixed_lans, fixed_ppo;
    Mpc mpc;

public:
    PreparedCase(const Case& c, const Actor* lans, const Actor* ppo, const double* transform,
                 const MatchedPlant& plant)
        : data(c),
          hybrid{lans, c.gains.data(), c.road.data(), transform, int(c.stride), 1,
                 c.speed / 3.6, c.lans_scale, std::exp(-c.decay * std::max(0., double(c.delay) - 5.))},
          neural{ppo, c.gains.data(), c.road.data(), transform, int(c.stride), 0,
                 c.speed / 3.6, c.ppo_scale, 1.},
          fixed_lans(hybrid, c.delay), fixed_ppo(neural, c.delay), mpc(plant) {}

    void reset() { mpc.reset(); }

    double decide(int slot, unsigned tick) {
        const Frame& frame = data.frames[tick];
        switch (slot) {
        case 0: return fixed_command(fixed_lans, frame.state.data(), tick);
        case 1: return fixed_command(fixed_ppo, frame.state.data(), tick);
        case 2: return steer_lqr(frame.state.data(), data.lqr.data());
        default: return mpc.decide(frame.state, frame.previous, data.road.data(), data.stride, tick);
        }
    }
};

void run_case(const Case& c, const Actor* lans, const Actor* ppo, const double* transform,
              const MatchedPlant& plant, int repeats, std::array<std::vector<double>, Controllers>& times,
              std::array<double, Controllers>& error) {
    PreparedCase workload(c, lans, ppo, transform, plant);

    for (int i = 0; i < 40; ++i)
        for (int slot = 0; slot < Controllers; ++slot)
            workload.decide(slot, 0);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        workload.reset();
        for (unsigned tick = 0; tick < c.ticks; ++tick) {
            for (int i = 0; i < Controllers; ++i) {
                const int slot = (i + tick) % Controllers;
                const auto begin = std::chrono::steady_clock::now();
                const double command = workload.decide(slot, tick);
                const auto end = std::chrono::steady_clock::now();
                if (!std::isfinite(command))
                    throw std::runtime_error("invalid steering command");
                times[slot].push_back(std::chrono::duration<double, std::micro>(end - begin).count());
                error[slot] = std::max(error[slot], std::abs(command - c.frames[tick].expected[slot]));
            }
        }
    }
}

class EnergyCounter {
    std::string file;
    std::uint64_t modulus = 0, last = 0, accumulated = 0;

    static std::uint64_t read(const std::string& path) {
        std::uint64_t value;
        if (!(std::ifstream(path) >> value))
            throw std::runtime_error("cannot read hardware energy counter: " + path);
        return value;
    }

public:
    explicit EnergyCounter(const char* path) : file(path ? path : "") {
        if (enabled()) {
            const auto slash = file.rfind('/');
            const auto directory = slash == std::string::npos ? "" : file.substr(0, slash + 1);
            std::string domain;
            if (!(std::ifstream(directory + "name") >> domain) || domain.rfind("package-", 0) != 0)
                throw std::runtime_error("select a package energy counter, not a core or other subdomain");
            modulus = read(directory + "max_energy_range_uj");
            if (!modulus)
                throw std::runtime_error("hardware energy range must be positive");
        }
    }

    bool enabled() const { return !file.empty(); }

    void begin() {
        if (enabled())
            last = read(file);
    }

    void sample() {
        if (!enabled())
            return;
        const auto current = read(file);
        if (current >= modulus || last >= modulus)
            throw std::runtime_error("hardware energy reading is outside the counter range");
        accumulated += current >= last ? current - last : modulus - last + current;
        last = current;
    }

    double joules() const { return accumulated / 1.e6; }
};

void run_workload(const char* name, double seconds, const char* counter, Reader& tape,
                  const Actor* lans, const Actor* ppo, const double* transform,
                  const MatchedPlant& slow, const MatchedPlant& fast) {
    const auto selected = std::find(ControllerNames.begin(), ControllerNames.end(), std::string(name));
    const bool idle = std::string(name) == "IDLE";
    if (!idle && selected == ControllerNames.end())
        throw std::runtime_error("workload controller must be LANS, PPO, LQR, MPC or IDLE");
    if (!std::isfinite(seconds) || seconds < .1 || seconds > 3600.)
        throw std::runtime_error("workload duration must be between 0.1 and 3600 seconds");
    const int slot = selected - ControllerNames.begin();
    std::vector<Case> cases;
    cases.reserve(24);
    for (int i = 0; i < 24; ++i)
        cases.emplace_back(tape);
    tape.finish();
    std::vector<std::unique_ptr<PreparedCase>> workloads;
    for (const auto& c : cases)
        workloads.push_back(std::make_unique<PreparedCase>(c, lans, ppo, transform, c.speed == 90 ? slow : fast));

    // Check the full shared tape and warm up the selected path before measuring.
    if (!idle) {
        for (std::size_t i = 0; i < workloads.size(); ++i) {
            if (slot == 3)
                workloads[i]->reset();
            for (unsigned tick = 0; tick < cases[i].ticks; ++tick) {
                const double u = workloads[i]->decide(slot, tick);
                if (!std::isfinite(u) || std::abs(u - cases[i].frames[tick].expected[slot]) > ReferenceLimits[slot])
                    throw std::runtime_error("energy workload reference steering check failed");
            }
        }
    }

    EnergyCounter energy(counter);
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::duration<double>;
    const auto unix_ns = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    };
    double checksum = 0.;
    std::uint64_t decisions = 0;
    energy.begin();
    const auto start_unix = unix_ns();
    const auto start = Clock::now();
    const auto deadline = start + Duration(seconds);
    auto next_read = start + std::chrono::seconds(1);
    const auto read_if_due = [&] {
        if (energy.enabled() && Clock::now() >= next_read) {
            energy.sample();
            next_read = Clock::now() + std::chrono::seconds(1);
        }
    };
    if (idle) {
        while (Clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            read_if_due();
        }
    } else {
        do {
            for (std::size_t i = 0; i < workloads.size(); ++i) {
                if (slot == 3)
                    workloads[i]->reset();
                for (unsigned tick = 0; tick < cases[i].ticks; ++tick)
                    checksum += workloads[i]->decide(slot, tick);
                decisions += cases[i].ticks;
                read_if_due();
            }
        } while (Clock::now() < deadline);
    }
    const double elapsed = Duration(Clock::now() - start).count();
    const auto end_unix = unix_ns();
    energy.sample();
    std::cout << "controller,decisions,elapsed_s,start_unix_ns,end_unix_ns,package_energy_j,checksum\n"
              << name << ',' << decisions << ',' << std::setprecision(12) << elapsed << ','
              << start_unix << ',' << end_unix << ',';
    if (energy.enabled())
        std::cout << energy.joules();
    std::cout << ',' << checksum << '\n';
}

}

int main(int argc, char** argv) {
    try {
        const bool workload_mode = argc > 1 && std::string(argv[1]) == "--workload";
        if ((!workload_mode && (argc < 2 || argc > 3)) ||
            (workload_mode && (argc < 5 || argc > 6)))
            throw std::runtime_error("usage: legacy-timing TAPE [REPEATS] or --workload TAPE CONTROLLER SECONDS [ENERGY_UJ]");
        const int repeats = !workload_mode && argc == 3 ? std::stoi(argv[2]) : 3;
        if (repeats < 1 || repeats > 100)
            throw std::runtime_error("repeat count must be between 1 and 100");
        pin_cpu();
        Reader tape(argv[workload_mode ? 2 : 1]);
        if (tape.read<std::uint32_t>() != 2 || tape.read<std::uint32_t>() != 24)
            throw std::runtime_error("unsupported legacy workload");
        const auto lans = load_actor(tape);
        const auto ppo = load_actor(tape);
        const auto transform = tape.read<std::array<double, 16>>();
        const MatchedPlant slow(tape), fast(tape);
        if (slow.speed != 90 || fast.speed != 150)
            throw std::runtime_error("unexpected MPC model speeds");
        if (workload_mode) {
            run_workload(argv[3], std::stod(argv[4]), argc == 6 ? argv[5] : nullptr,
                         tape, lans.get(), ppo.get(), transform.data(), slow, fast);
            return 0;
        }
        std::array<std::vector<double>, Controllers> times;
        std::array<double, Controllers> error{};
        for (int i = 0; i < 24; ++i) {
            const Case c(tape);
            run_case(c, lans.get(), ppo.get(), transform.data(), c.speed == 90 ? slow : fast, repeats, times, error);
        }
        tape.finish();
        // Float32 accumulation order can differ on the scalar backend. The
        // reference limit is checked over every timed state and reported.
        for (int i = 0; i < Controllers; ++i) {
            if (error[i] > ReferenceLimits[i]) {
                std::cerr << ControllerNames[i] << " max reference steering difference " << error[i] << " rad\n";
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
            std::cout << i + 1 << ',' << ControllerNames[slot] << ',' << t.size() << ',' << std::setprecision(12)
                      << percentile(t, .5) << ',' << percentile(t, .95) << ','
                      << std::accumulate(t.begin(), t.end(), 0.) / t.size() << ',' << error[slot] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Legacy timing failed: " << error.what() << '\n';
        return 1;
    }
}
