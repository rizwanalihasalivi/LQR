#include <algorithm>
#include <cmath>

extern "C" void evaluate(int n, int lag, int features, double authority, const double *p,
                         const double *A, const double *B, const double *gain,
                         const double *disturbance, const double *preview,
                         const double *initial, const double *first, const double *transform,
                         const double *head, double *out) {
    for (int c = 0; c < n; ++c) {
        double x[4], commands[1000];
        std::copy_n(initial + 4*c, 4, x);
        double sse = x[0]*x[0];
        bool failed = false;
        int k = 0;
        while (k < 1000) {
            const int offset = 1000*c + k;
            double u = first[c];
            if (k) {
                double base = 0, linear = 0;
                for (int i = 0; i < 4; ++i) {
                    base -= gain[i] * x[i];
                    linear += p[i] * x[i];
                }
                if (features == -6) {
                    const double *ref = preview + 6*offset;
                    double q[4];
                    const double lateral = ref[0] + x[0];
                    const double heading = ref[1]/25. + x[2];
                    for (int j = 0; j < 4; ++j) {
                        const double h = .01*(j+1);
                        const float raw = lateral + (25.*heading)*h - ref[j+2];
                        q[j] = static_cast<float>((raw - x[0] - (25.*x[2])*h)/(10.*h*h));
                    }
                    for (int i = 0; i < 3; ++i) {
                        double transformed = 0.;
                        for (int j = 0; j < 4; ++j) transformed += q[j]*transform[3*j+i];
                        linear += p[i+4] * static_cast<float>(transformed);
                    }
                    const float first_unit = 32. + (linear-base)/authority;
                    const float residual = head[0]*first_unit + head[1]*32. + head[2];
                    u = base + authority*std::clamp(residual, -1.f, 1.f);
                } else {
                    for (int i = 0; i < features; ++i)
                        linear += p[i+4] * preview[features*offset+i];
                    u = base + std::clamp(linear - base, -authority, authority);
                }
            }
            commands[k] = std::clamp(u, -.5, .5);
            double next[4];
            const double applied = k >= lag ? commands[k-lag] : 0.;
            for (int i = 0; i < 4; ++i) {
                next[i] = B[i]*applied + disturbance[4*offset+i];
                for (int j = 0; j < 4; ++j) next[i] += A[4*i+j]*x[j];
            }
            std::copy_n(next, 4, x);
            sse += x[0]*x[0];
            ++k;
            if (std::abs(x[0]) > 2.5 || std::abs(x[2]) > .5 || !std::isfinite(x[0])) {
                failed = true;
                break;
            }
        }
        const double penalty = std::max(2.5, std::abs(x[0]));
        out[3*c] = std::sqrt((sse + (1000-k)*penalty*penalty)/1001);
        out[3*c+1] = k;
        out[3*c+2] = !failed;
    }
}
