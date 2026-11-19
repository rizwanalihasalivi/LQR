#include <algorithm>
#include <cmath>
#include <vector>
extern "C" void rollout(int n, int lag, const double *params, const double *initial_policy,
                        const double *A, const double *B, const double *disturbance,
                        const double *preview, const double *initial, double *out) {
    for (int c = 0; c < n; ++c) {
        double x[4];
        std::copy_n(initial + 4*c, 4, x);
        double sse = x[0]*x[0];
        double queue[1000];
        int k = 0;
        bool failed = false;
        while (k < 1000) {
            const int offset = 1000*c + k;
            const double *p = (k == 0 && lag > 0) ? initial_policy : params;
            double u = 0;
            for (int i=0;i<4;++i) u += p[i]*x[i];
            for (int i=0;i<3;++i) u += p[4+i]*preview[3*offset+i];
            queue[k] = std::clamp(u, -.5, .5);
            const double applied = (k >= lag) ? queue[k-lag] : 0;
            double next[4];
            for (int i=0;i<4;++i) {
                next[i] = B[i]*applied + disturbance[4*offset+i];
                for (int j=0;j<4;++j) next[i] += A[4*i+j]*x[j];
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
