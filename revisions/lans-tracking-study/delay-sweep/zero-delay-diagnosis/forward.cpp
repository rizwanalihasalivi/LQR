#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

constexpr int H = 1000;
constexpr int N = 14;
constexpr int P = 8 + 6*N;
constexpr double norm[] = {.1,.5,.05,.2};

extern "C" double score(int n, const double *p, const double *A, const double *B,
                        const double *dist, const double *preview, const double *initial,
                        double *grad, int horizon, double start_weight, const double *phase) {
    constexpr double clip = .5;
    std::fill_n(grad,P,0.);
    double loss = 0.;
    std::vector<std::array<double,4>> states(horizon+1);
    std::vector<double> controls(horizon);
    for(int c=0;c<n;++c) {
        auto &x0=states[0];
        std::copy_n(initial+4*c,4,x0.begin());
        double sse=x0[0]*x0[0];
        for(int t=0;t<horizon;++t) {
            auto &x=states[t];
            const double *q=preview+4*(c*H+t);
            const double *b=dist+4*(c*H+t);
            double u=0.;
            for(int j=0;j<4;++j) u+=p[j]*x[j]/norm[j]+p[4+j]*q[j];
            for(int h=0;h<N;++h) {
                const double *w=p+8+6*h;
                double z=0.;
                for(int j=0;j<4;++j) z+=w[j]*(x[j]/norm[j]-(phase[4*j]*q[0]+phase[4*j+1]*q[1]+phase[4*j+2]*q[2]+phase[4*j+3]*q[3]));
                u+=w[5]*(std::max(0.,z-w[4])-std::max(0.,-z-w[4]));
            }
            controls[t]=u;
            u=std::clamp(u,-clip,clip);
            auto &next=states[t+1];
            for(int j=0;j<4;++j) {
                next[j]=b[j]+B[j]*u;
                for(int k=0;k<4;++k) next[j]+=A[4*j+k]*x[k];
            }
            const double w=(t<30)?std::abs(start_weight):1.;
            sse+=w*next[0]*next[0];
        }
        const double root=std::sqrt(std::max(sse,1e-30)/1001.);
        if(!std::isfinite(root)) return 1e10;
        loss+=root;
        if(start_weight<0.) continue;
        std::array<double,4> lambda={0.,0.,0.,0.};
        const double sse_multiplier=1./(1001.*root*n);
        for(int t=horizon-1;t>=0;--t) {
            const auto &x=states[t];
            const auto &next=states[t+1];
            const double *q=preview+4*(c*H+t);
            lambda[0]+=sse_multiplier*((t<30)?std::abs(start_weight):1.)*next[0];
            double dU=0.;
            if(std::abs(controls[t])<clip) {
                for(int j=0;j<4;++j) dU+=B[j]*lambda[j];
            }
            std::array<double,4> new_lambda={0.,0.,0.,0.};
            for(int j=0;j<4;++j) {
                for(int k=0;k<4;++k) new_lambda[j]+=A[4*k+j]*lambda[k];
                new_lambda[j]+=dU*p[j]/norm[j];
                grad[j]+=dU*x[j]/norm[j];
                grad[4+j]+=dU*q[j];
            }
            if(dU!=0.) {
                for(int h=0;h<N;++h) {
                    const double *w=p+8+6*h;
                    double *g=grad+8+6*h;
                    double z=0.;
                    for(int j=0;j<4;++j) z+=w[j]*(x[j]/norm[j]-(phase[4*j]*q[0]+phase[4*j+1]*q[1]+phase[4*j+2]*q[2]+phase[4*j+3]*q[3]));
                    if(std::abs(z)>w[4]) {
                        const double sign=(z>0.)?1.:-1.;
                        for(int j=0;j<4;++j) {
                            g[j]+=dU*w[5]*(x[j]/norm[j]-(phase[4*j]*q[0]+phase[4*j+1]*q[1]+phase[4*j+2]*q[2]+phase[4*j+3]*q[3]));
                            new_lambda[j]+=dU*w[5]*w[j]/norm[j];
                        }
                        g[4]-=dU*w[5]*sign;
                        g[5]+=dU*sign*(std::abs(z)-w[4]);
                    }
                }
            }
            lambda=new_lambda;
        }
    }
    return loss/n;
}
