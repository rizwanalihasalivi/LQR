#include <algorithm>
#include <cmath>

extern "C" void recovery(int cases,int horizon,int steps,int nodes,
    const double *A,const double *B,const double *K,const double *force,
    const double *refs,const double *initial,const double *T,const double *p,
    const double *directions,const double *th,const double *coeff,
    const double *reference_map,double *out,const double *switching,const double *preview_weights) {
    const double scale[]={.1,.5,.05,.2};
    for(int j=0;j<cases;++j){
        double x[4];
        std::copy_n(initial+4*j,4,x);
        double squared=x[0]*x[0];
        bool failed=false;
        int t=0;
        for(;t<steps;++t){
            int offset=horizon*j+t;
            const double *r=refs+7*offset;
            double pre[4],channels[3]={0.,0.,0.};
            for(int k=0;k<4;++k){
                double h=.01*(k+1);
                const float first=r[0]+x[0]+r[2]*(r[1]+x[2])*h-r[k+3];
                pre[k]=static_cast<float>((first-x[0]-r[2]*x[2]*h)/(10*h*h));
            }
            for(int q=0;q<3;++q){
                for(int k=0;k<4;++k)channels[q]+=pre[k]*T[3*k+q];
                channels[q]=static_cast<float>(channels[q]);
            }
            double v[4],base=0.,u=0.;
            for(int k=0;k<4;++k){
                base-=K[k]*x[k];
                u+=p[k]*x[k];
                double center=0.;
                for(int q=0;q<3;++q)center+=channels[q]*reference_map[4*q+k];
                v[k]=(x[k]-center)/scale[k];
            }
            for(int q=0;q<3;++q)u+=p[4+q]*channels[q];
            double additional[8];
            if(switching){
                const double displacement=(x[0]+.008*x[1])/.01;
                const double hinges[]={.10,.28,.56,1.0};
                for(int k=0;k<4;++k){
                    additional[2*k]=std::max(displacement-hinges[k],0.);
                    additional[2*k+1]=std::max(-displacement-hinges[k],0.);
                }
            }
            for(int m=0;m<nodes;++m){
                double z=-th[m];
                for(int k=0;k<4;++k)z+=v[k]*directions[4*m+k];
                if(switching)
                    for(int k=0;k<8;++k)z+=additional[k]*switching[8*m+k];
                if(preview_weights)
                    for(int k=0;k<3;++k)z+=channels[k]*preview_weights[3*m+k];
                u+=coeff[m]*std::max(z,0.);
            }
            u=std::clamp(base+std::clamp(u-base,-1.5,1.5),-.5,.5);
            double newx[4];
            for(int k=0;k<4;++k){
                newx[k]=force[4*offset+k]+B[k]*u;
                for(int i=0;i<4;++i)newx[k]+=A[4*k+i]*x[i];
            }
            std::copy_n(newx,4,x);
            squared+=x[0]*x[0];
            if(std::abs(x[0])>2.5||std::abs(x[2])>.5||!std::isfinite(squared)){
                ++t;
                failed=true;
                break;
            }
        }
        const double penalty=failed?std::max(2.5,std::abs(x[0])):0.;
        out[3*j]=std::sqrt((squared+(horizon-t)*penalty*penalty)/(horizon+1));
        out[3*j+1]=!failed;
        out[3*j+2]=t;
    }
}
