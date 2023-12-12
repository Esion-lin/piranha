#pragma once
#include <cstddef>
#include "../gpu/DeviceData.h"
#include "../gpu/ring.cuh"
#include "../globals.h"

template<uint32_t D>
class PolyRing{
    public:
    RSS<uint64_t> data_s;
    PolyRing(size_t lens):data_s(0){
        data_s.getShare(0)->resize(D*lens);
        data_s.getShare(1)->resize(D*lens);
    }
};
class DotForRing{
    public:
    bool ismalicious = false;
    DotForRing(){}
    //need_gen: need generate r_z
    void online(const RSS<uint64_t> &x, const RSS<uint64_t> &y, RSS<uint64_t>& output){
        PolyRing<32> xx(x.size());
        PolyRing<32> yy(y.size());
        DeviceData<uint64_t> summed(xx.data_s.size());
        DeviceData<uint64_t> temp(32);
        DeviceData<uint64_t> temp2(32);
        summed.zero();
        summed += *yy.data_s.getShare(0);
        summed += *yy.data_s.getShare(1);
        gpu::ringDot(xx.data_s.getShare(0), &summed, &temp, x.size(), 32);
        gpu::ringDot(xx.data_s.getShare(1), yy.data_s.getShare(0), &temp2, x.size(), 32);
        temp+=temp2;
        reshare(temp, output);
    }
    void verify(){}
    private:
    
};

// template <uint32_t LENS, uint32_t D>
class PolyRingVerify{
    //N: pairs
    //n fan-ins
    //D degree
    public:
    // void set_up(const AShare<T> x, const AShare<T> y, AShare<T>& output, bool need_gen){
    //     random_T<T>(r_list, N/2);
    // }
    PolyRingVerify(uint32_t N_){
        N = N_;
    }
    uint32_t N;
    bool verify(PolyRing<32>* x, PolyRing<32>* y, PolyRing<32> z, PolyRing<32>* x_p, PolyRing<32>* y_p, PolyRing<32>& z_p){
        SetCoeff(P, 32, 1);
        //set N/2 linear function f^{N/2}_m
        PolyRing<32> ff[3][N/2], gg[3][N/2], h[3];
        DeviceData<uint64_t> delta ;
        ZZX d0 = MulMod(delta - 1, delta - 2, P) /2;
        ZZX d1 = MulMod(delta, 2 - delta, P);
        ZZX d2 = MulMod(delta - 1, delta, P) /2;
        for(int i = 0; i < N /2; i++){

            ff[0][i] = x[2*i];
            ff[1][i] = x[2*i + 1];
            papply<ZZX>(ff[2][i].r, ff[1][i].r, ff[0][i].r, LENS, [](ZZX a, ZZX b) -> ZZX{return 2*a - b;});
            papply<ZZX>(ff[2][i].r_1, ff[1][i].r_1, ff[0][i].r_1, LENS, [](ZZX a, ZZX b) -> ZZX{return 2*a - b;});
            papply<ZZX>(ff[2][i].r_2, ff[1][i].r_2, ff[0][i].r_2, LENS, [](ZZX a, ZZX b) -> ZZX{return 2*a - b;});
            
            gg[0][i] = y[2*i];
            gg[1][i] = y[2*i + 1];
            papply<ZZX>(gg[2][i].r, gg[1][i].r, gg[0][i].r, LENS, [](ZZX a, ZZX b) -> ZZX{return 2*a - b;});
            papply<ZZX>(gg[2][i].r_1, gg[1][i].r_1, gg[0][i].r_1, LENS, [](ZZX a, ZZX b) -> ZZX{return 2*a - b;});
            papply<ZZX>(gg[2][i].r_2, gg[1][i].r_2, gg[0][i].r_2, LENS, [](ZZX a, ZZX b) -> ZZX{return 2*a - b;});
            inter(1-delta, delta, ff[0][i], ff[1][i], P, x_p[i]);
            inter(1-delta, delta, gg[0][i], gg[1][i], P, y_p[i]);
            
        }
        dot1.ismalicious = true;
        dot1.set_up(ff[0], gg[0], h[0], N/2,true);
        dot1.online(ff[0], gg[0], h[0], N/2);
        // h[1] = z - h[0];
        papply<ZZX>(h[1].r, z.r, h[0].r, LENS, [](ZZX a, ZZX b) -> ZZX{return a - b;});
        papply<ZZX>(h[1].r_1, z.r_1, h[0].r_1, LENS, [](ZZX a, ZZX b) -> ZZX{return a - b;});
        papply<ZZX>(h[1].r_2, z.r_2, h[0].r_2, LENS, [](ZZX a, ZZX b) -> ZZX{return a - b;});

        dot2.set_up(ff[2], gg[2], h[2], N/2,true);
        dot2.online(ff[2], gg[2], h[2], N/2);

        inter(d0, d1, d2, h[0], h[1], h[2], P, z_p);
        // for(int i = 0; i < N /2; i++){
        //     // release_ashare<T>(ff[0][i]);
        //     // release_ashare<T>(ff[1][i]);
        //     release_ashare<T>(ff[2][i]);
        //     // release_ashare<T>(gg[0][i]);
        //     // release_ashare<T>(gg[1][i]);
        //     release_ashare<T>(gg[2][i]);
        // }
        // release_ashare<T>(h[0]);
        // release_ashare<T>(h[1]);
        // release_ashare<T>(h[2]);
        return true;
    }
    private:
    
    DotForRing<LENS>  dot1,dot2;
};

// class Multiple_verify{

// }