#pragma once
#include <cstddef>
#include "../gpu/DeviceData.h"
#include "../gpu/ring.cuh"
#include "../globals.h"

#define Degree 32

extern int partyNum;
enum Party { PARTY_A, PARTY_B, PARTY_C };
Party nextParty(int party){
    switch(party) {
        case PARTY_A:
            return PARTY_B;
        case PARTY_B:
            return PARTY_C;
        default: // PARTY_C 
            return PARTY_A;
    }	
}
Party prevParty(int party){
    switch(party) {
        case PARTY_A:
            return PARTY_C;
        case PARTY_B:
            return PARTY_A;
        default: // PARTY_C
            return PARTY_B;
	}		
}
template<class T>
class MSS_Single{
    public:
    DeviceData<T>r_1,r_2,r;
    MSS_Single():r_1(1), r_2(1), r(1){

    }
    MSS_Single(uint32_t len):r_1(len), r_2(len), r(len){
    }
};
template<class T>
class MSS_Multiplication{
    public:
    MSS_Multiplication(){};
    void set_up(MSS_Single<T>& x, MSS_Single<T>& y, MSS_Single<T>& z){
        if(partyNum == PARTY_A){
            DeviceData<T> sh1(x.r_1.size());
            DeviceData<T> r_xy(x.r_1.size());
            //share [x.r*y.r]
            r_xy.zero();y.r.zero();
            r_xy += x.r_1;
            r_xy += x.r_2;
            y.r += y.r_1;
            y.r += y.r_2;
            r_xy *= y.r;
            r_xy -= sh1;
            r_xy.transmit(PARTY_C);
            r_xy.join();
        }else if(partyNum == PARTY_B){
            
        }else{
            z.r.receive(PARTY_A);
            z.r.join();
        }
    }
    void online(MSS_Single<T>& x, MSS_Single<T>& y, MSS_Single<T>& z){
        //m_x * m_y - m_x [r_y] - m_y [r_x] + [r_x r_y] - [r_z]
        if(partyNum != PARTY_A){
            DeviceData<T> temp(x.r_1.size());
            gpu::mssOnline_1d(&x.r_1, &y.r_1, &x.r_2, &y.r_2, &z.r, &z.r_2, &temp, sqrt(x.r_1.size()),sqrt(x.r_1.size()));
            if(partyNum == PARTY_B){
                z.r_1.zero();
                z.r_1 += x.r_1;
                z.r_1 *= y.r_1;
                temp += z.r_1;
                //reveal m_z;
                temp.transmit(PARTY_C);
                z.r_1.receive(PARTY_C);
                temp.join();
                z.r_1.join();
                z.r_1 += temp;
            }
            else{
                temp.transmit(PARTY_B);
                z.r_1.receive(PARTY_B);
                temp.join();
                z.r_1.join();
                z.r_1 += temp;
            }
        }
    }
    
};

template<uint32_t D>
class PolyRing{
    public:
    RSS<uint64_t> data_s;
    PolyRing(size_t lens):data_s(0){
        data_s.getShare(0)->resize(D*lens);
        data_s.getShare(1)->resize(D*lens);
    }
};

class MSS
{
    public:
    DeviceData<uint64_t>r_1,r_2,r;
    MSS():r_1(Degree), r_2(Degree), r(Degree){

    }
    MSS(MSS_Single<uint64_t>& mss):r_1(mss.r_1.size()*Degree),r_2(mss.r_1.size()*Degree),r(mss.r_1.size()*Degree){
        gpu::ringExpend(&mss.r_1, &r_1, sqrt(mss.r_1.size()), sqrt(mss.r_1.size()), Degree);
        gpu::ringExpend(&mss.r_2, &r_2, sqrt(mss.r_1.size()), sqrt(mss.r_1.size()), Degree);
    }
    MSS(uint32_t len):r_1(len*Degree), r_2(len*Degree), r(len*Degree){
    }
    void resize(uint32_t len){
        r_1.resize(len);
        r_2.resize(len);
        r.resize(len);
    }
    
};

template<typename T>
void reshare(DeviceData<T> &c, MSS &out) {

    auto next = nextParty(partyNum);
    auto prev = prevParty(partyNum);

    DeviceData<T> rndMask(c.size());
    rndMask += c;

    // jank equivalent to =
    out.r_1.zero();
    out.r_1 += rndMask;

    out.r_1.transmit(prev);
    out.r_2.receive(next);

    out.r_1.join();
    out.r_2.join();

}

class DotVerify{
    public:
    //TODO:random generate alpha
    
    MSS alpha;
    DeviceData<uint64_t> gamma, temp;
    DotVerify():gamma(Degree), temp(0){
        //set alpha
    }
    //input r_x r_y r_z
    void set_up( MSS *x,  MSS  *y, MSS  *z){
        temp.resize(3*x->r.size());
        if(partyNum == PARTY_A){
            
            gpu::mssOffline(&x->r, &y->r,&alpha.r, &gamma, &temp,x->r.size()/Degree, Degree);
            gamma += z->r;
            gamma.transmit(PARTY_B);
            gamma.join();
            printf("-----------------1\n");
            temp.transmit(PARTY_B);
            temp.join();
            printf("-----------------2\n");
        }else{
            if(partyNum == PARTY_B){
                gamma.receive(PARTY_A);
                gamma.join();
                printf("-----------------1\n");
                temp.receive(PARTY_A);
                temp.join();
                printf("-----------------2\n");
            }
        }
    }
    void verify( MSS *x,  MSS  *y, MSS  *z){
        if(partyNum != PARTY_A){
            gpu::mssOnline(&temp, &x->r_1, &y->r_1, &alpha.r_1, &gamma, &z->r_1, x->r.size()/Degree, Degree);
        }
        //open z
    }
};
class DotForRing{
    public:
    bool ismalicious = false;
    DotForRing(){}
    //need_gen: need generate r_z
    void online(const RSS<uint64_t> &x, const RSS<uint64_t> &y, RSS<uint64_t>& output){
        PolyRing<Degree> xx(x.size());
        PolyRing<Degree> yy(y.size());
        DeviceData<uint64_t> summed(xx.data_s.size());
        DeviceData<uint64_t> temp(Degree);
        DeviceData<uint64_t> temp2(Degree);
        summed.zero();
        summed += *yy.data_s.getShare(0);
        summed += *yy.data_s.getShare(1);
        gpu::ringDot(xx.data_s.getShare(0), &summed, &temp, x.size(), Degree);
        gpu::ringDot(xx.data_s.getShare(1), yy.data_s.getShare(0), &temp2, x.size(), Degree);
        temp+=temp2;
        reshare(temp, output);
    }
    void verify(){}
    private:
    
};



bool DotReduce(MSS* x,MSS* y, MSS* z, MSS* x_p, MSS* y_p, MSS* z_p){
        
        //set N/2 linear function f^{N/2}_m
    MSS ff0(x->r.size()/Degree/2),ff1(x->r.size()/Degree/2),ff2(x->r.size()/Degree/2);
    MSS gg0(x->r.size()/Degree/2),gg1(x->r.size()/Degree/2),gg2(x->r.size()/Degree/2);
    MSS hh0(1), hh1(1), hh2(1);
    // gpu::ringLineInterpolation(x.r, ff0.r, ff1.r, ff2.r, x.r.size(), Degree);
    // printf("start interpolation%d %d\n", x->r_1.size(), Degree);
    gpu::ringLineInterpolation(&x->r_1, &ff0.r_1, &ff1.r_1, &ff2.r_1, x->r_1.size()/Degree, Degree);
    gpu::ringLineInterpolation(&x->r_2, &ff0.r_2, &ff1.r_2, &ff2.r_2, x->r_2.size()/Degree, Degree);
    // gpu::ringLineInterpolation(y.r, gg0.r, gg1.r, gg2.r, x.r.size(), Degree);
    gpu::ringLineInterpolation(&y->r_1, &gg0.r_1, &gg1.r_1, &gg2.r_1, x->r_1.size()/Degree, Degree);
    gpu::ringLineInterpolation(&y->r_2, &gg0.r_2, &gg1.r_2, &gg2.r_2, x->r_2.size()/Degree, Degree);
    //todo::generator delta with random
    DeviceData<uint64_t> temp(Degree);
    DeviceData<uint64_t> temp2(Degree);
    gg0.r.zero();
    gg0.r += gg0.r_1;
    gg0.r += gg0.r_2;
    gpu::ringDot(&ff0.r_1, &gg0.r, &hh0.r_1, gg0.r.size()/Degree, Degree);
    gpu::ringDot(&ff0.r_2, &gg0.r_1, &temp, gg0.r_1.size()/Degree, Degree);
    hh0.r_1+=temp;
    gg2.r.zero();
    gg2.r += gg2.r_1;
    gg2.r += gg2.r_2;
    gpu::ringDot(&ff2.r_1, &gg2.r, &hh2.r_2, gg2.r.size()/Degree, Degree);
    gpu::ringDot(&ff2.r_2, &gg2.r_1, &temp, gg2.r_1.size()/Degree, Degree);
    hh2.r_1+=temp;
    reshare(hh0.r_1, hh0);
    reshare(hh2.r_1, hh2);
    
    DeviceData<uint64_t> delta1(Degree), delta2(Degree), deltaprev(Degree), deltanext(Degree);
    //reveal delta
    auto next = nextParty(partyNum);
    auto prev = prevParty(partyNum);
    delta1.transmit(prev);
    delta2.transmit(next);
    deltaprev.receive(next);
    deltanext.receive(prev);
    delta1.join();
    deltaprev.join();
    delta2.join();
    deltanext.join();
    //check deltaprev == deltanext
    delta2 += deltaprev;
    delta1 += delta2;
    
    //calculate h1 = z - h0
    hh1.r_1 += z->r_1;
    hh1.r_1 -= hh0.r_1;
    hh1.r_2 += z->r_2;
    hh1.r_2 -= hh0.r_2;
    
    // //interpolate
    gpu::ringInterpolation_1d(&delta1, &ff0.r_1,  &ff1.r_1, &gg0.r_1, &gg1.r_1, &x_p->r_1, &y_p->r_1, ff0.r_1.size()/Degree, Degree);
    gpu::ringInterpolation_1d(&delta1, &ff0.r_2,  &ff1.r_2, &gg0.r_2, &gg1.r_2, &x_p->r_2, &y_p->r_2, ff0.r_1.size()/Degree, Degree);
    gpu::ringInterpolation_2d(&delta1, &hh0.r_1,  &hh1.r_1, &hh2.r_1, &z_p->r_1, hh0.r_1.size()/Degree, Degree);
    gpu::ringInterpolation_2d(&delta1, &hh0.r_2,  &hh1.r_2, &hh2.r_2, &z_p->r_2, hh0.r_2.size()/Degree, Degree);
    return true;
}
void swap(MSS*&a, MSS*&b){
    MSS*c;
    c = a;
    a = b;
    b = c;
}
template<uint32_t R>
bool DotVerifyWithReduce(MSS* x, MSS* y, MSS* z){
    MSS *px, *py, *pz, *px2, *py2, *pz2;
    MSS xorg,yorg,zorg;
    MSS x2,y2,z2;

    x2.resize(x->r_1.size()/2);
    y2.resize(x->r_1.size()/2);

    DotReduce(x,y,z, &x2, &y2, &z2);
    px = &x2;
    py = &y2;
    pz = &z2;
    px2 = &xorg;
    py2 = &yorg;
    pz2 = &zorg;
    for(int i = 0; i < R - 1; i++){
        px2->resize(px->r_1.size()/2); 
        py2->resize(py->r_1.size()/2); 
        
        DotReduce(px,py,pz, px2, py2, pz2);
        swap(px, px2);
        swap(py, py2);
        swap(pz, pz2);
    
    }
    
    DotVerify dotv;

    dotv.set_up(px, py, pz);

    dotv.verify(px, py, pz);

    return true;
}
// class Multiple_verify{

// }