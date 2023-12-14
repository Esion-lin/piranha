#pragma once
#include <cstddef>
#include "../gpu/DeviceData.h"
#include "../gpu/ring.cuh"
#include "../globals.h"

#include "RSS.h"
template<class T>
class SwiftShareType{
    public:
    DeviceData<T> alpha1,alpha2,gamma,beta;
    SwiftShareType():alpha1(0),alpha2(0),gamma(0),beta(0){

    }
    SwiftShareType(uint32_t len):alpha1(len),alpha2(len),gamma(len),beta(len){
        
    }
};
template<class T>
class Swift{
    public:
    RSS<T> d,e,f;
    DeviceData<T> alpha_z_1,alpha_z_2,gamma_z, chi_1,chi_2,psi;

    Swift():d(0),e(0),f(0){

    }
    void set_up(SwiftShareType<T>*x, SwiftShareType<T>*y, SwiftShareType<T>*z){
        //TODO:random
        
        uint32_t len = x->alpha1.size();
        alpha_z_1.resize(len);
        alpha_z_2.resize(len);
        gamma_z.resize(len);
        chi_1.resize(len);
        chi_2.resize(len);
        psi.resize(len);
        d.resize(len);
        e.resize(len);
        f.resize(len);
        if(partyNum == RSS<T>::PARTY_A){
            *d.getShare(0) += x->alpha1;
            *d.getShare(1) += x->gamma;

            *e.getShare(0) += y->alpha1;
            *e.getShare(1) += y->gamma;

        }else if(partyNum == RSS<T>::PARTY_B){
            *d.getShare(0) += x->alpha2;
            *d.getShare(1) += x->gamma;

            *e.getShare(0) += y->alpha2;
            *e.getShare(1) += y->gamma;
        }else{
            *d.getShare(0) += x->alpha2;
            *d.getShare(1) += x->alpha1;

            *e.getShare(0) += y->alpha2;
            *e.getShare(1) += y->alpha1;
        }
        f.zero();
        f+=d;
        f*=e;
        if(partyNum == RSS<T>::PARTY_A){
            chi_1 += *f.getShare(0);
            chi_2 += *f.getShare(1);
        }else if(partyNum == RSS<T>::PARTY_B){
            chi_1 += *f.getShare(1);
            psi += *f.getShare(0);
            DeviceData<T> temp(len);
            temp += x->gamma;
            temp *= y->gamma;
            psi -= temp;
        }else{
            chi_2 += *f.getShare(0);
            psi += *f.getShare(1);
            DeviceData<T> temp(len);
            temp += x->gamma;
            temp *= y->gamma;
            psi -= temp;
        }
    }
    void online(SwiftShareType<T>*x, SwiftShareType<T>*y, SwiftShareType<T>*z){
        uint32_t len = x->alpha1.size();
        if(partyNum == RSS<T>::PARTY_A){
            DeviceData<T> temp(len), temp2(len), temp3(len);
            //for P1

            temp -= x->gamma;
            temp *= y->alpha1;
            temp2 -= y->gamma;
            temp2 *= x->alpha1;   
            temp -= temp2;
            temp += alpha_z_1;
            temp += chi_1;
            temp.transmit(RSS<T>::PARTY_C);
            temp.join();
            //for P2
            temp3 -= x->gamma;
            temp3 *= y->alpha2;
            temp2.zero();
            temp2 -= y->gamma;
            temp2 *= x->alpha2;
            temp3 -= temp2;
            temp3 += alpha_z_2;
            temp3 += chi_2;
            temp3.transmit(RSS<T>::PARTY_B);
            temp3.join();
            z->gamma.receive(RSS<T>::PARTY_B);
            z->gamma.join();
        }else if(partyNum == RSS<T>::PARTY_B){
            //TODO: send hash to PARTY_C
            DeviceData<T> temp(len), temp2(len);
            temp -= x->gamma;
            temp -= x->beta;
            temp2 -= y->gamma;
            temp2 -= y->beta;
            temp2 *= x->alpha1;
            temp -= temp2;
            temp += alpha_z_1;
            temp += chi_1;
            temp2.receive(RSS<T>::PARTY_A);
            temp2.join();
            temp += temp2;
            z->beta.zero();
            z->beta += x->beta;
            z->beta *= y->beta;
            z->beta += temp;
            z->beta += psi;
            z->gamma += gamma_z;
            z->alpha1 += alpha_z_1;
            gamma_z += z->beta;
            gamma_z.transmit(RSS<T>::PARTY_A);
            gamma_z.join();
        }else{
            //TODO: send hash to PARTY_B
            DeviceData<T> temp(len), temp2(len);
            temp -= x->gamma;
            temp -= x->beta;
            temp2 -= y->gamma;
            temp2 -= y->beta;
            temp2 *= x->alpha2;
            temp -= temp2;
            temp += alpha_z_2;
            temp += chi_2;
            temp2.receive(RSS<T>::PARTY_A);
            temp2.join();
            temp += temp2;
            z->beta.zero();
            z->beta += x->beta;
            z->beta *= y->beta;
            z->beta += temp;
            z->beta += psi;
            z->gamma += gamma_z;
            z->alpha2 += alpha_z_2;
            gamma_z += z->beta;
            
        }
        
        
    }
};


