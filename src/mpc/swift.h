#pragma once
#include <cstddef>
#include "../gpu/DeviceData.h"
#include "../gpu/ring.cuh"
#include "../globals.h"

#include "RSS.h"

template<class T>
void Dummy_Swift(DeviceData<T> *alpha_x, DeviceData<T> *alpha_y, DeviceData<T> *beta_x, DeviceData<T> *beta_y, DeviceData<T> * gamma){
    if(partyNum == RSS<T>::PARTY_A){

    }else if(partyNum == RSS<T>::PARTY_B){

    }else{

    }
}