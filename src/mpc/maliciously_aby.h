#pragma once
#include <cstddef>
#include "../gpu/DeviceData.h"
#include "../gpu/ring.cuh"
#include "../globals.h"

#include "RSS.h"
template<class T>
bool RSS_verify(RSS<T> *x, RSS<T> *y, RSS<T> *z){
    RSS<T> a (x->size()); 
    RSS<T> b (x->size());
    RSS<T> c (x->size());
    c.zero();
    c+=a;
    c*=b;
    //perform verify
    *x -= a;
    *y -= b;
    DeviceData<T> rho(a.size()),sigma(a.size()),delta(a.size());
    reconstruct(*x, rho);
    reconstruct(*y, sigma);
    a *= sigma;
    b *= rho;
    sigma *= rho;
    *z -= c;
    *z -= a;
    *z -= b;
    *z -= sigma;
    reconstruct(*z, delta);
    return true;
}