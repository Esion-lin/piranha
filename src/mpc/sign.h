#pragma once
#include <cstddef>
#include <math.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"
#include "../gpu/sign.cuh"
#include "../gpu/fake_random.cuh"
void test_bit_operator(){
    std::vector<uint8_t> res1(8), res2(8 * 64);
    DeviceData<uint64_t> datas({1,2,3,12883,123,834583, (((uint64_t)1)<<63) + 218912, 1231});
    DeviceData<uint64_t> datas2({1,2,3,12883,123,834583, 12312, (((uint64_t)1)<<63) + 1231});
    DeviceData<uint8_t> msb(8);
    DeviceData<uint8_t> bits(8 * 64);
    gpu::bit_operator<uint64_t>(&datas, &datas2, &bits, 8, 64);
    gpu::msb<uint64_t>(&datas, &datas2, &msb, 2, 4);
    thrust::copy(msb.begin(), msb.end(), res1.begin());
    thrust::copy(bits.begin(), bits.end(), res2.begin());
    printf("\n");
    for(int i = 0; i < 8; i++){
        printf("%d ",res1[i]);
    }
    printf("\n");
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 64; j++)
            printf("%d ",res2[i*64 + j]);
        printf("\n");
    }
    printf("\n");

}
void test_random(){
    DeviceData<uint64_t> res2(8 * 64);
    std::vector<uint64_t> datas(8 * 64);
    gpu::generateRandomNumbers(&res2, 16, 32);
    thrust::copy(res2.begin(), res2.end(), datas.begin());
    for(int i = 0; i < 8*64; i++){
        printf("%d ",datas[i]);
    }
}
int split(uint32_t len){
    //assert len = 2^a
    int a = log(len);
    return 1<<(int)a/2;
}
template<class T>
class Sign{
    public:
    DeviceData<uint8_t> msb_r;
    DeviceData<uint8_t> delta;
    DeviceData<uint8_t> bits, bits_p;
    DeviceData<T> gamma,z_p;

    
    Sign(){}
    void set_up(MSS_Single<T>& x, MSS_Single<T>& z){
        if(partyNum == PARTY_A){


            msb_r.resize(x.r_1.size());
            bits.resize(8*sizeof(T) * x.r_1.size());
            //r bits
            gpu::bit_operator<uint64_t>(&x.r_1, &x.r_2, &bits, x.r_1.size(), 8*sizeof(T));
            gpu::msb<uint64_t>(&x.r_1, &x.r_2, &msb_r, split(x.r_1.size()), x.r_1.size()/split(x.r_1.size()));
            //TODO: random->sh1
            DeviceData<T> sh1(8*sizeof(T) * x.r_1.size());
            bits -= sh1;
            bits.transmit(PARTY_C);
            bits.join();
            z_p.resize(x.r_1.size());
            //random r_z
            
        }else if(partyNum == PARTY_B){
            bits.resize(8*sizeof(T) * x.r_1.size());
            bits_p.resize(8*sizeof(T) * x.r_1.size());
            delta.resize(x.r_1.size());
            z_p.resize(x.r_1.size());
            gamma.resize(x.r_1.size());
            DeviceData<T> gamma_p(x.r_1.size());
            calculate_gamma(&z_p, &delta, &z.r_1, gamma_p);
            gamma_p.transmit(PARTY_C);
            gamma_p.join();
            gamma.reveive(PARTY_C);
            gamma.join();
            gamma += gamma_p;
        }else{
            //TODO: randomize
            bits.resize(8*sizeof(T) * x.r_1.size());
            bits_p.resize(8*sizeof(T) * x.r_1.size());
            delta.resize(x.r_1.size());
            gamma.resize(x.r_1.size());
            z_p.resize(x.r_1.size());
            bits.receive(PARTY_A);
            bits.join();
            DeviceData<T> gamma_p(x.r_1.size());
            calculate_gamma(&z_p, &delta, &z.r_2, gamma_p);
            gamma_p.transmit(PARTY_B);
            gamma_p.join();
            gamma.reveive(PARTY_B);
            gamma.join();
            gamma += gamma_p;

        }
    }
    void online(MSS_Single<T>& x, MSS_Single<T>& z){
        if(partyNum != PARTY_A){
            DeviceData<uint8_t> m_sigmas(8*sizeof(T)*x.r_1.size()), m_j(8*sizeof(T)*x.r_1.size()), w_j(8*sizeof(T)*x.r_1.size()), w_j_p(8*sizeof(T)*x.r_1.size()), u_j(8*sizeof(T)*x.r_1.size()), v_j(8*sizeof(T)*x.r_1.size()), temp(8*sizeof(T)*x.r_1.size());
            //random w_j,w'_j
            gpu::generateRandomNumbers(&w_j, 8*split(x.r_1.size()), sizeof(T)*x.r_1.size()/split(x.r_1.size()));
            gpu::generateRandomNumbers(&w_j_p, 8*split(x.r_1.size()), sizeof(T)*x.r_1.size()/split(x.r_1.size()));
            //chop m_x
            gpu::chop(&x.r_1, len1, len2);
            //
        }
    }
};