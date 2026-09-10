#include "packed_int4.cuh"
#include <array>
#include <iostream>
#include <random>
int main() {
    std::mt19937 rng(20260909u);
    std::uniform_int_distribution<int> code(-8,7);
    constexpr int N=100000;
    for(int trial=0;trial<N;++trial) {
        std::array<int16_t,32> p{};
        std::array<int8_t,32> a{};
        int r0=0,r1=0;
        for(int k=0;k<32;++k) {
            int x=code(rng),y=code(rng),z=code(rng);
            if(trial<8) { x=(trial&1)?-8:7; y=(trial&2)?-8:7; z=(trial&4)?-8:7; }
            p[k]=packed_int4::pack_shared32(x,y);a[k]=static_cast<int8_t>(z);
            r0+=x*z;r1+=y*z;
        }
        auto got=packed_int4::shared_dot32(p.data(),a.data());
        if(got.first!=r0 || got.second!=r1) {std::cerr<<"shared failure\n";return 1;}
        std::array<int16_t,8> w{},q{},qr{};
        r0=r1=0;int rd=0;uint32_t c=0;
        for(int k=0;k<8;++k) {
            int x=code(rng),y=code(rng),z=code(rng),t=code(rng);
            if(trial<16) {x=(trial&1)?-8:7;y=(trial&2)?-8:7;z=(trial&4)?-8:7;t=(trial&8)?-8:7;}
            w[k]=packed_int4::pack_independent8(x,y);
            q[k]=packed_int4::pack_independent8(z,t);
            qr[k]=packed_int4::pack_independent8(t,z);
            c=packed_int4::mad_wide_s16(w[k],qr[k],c);
            r0+=x*z;r1+=y*t;rd+=x*z+y*t;
        }
        got=packed_int4::independent_dots8(w.data(),q.data());
        if(got.first!=r0 || got.second!=r1) {std::cerr<<"independent failure\n";return 1;}
        if(packed_int4::unpack_kpair16(c)!=rd) {std::cerr<<"K-pair failure\n";return 1;}
    }
    std::cout<<"100000 cases each: shared G32, independent G8, K-pair G16. All passed.\n";
}
