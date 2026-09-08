#include "common.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

uint32_t rng32(uint32_t &s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
void require(bool ok,const char *message) {
    if(!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(2); }
}
template<int W,int F> void check_group(const std::array<int,32> &weights,const uint32_t (&a)[8],int ref,int asum) {
    uint32_t planes[W]={};
    for(int j=0;j<32;++j) for(int p=0;p<W;++p) {
        unsigned b=(unsigned(weights[j])>>p)&1;
        planes[p]|=(F==2 ? 1-b : b)<<bit_position(F,j);
    }
    for(int j=0;j<32;++j) {
        unsigned code=0;
        for(int p=0;p<W;++p) {
            unsigned b=(planes[p]>>bit_position(F,j))&1;
            code|=(F==2 ? 1-b : b)<<p;
        }
        require(signed_code(code,W)==weights[j],"lossless pack/unpack");
    }
    int sad_sums[W]={},lut_sums[W]={},integer_sum=0;
    for(int q=0;q<8;++q) {
        integer_sum=vmad4(decode4<W,F>(planes,q),a[q],integer_sum);
        int table[32];
        for(int lane=0;lane<32;++lane) {
            table[lane]=table_entry(a[q],lane&15);
            require(table[lane]==table_entry_cast(a[q],lane&15),"int8-cast table construction");
        }
        for(int p=0;p<W;++p) {
            unsigned idx=index4<F>(planes[p],q),expected_idx=0;
            for(int j=0;j<4;++j) {
                unsigned bit=(unsigned(weights[4*q+j])>>p)&1;
                expected_idx|=(F==2 ? 1-bit : bit)<<j;
            }
            require(idx==expected_idx,"quartet-index extraction");
            lut_sums[p]+=table[idx];
            if constexpr(F!=3) {
                uint32_t expected_mask=0;
                for(int j=0;j<4;++j) expected_mask|=255u*((idx>>j)&1)<<(8*j);
                uint32_t actual=sign_bytes(planes[p]<<(F==0 ? 7-q : q));
                require(actual==expected_mask,"PRMT sign expansion / cross-byte shift");
                sad_sums[p]=sad4(a[q]^0x80808080u,actual,sad_sums[p]);
            }
        }
    }
    require((F==2 ? -asum-integer_sum : integer_sum)==ref,"ordinary arithmetic control");
    int lut=combine(lut_sums);
    require((F==2 ? -asum-lut : lut)==ref,"register-LUT equation");
    if constexpr(F!=3) require(finish_sad<W,F>(planes,sad_sums,asum)==ref,"endpoint-SAD equation");
}
template<int W> void correctness() {
    uint32_t rng=0x60ba9800+W;
    constexpr int cases=65536;
    for(int trial=0;trial<cases;++trial) {
        std::array<int,32> w;
        uint32_t a[8]={}; int ref=0,asum=0;
        for(int j=0;j<32;++j) {
            int av=trial<32 ? (trial&1 ? -128 : 127) : int(rng32(rng)&255)-128;
            w[j]=signed_code((trial<32 ? unsigned(trial>>1) : rng32(rng))&((1<<W)-1),W);
            a[j/4]|=(unsigned(av)&255)<<(8*(j%4));
            ref+=av*w[j]; asum+=av;
        }
        check_group<W,0>(w,a,ref,asum); check_group<W,1>(w,a,ref,asum);
        check_group<W,2>(w,a,ref,asum); check_group<W,3>(w,a,ref,asum);
    }
    std::printf("PASS W=%d groups=%d formats=4 group_size=32 ordinary_LUT_SAD=exact seed=0x%08x\n",W,cases,0x60ba9800+W);
}
void table_reuse_and_tails() {
    uint32_t rng=0x60ba9832;
    uint64_t tested=0;
    for(int bits : {2,4}) for(int reuse : {1,2,4}) for(int rows : {1,17,31,32,33,65,129}) {
        int blocks=(rows+128*reuse-1)/(128*reuse);
        std::vector<int> writes(rows,0);
        for(int block=0;block<blocks;++block) for(int warp=0;warp<4;++warp) {
            uint32_t a=rng32(rng);
            int table[32];
            for(int lane=0;lane<32;++lane) table[lane]=table_entry(a,lane&15);
            for(int r=0;r<reuse;++r) for(int lane=0;lane<32;++lane) {
                int row=(block*4+warp)*32*reuse+32*r+lane;
                int actual=0,expected=0;
                unsigned codes[4]={};
                for(int p=0;p<bits;++p) {
                    unsigned idx=rng32(rng)&15;
                    actual+=coefficient(p,bits)*table[idx];
                    for(int j=0;j<4;++j) codes[j]|=((idx>>j)&1)<<p;
                }
                for(int j=0;j<4;++j) expected+=signed_code(codes[j],bits)*signed_code((a>>(8*j))&255,8);
                require(actual==expected,"warp lookup / row reuse");
                if(row<rows) ++writes[row];
                ++tested;
            }
        }
        for(int n:writes) require(n==1,"tail row ownership exactly once");
    }
    std::printf("PASS warp_reuse_and_tail_lane_cases=%llu rows_per_warp=32,64,128\n",(unsigned long long)tested);
}
void address_and_basis_tests() {
    for(int bit=0;bit<32;++bit) for(int q=0;q<8;++q) {
        uint32_t expected=0;
        for(int j=0;j<4;++j) if(bit==8*j+7-q) expected|=255u<<(8*j);
        require(sign_bytes((1u<<bit)<<q)==expected,"all basis bits / no shift contamination");
    }
    for(int bits : {2,4}) for(int rows : {1,33,128}) for(int groups : {1,3,160}) for(bool old : {false,true}) {
        int padded=(rows+31)/32*32;
        std::vector<uint32_t> weights(padded*groups*bits,0),scales(padded*groups,0);
        for(int row=0;row<rows;++row) for(int g=0;g<groups;++g) {
            auto sa=scale_address(row,g,groups,old);
            require(sa<scales.size() && !scales[sa],"scale address bounds/uniqueness");
            scales[sa]=1+row*groups+g;
            for(int p=0;p<bits;++p) {
                auto wa=weight_address(row,g,p,groups,bits,old);
                require(wa<weights.size() && !weights[wa],"weight address bounds/uniqueness");
                weights[wa]=1+(row*groups+g)*bits+p;
            }
        }
        for(int row=0;row<rows;++row) for(int g=0;g<groups;++g) {
            require(scales[scale_address(row,g,groups,old)]==unsigned(1+row*groups+g),"scale address round trip");
            for(int p=0;p<bits;++p) require(weights[weight_address(row,g,p,groups,bits,old)]==unsigned(1+(row*groups+g)*bits+p),"weight address round trip");
        }
    }
    std::puts("PASS all 256 bit/shift basis cases and shared host/device weight+scale address functions including tails");
}
void floating_numerics() {
    uint32_t rng=0x60ba98f0;
    for(int bits : {2,4}) {
        int changed=0;
        double worst_lut=0,worst_direct=0;
        for(int trial=0;trial<32768;++trial) {
            float table[16]={},a[4]; int weights[4];
            double exact=0; float direct=0;
            for(int j=0;j<4;++j) {
                a[j]=std::ldexp(float(int(rng32(rng)&255)-128),int(rng32(rng)%20)-10);
                weights[j]=signed_code(rng32(rng)&((1<<bits)-1),bits);
                exact+=double(a[j])*weights[j];
                direct=std::fma(float(weights[j]),a[j],direct);
            }
            for(int e=0;e<16;++e) for(int j=0;j<4;++j) if(e&(1<<j)) table[e]+=a[j];
            float lut=0;
            for(int p=0;p<bits;++p) {
                unsigned idx=0;
                for(int j=0;j<4;++j) idx|=((unsigned(weights[j])>>p)&1)<<j;
                lut=std::fma(float(coefficient(p,bits)),table[idx],lut);
            }
            changed+=lut!=direct;
            if(changed==1 && lut!=direct) {
                std::printf("FP32_WITNESS W=%d weights=%d,%d,%d,%d a=%a,%a,%a,%a lut=%a direct=%a exact=%a\n",
                    bits,weights[0],weights[1],weights[2],weights[3],double(a[0]),double(a[1]),double(a[2]),double(a[3]),double(lut),double(direct),exact);
            }
            worst_lut=std::fmax(worst_lut,std::abs(double(lut)-exact));
            worst_direct=std::fmax(worst_direct,std::abs(double(direct)-exact));
        }
        std::printf("NUMERICS W=%d finite_FP32_quartets=32768 regrouped_differs=%d max_abs_error_lut=%g direct=%g NOT_a_quality_gate\n",bits,changed,worst_lut,worst_direct);
    }
}
int main() {
    correctness<2>(); correctness<4>(); table_reuse_and_tails(); address_and_basis_tests(); floating_numerics();
    return 0;
}
