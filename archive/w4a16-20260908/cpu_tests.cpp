#include "common.hpp"
#include <cstdio>
#include <vector>

int main() {
    int finite=0;
    for(unsigned bits=0;bits<65536;++bits) {
        unsigned e=(bits>>10)&31,m=bits&1023;
        if(e==31)continue;
        double independent=e ? std::ldexp(1.0+double(m)/1024,int(e)-15) : std::ldexp(double(m),-24);
        if(bits&32768)independent=-independent;
        require(double(half_float(bits))==independent,"exact half widening");
        require(std::ldexp(double(half_fixed(bits)),-24)==independent,"fixed-point oracle");
        for(int q=-8;q<8;++q)require(double(float(q)*half_float(bits))==q*independent,"W4 times A16 exact FP32 product");
        ++finite;
    }
    uint32_t rng=0x41660001;
    for(int i=0;i<65536;++i) {
        Q4Block b{};for(auto &q:b.qs)q=random32(rng);
        for(int f:{0,1}) {uint32_t w[4];pack_block(b,f,w);for(int j=0;j<32;++j)require(unpack_block(w,f,j)==native_code(b,j),"lossless repack");}
    }
    for(int old:{0,1}) {
        constexpr int rows=160,groups=17;
        std::vector<bool> seen(rows*groups*4),scales(rows*groups);
        for(int r=0;r<rows;++r)for(int g=0;g<groups;++g) {
            size_t s=scale_address(r,g,groups,old);require(s<scales.size() && !scales[s],"scale mapping");scales[s]=true;
            for(int p=0;p<4;++p) {size_t x=word_address(r,g,p,groups,old);require(x<seen.size() && !seen[x],"weight mapping");seen[x]=true;}
        }
    }
    std::printf("CPU_PASS finite_half=%d exact_W4_products=%d random_lossless_blocks=65536 layouts=2 address_bijections=2\n",finite,finite*16);
}
