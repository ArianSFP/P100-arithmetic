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
    for(int rows:{32,64,160}) {
        constexpr int groups=17;std::vector<bool> seen(rows*groups*4);
        std::vector<uint32_t> packed(rows*groups*4);
        for(int row=0;row<rows;++row)for(int g=0;g<groups;++g)for(int p=0;p<4;++p) {
            size_t x=((size_t(row/32)*groups+g)*32+row%32)*4+p;
            require(x<seen.size() && !seen[x],"vector word mapping");seen[x]=true;
            packed[x]=uint32_t((row*groups+g)*4+p);
        }
        for(int tile=0;tile<rows/32;++tile)for(int lane=0;lane<32;++lane)for(int g=0;g<groups;++g) {
            unsigned x=((tile*groups+g)*32+lane)*4;require(x%4==0,"128-bit alignment");
            for(int p=0;p<4;++p)require(packed[x+p]==unsigned(((tile*32+lane)*groups+g)*4+p),"vector load order");
        }
    }
    require(uint64_t(17408)*544*4*4<(uint64_t(1)<<32),"bounded 32-bit byte addresses");
    std::printf("CPU_PASS finite_half=%d exact_W4_products=%d random_lossless_blocks=65536 layouts=2 address_bijections=3 vector_alignment=1 index_bound=1\n",finite,finite*16);
}
