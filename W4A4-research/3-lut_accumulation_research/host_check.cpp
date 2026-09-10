// Independent scalar C++ validation. CPU only; no CUDA or timing claims.
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Planes = std::array<std::uint32_t, 4>;

std::array<int,2> finish2(const Planes& p, const std::array<int,2>& bias) {
    const std::uint32_t t=p[0]+(p[1]<<1)+(p[2]<<2)+(p[3]<<3);
    return {int(t&65535u)-bias[0], int(t>>16)-bias[1]};
}
std::array<int,4> finish4(const Planes& p, const std::array<int,4>& bias) {
    std::uint32_t even=0, odd=0;
    for (unsigned b=0;b<4;++b) {
        even += (p[b]&0x00ff00ffu)<<b;
        odd  += ((p[b]>>8)&0x00ff00ffu)<<b;
    }
    return {int(even&65535u)-bias[0],int(odd&65535u)-bias[1],
            int(even>>16)-bias[2],int(odd>>16)-bias[3]};
}

template<int R, int G, int BITS>
std::array<int,R> dot_lut(const std::array<std::array<int,G>,R>& values,
                          const std::array<int,G>& selector) {
    static_assert(R*BITS==32);
    static_assert(G%8==0);
    static_assert(BITS==16 || (BITS==8 && G<=32));
    if constexpr (BITS==8) {
        for (const auto& row:values) {
            int bound=0; for (int v:row) bound += v<0 ? -v:v;
            if (bound>255) throw std::runtime_error("unsafe byte accumulation");
        }
    }
    Planes accum{};
    std::array<int,R> correction{};
    for (int r=0;r<R;++r)
        for (int k=0;k<G;++k)
            correction[r] += 15*(values[r][k]<0 ? -values[r][k]:0)+8*values[r][k];
    for (int k=0;k<G;k+=8) {
        std::array<std::uint32_t,256> table{};
        for (unsigned key=0;key<256;++key) {
            std::uint32_t word=0;
            for (int r=0;r<R;++r) {
                int total=0;
                for (int j=0;j<8;++j) {
                    const int v=values[r][k+j];
                    total += ((key>>j)&1u) ? v : 0;
                    total += v<0 ? -v : 0;
                }
                word |= std::uint32_t(total)<<(BITS*r);
            }
            table[key]=word;
        }
        for (unsigned b=0;b<4;++b) {
            unsigned key=0;
            for (unsigned j=0;j<8;++j)
                key |= ((unsigned(selector[k+j]+8)>>b)&1u)<<j;
            accum[b] += table[key];
        }
    }
    if constexpr (R==2) return finish2(accum, correction);
    else return finish4(accum, correction);
}

int main() {
    std::mt19937 rng(20260909);
    std::uniform_int_distribution<int> code(-8,7);
    constexpr int N=10000;
    for (int test=0;test<N;++test) {
        std::array<std::array<int,32>,2> v2{};
        std::array<std::array<int,16>,4> v4{};
        std::array<std::array<int,32>,4> v4g32{};
        std::array<int,32> s32{};
        std::array<int,16> s16{};
        for (auto& v:s32) v=test<4 ? (test&1 ? 7:-8) : code(rng);
        for (auto& v:s16) v=test<4 ? (test&1 ? 7:-8) : code(rng);
        for (auto& row:v2) for (auto& v:row) v=test<4 ? (test&2 ? 7:-8) : code(rng);
        for (auto& row:v4) for (auto& v:row) v=test<4 ? (test&2 ? 7:-8) : code(rng);
        for (int r=0;r<4;++r) for (int k=0;k<32;++k)
            v4g32[r][k] = test<4 ? (test&2 ? 7:-8) : code(rng);
        if (test==4 || test==5) {
            for (auto& row:v4g32) row.fill(-8);
            v4g32[0][0]=-7; v4g32[1].fill(0); v4g32[3].fill(7);
            s32.fill(test==4 ? -8:7);
        }
        auto norm=v4g32;
        std::array<int,4> factor{1,1,1,1};
        for (int r=0;r<4;++r) {
            bool special=true; for (int v:norm[r]) special &= v==-8;
            if (special) {factor[r]=2; for (int& v:norm[r]) v/=2;}
        }
        const auto g32=dot_lut<4,32,8>(norm,s32);
        for (int r=0;r<4;++r) {
            int ref=0; for (int k=0;k<32;++k) ref+=v4g32[r][k]*s32[k];
            if (ref!=g32[r]*factor[r]) throw std::runtime_error("normalized G32 mismatch");
        }
        const auto got2=dot_lut<2,32,16>(v2,s32);
        const auto got4=dot_lut<4,16,8>(v4,s16);
        for (int r=0;r<2;++r) {
            int ref=0; for (int k=0;k<32;++k) ref+=v2[r][k]*s32[k];
            if (ref!=got2[r]) throw std::runtime_error("two-lane mismatch");
        }
        for (int r=0;r<4;++r) {
            int ref=0; for (int k=0;k<16;++k) ref+=v4[r][k]*s16[k];
            if (ref!=got4[r]) throw std::runtime_error("four-lane mismatch");
        }
    }
    std::cout << "PASS: " << N << " two-lane G32 blocks and " << N
              << " four-lane G16 blocks plus " << N
              << " normalized four-lane G32 blocks; 100000 exact output dots; zero mismatches\n";
}
