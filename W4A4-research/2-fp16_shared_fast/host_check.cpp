#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

#ifndef __FLT16_MANT_DIG__
#error "This check needs compiler support for IEEE binary16 _Float16."
#endif
static_assert(__FLT16_MANT_DIG__ == 11);
using H = _Float16;
static H fma16(double x, double a, double c) {
    // The dyadic values used here have exact product+sum in double.
    return static_cast<H>(x*a+c);
}
static int floor128(int v) { return v >= 0 ? v/128 : -((-v+127)/128); }
int main() {
    std::size_t tested=0;
    for (int x=-8;x<=7;++x) for (int y=-8;y<=7;++y) for (int a=-8;a<=7;++a) {
        H p=static_cast<H>(y+x/128.0);
        H t=fma16(p,a,1536.0);
        H high=static_cast<H>(static_cast<double>(t)-1536.0);
        H low=fma16(p,a,-static_cast<double>(high));
        assert(static_cast<double>(high)==y*a);
        assert(static_cast<double>(low)*128==x*a);
        int r=static_cast<int>(fma16(x+128*y,a,0));
        int h=floor128(r+63);
        int residue=(x*a)%8; if (residue<0) residue+=8;
        int d=((residue-r+4)%8+8)%8-4;
        assert(h==y*a);
        assert(r-128*h+d==x*a);
        ++tested;
    }
    std::mt19937 gen(20260909);
    std::uniform_int_distribution<int> dist(-8,7);
    constexpr int blocks=100000;
    for (int b=0;b<blocks;++b) {
        H lo=0,hi=0; int il=0,ih=0;
        for (int k=0;k<32;++k) {
            int x=dist(gen),y=dist(gen),a=dist(gen);
            H p=static_cast<H>(y+x/128.0);
            H t=fma16(p,a,1536.0);
            H h=static_cast<H>(static_cast<double>(t)-1536.0);
            H l=fma16(p,a,-static_cast<double>(h));
            lo=static_cast<H>(static_cast<double>(lo)+l);
            hi=static_cast<H>(static_cast<double>(hi)+h);
            il+=x*a; ih+=y*a;
        }
        assert(static_cast<double>(lo)*128==il);
        assert(static_cast<double>(hi)==ih);
    }
    std::cout << "Exhaustive signed INT4 triples: " << tested << "; mismatches: 0\n";
    std::cout << "Independent G32 random blocks: " << blocks << "; mismatches: 0\n";
}
