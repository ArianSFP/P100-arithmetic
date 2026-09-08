#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

struct Vec {
    const char * name;
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
    std::uint32_t seed;
};

constexpr Vec kVectors[] = {
    {"lane_distinguish", 0x40003c00u, 0x45004200u, 0x49804700u, 0x5a405640u},
    {"signed_lane_distinguish", 0x4000bc00u, 0xc5004200u, 0xc9804700u, 0x5dc0d640u},
    {"independent_seed", 0x3c003c00u, 0x40004000u, 0x44004400u, 0x5c005800u},
};

[[noreturn]] void fail(const char * what, CUresult result) {
    const char * name = nullptr;
    const char * text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::cerr << "status=error operation=" << what
              << " code=" << static_cast<unsigned int>(result)
              << " name=" << (name ? name : "unknown")
              << " text=" << (text ? text : "unknown") << '\n';
    std::exit(3);
}

void check(const char * what, CUresult result) {
    if (result != CUDA_SUCCESS) {
        fail(what, result);
    }
}

std::string hex32(std::uint32_t value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
    return stream.str();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " CUBIN\n";
        return 2;
    }
    check("cuInit", cuInit(0));
    CUdevice device;
    check("cuDeviceGet", cuDeviceGet(&device, 0));
    int major = 0;
    int minor = 0;
    check("cuDeviceComputeCapability", cuDeviceComputeCapability(&major, &minor, device));
    std::cout << "status=ready device_compute=" << major << "." << minor
              << " cubin=" << argv[1] << '\n';

    CUcontext context;
    check("cuCtxCreate", cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
    CUmodule module;
    check("cuModuleLoad", cuModuleLoad(&module, argv[1]));
    CUfunction function;
    check("cuModuleGetFunction", cuModuleGetFunction(&function, module, "merge_probe"));

    constexpr unsigned int n = sizeof(kVectors) / sizeof(kVectors[0]);
    std::uint32_t a[n], b[n], c[n], seed[n], out[n] = {};
    for (unsigned int i = 0; i < n; ++i) {
        a[i] = kVectors[i].a;
        b[i] = kVectors[i].b;
        c[i] = kVectors[i].c;
        seed[i] = kVectors[i].seed;
    }
    CUdeviceptr da, db, dc, dseed, dout, dside;
    check("cuMemAlloc(a)", cuMemAlloc(&da, sizeof(a)));
    check("cuMemAlloc(b)", cuMemAlloc(&db, sizeof(b)));
    check("cuMemAlloc(c)", cuMemAlloc(&dc, sizeof(c)));
    check("cuMemAlloc(seed)", cuMemAlloc(&dseed, sizeof(seed)));
    check("cuMemAlloc(out)", cuMemAlloc(&dout, sizeof(out)));
    check("cuMemAlloc(side)", cuMemAlloc(&dside, sizeof(out)));
    check("cuMemcpyHtoD(a)", cuMemcpyHtoD(da, a, sizeof(a)));
    check("cuMemcpyHtoD(b)", cuMemcpyHtoD(db, b, sizeof(b)));
    check("cuMemcpyHtoD(c)", cuMemcpyHtoD(dc, c, sizeof(c)));
    check("cuMemcpyHtoD(seed)", cuMemcpyHtoD(dseed, seed, sizeof(seed)));

    unsigned int device_n = n;
    void * args[] = {&da, &db, &dc, &dseed, &dout, &dside, &device_n};
    check("cuLaunchKernel", cuLaunchKernel(function, 1, 1, 1, 32, 1, 1,
                                             0, nullptr, args, nullptr));
    check("cuCtxSynchronize", cuCtxSynchronize());
    check("cuMemcpyDtoH(out)", cuMemcpyDtoH(out, dside, sizeof(out)));
    std::cout << "status=ok vectors=" << n << '\n';
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << "vector=" << kVectors[i].name
                  << " a=" << hex32(a[i])
                  << " b=" << hex32(b[i])
                  << " c=" << hex32(c[i])
                  << " seed=" << hex32(seed[i])
                  << " observed_side=" << hex32(out[i]) << '\n';
    }
    cuMemFree(dside);
    cuMemFree(dout);
    cuMemFree(dseed);
    cuMemFree(dc);
    cuMemFree(db);
    cuMemFree(da);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
}
