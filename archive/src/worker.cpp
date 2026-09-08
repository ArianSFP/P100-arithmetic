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
};

constexpr Vec kVectors[] = {
    {"lane_distinguish", 0x40003c00u, 0x45004200u, 0x49804700u},
    {"signed_lane_distinguish", 0x4000bc00u, 0xc5004200u, 0xc9804700u},
    {"a_f32_scalar", 0x3fc00000u, 0x45004200u, 0x49804700u},
    {"b_f32_scalar", 0x40003c00u, 0x3fc00000u, 0x49804700u},
    {"c_f32_scalar", 0x40003c00u, 0x45004200u, 0x3fc00000u},
    // a = 1 + 2^-10, b = 256 * (1 - 2^-10), c = -256.
    {"fused_rounding", 0x3c013c01u, 0x5bfe5bfeu, 0xdc00dc00u},
    {"zeros", 0x00000000u, 0x00000000u, 0x00000000u},
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
    check("cuDeviceComputeCapability",
          cuDeviceComputeCapability(&major, &minor, device));
    std::cout << "status=ready device_compute=" << major << "." << minor
              << " cubin=" << argv[1] << '\n';

    CUcontext context;
    check("cuCtxCreate", cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));

    CUmodule module;
    check("cuModuleLoad", cuModuleLoad(&module, argv[1]));

    CUfunction function;
    check("cuModuleGetFunction", cuModuleGetFunction(&function, module, "probe"));

    constexpr unsigned int n = sizeof(kVectors) / sizeof(kVectors[0]);
    std::uint32_t host_a[n];
    std::uint32_t host_b[n];
    std::uint32_t host_c[n];
    std::uint32_t host_out[n] = {};
    for (unsigned int i = 0; i < n; ++i) {
        host_a[i] = kVectors[i].a;
        host_b[i] = kVectors[i].b;
        host_c[i] = kVectors[i].c;
    }

    CUdeviceptr device_a;
    CUdeviceptr device_b;
    CUdeviceptr device_c;
    CUdeviceptr device_out;
    check("cuMemAlloc(a)", cuMemAlloc(&device_a, sizeof(host_a)));
    check("cuMemAlloc(b)", cuMemAlloc(&device_b, sizeof(host_b)));
    check("cuMemAlloc(c)", cuMemAlloc(&device_c, sizeof(host_c)));
    check("cuMemAlloc(out)", cuMemAlloc(&device_out, sizeof(host_out)));

    check("cuMemcpyHtoD(a)", cuMemcpyHtoD(device_a, host_a, sizeof(host_a)));
    check("cuMemcpyHtoD(b)", cuMemcpyHtoD(device_b, host_b, sizeof(host_b)));
    check("cuMemcpyHtoD(c)", cuMemcpyHtoD(device_c, host_c, sizeof(host_c)));

    unsigned int device_n = n;
    void * args[] = {
        &device_a,
        &device_b,
        &device_c,
        &device_out,
        &device_n,
    };
    check("cuLaunchKernel", cuLaunchKernel(function,
                                            1, 1, 1,
                                            32, 1, 1,
                                            0, nullptr, args, nullptr));

    // An illegal instruction or device-side fault is reported here. The
    // supervisor must terminate this worker and must not reuse its context.
    check("cuCtxSynchronize", cuCtxSynchronize());
    check("cuMemcpyDtoH(out)", cuMemcpyDtoH(host_out, device_out, sizeof(host_out)));

    std::cout << "status=ok vectors=" << n << '\n';
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << "vector=" << kVectors[i].name
                  << " a=" << hex32(host_a[i])
                  << " b=" << hex32(host_b[i])
                  << " c=" << hex32(host_c[i])
                  << " out=" << hex32(host_out[i]) << '\n';
    }

    cuMemFree(device_out);
    cuMemFree(device_c);
    cuMemFree(device_b);
    cuMemFree(device_a);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
}
