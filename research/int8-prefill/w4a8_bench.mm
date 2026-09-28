// Host for w4a8_bench.metal. Build/run: ./run.sh
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

struct Params { uint32_t output_size; uint32_t input_size; };
constexpr uint32_t kTileM = 32, kGroup = 64, kWeightTileN = 256;

using bf16 = uint16_t;
inline bf16 toBf16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1u; u += 0x7FFFu + lsb; return bf16(u >> 16);
}
inline float fromBf16(bf16 b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

struct Ctx {
  id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
  id<MTLComputePipelineState> pipeline(const std::string &name) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    id<MTLFunction> fn = [library newFunctionWithName:@(name.c_str())];
    if (!fn) { fprintf(stderr, "missing kernel %s\n", name.c_str()); exit(1); }
    NSError *error = nil;
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) { fprintf(stderr, "pipeline %s: %s\n", name.c_str(), error.localizedDescription.UTF8String); exit(1); }
    pipelines[name] = p; return p;
  }
  id<MTLBuffer> buffer(uint64_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:std::max<uint64_t>(bytes, 16) options:MTLResourceStorageModeShared];
    if (!b) { fprintf(stderr, "alloc %llu failed\n", bytes); exit(1); }
    return b;
  }
};

struct Dispatch {
  std::string name; std::vector<id<MTLBuffer>> buffers; Params params;
  MTLSize groups; MTLSize threads{256, 1, 1};
};

double submit(Ctx &c, const std::vector<Dispatch> &dispatches) {
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  for (const Dispatch &d : dispatches) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c.pipeline(d.name)];
    for (size_t i = 0; i < d.buffers.size(); ++i) [enc setBuffer:d.buffers[i] offset:0 atIndex:i];
    [enc setBytes:&d.params length:sizeof(d.params) atIndex:d.buffers.size()];
    [enc dispatchThreadgroups:d.groups threadsPerThreadgroup:d.threads];
    [enc endEncoding];
  }
  [cb commit]; [cb waitUntilCompleted];
  if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
  return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

double timeIt(Ctx &c, const std::vector<Dispatch> &d, int repeats = 7) {
  for (int i = 0; i < 2; ++i) submit(c, d);
  std::vector<double> s; for (int i = 0; i < repeats; ++i) s.push_back(submit(c, d));
  std::sort(s.begin(), s.end()); return s[s.size() / 2];
}

// Weights: [tile][group][col in 256][64 nibbles], low nibble = even k.
struct Layer {
  uint32_t K, N;
  std::vector<uint8_t> codes;       // K*N unsigned 4-bit codes, [n][k]
  std::vector<float> scale, bias;   // [n][g]
  id<MTLBuffer> weightsU, weightsS, scales, biases, biasesS;
};

uint64_t paramIndex(uint32_t K, uint32_t n, uint32_t g) {
  uint32_t groups = K / kGroup;
  return (uint64_t(n / kWeightTileN) * groups + g) * kWeightTileN + n % kWeightTileN;
}

Layer makeLayer(Ctx &c, uint32_t K, uint32_t N, std::mt19937 &rng, bool fill) {
  Layer L{K, N};
  uint32_t groups = K / kGroup;
  uint64_t wbytes = uint64_t(K) * N / 2, pcount = uint64_t(K) * N / kGroup;
  L.weightsU = c.buffer(wbytes); L.weightsS = c.buffer(wbytes);
  L.scales = c.buffer(pcount * 2); L.biases = c.buffer(pcount * 2); L.biasesS = c.buffer(pcount * 2);
  auto *wu = (uint8_t *)L.weightsU.contents; auto *ws = (uint8_t *)L.weightsS.contents;
  auto *sc = (bf16 *)L.scales.contents; auto *bi = (bf16 *)L.biases.contents; auto *bs = (bf16 *)L.biasesS.contents;
  if (!fill) {
    // Timing only: resident nonzero bytes.
    std::memset(wu, 0x5a, wbytes); std::memset(ws, 0x5a ^ 0x88, wbytes);
    std::memset(sc, 0x3c, pcount * 2); std::memset(bi, 0x3c, pcount * 2); std::memset(bs, 0x3c, pcount * 2);
    return L;
  }
  L.codes.resize(uint64_t(K) * N); L.scale.resize(uint64_t(N) * groups); L.bias.resize(uint64_t(N) * groups);
  std::uniform_int_distribution<int> q(0, 15);
  std::uniform_real_distribution<float> sd(0.005f, 0.05f), bd(-0.3f, 0.0f);
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t g = 0; g < groups; ++g) {
      float s = fromBf16(toBf16(sd(rng))), b = fromBf16(toBf16(bd(rng) * s * 16.0f));
      L.scale[uint64_t(n) * groups + g] = s; L.bias[uint64_t(n) * groups + g] = b;
      uint64_t p = paramIndex(K, n, g);
      sc[p] = toBf16(s); bi[p] = toBf16(b); bs[p] = toBf16(b + 8.0f * s);
      for (uint32_t k = g * kGroup; k < (g + 1) * kGroup; ++k) {
        uint8_t code = uint8_t(q(rng)); L.codes[uint64_t(n) * K + k] = code;
        uint64_t nib = (uint64_t(n / kWeightTileN) * groups + g) * kWeightTileN * kGroup + uint64_t(n % kWeightTileN) * kGroup + (k % kGroup);
        uint64_t byte = nib / 2; uint32_t shift = (nib % 2) * 4;
        wu[byte] |= uint8_t(code << shift);
        ws[byte] |= uint8_t(((code - 8) & 0xF) << shift); // == code ^ 8
      }
    }
  }
  return L;
}

struct Acts {
  uint32_t M, K;
  std::vector<float> x;
  id<MTLBuffer> input, sums, quantized, rowScales, scaledSums;
};

Acts makeActs(Ctx &c, uint32_t M, uint32_t K, std::mt19937 &rng, bool fill) {
  Acts A{M, K};
  uint32_t groups = K / kGroup;
  A.input = c.buffer(uint64_t(M) * K * 2); A.sums = c.buffer(uint64_t(M) * groups * 4);
  A.quantized = c.buffer(uint64_t(M) * K); A.rowScales = c.buffer(uint64_t(M) * 4);
  A.scaledSums = c.buffer(uint64_t(M) * groups * 4);
  auto *in = (bf16 *)A.input.contents;
  if (!fill) { std::memset(in, 0x3c, uint64_t(M) * K * 2); return A; }
  A.x.resize(uint64_t(M) * K);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (uint64_t i = 0; i < A.x.size(); ++i) {
    float v = nd(rng);
    if (i % 977 == 0) v *= 12.0f; // outlier channels, post-RMSNorm style
    in[i] = toBf16(v); A.x[i] = fromBf16(in[i]);
  }
  return A;
}

std::vector<Dispatch> bf16Pass(const Acts &A, const Layer &L, const std::string &kernel, uint32_t tileN, id<MTLBuffer> out, bool withSums) {
  Params p{L.N, L.K};
  std::vector<Dispatch> d;
  if (withSums) d.push_back({"sums32", {A.input, A.sums}, p, {A.M / kTileM, 1, 1}});
  d.push_back({kernel, {A.input, L.weightsU, L.scales, L.biases, out, A.sums, A.rowScales}, p, {A.M / kTileM, L.N / tileN, 1}});
  return d;
}
std::vector<Dispatch> i8Pass(const Acts &A, const Layer &L, const std::string &kernel, uint32_t tileN, id<MTLBuffer> out, bool withQuant) {
  Params p{L.N, L.K};
  std::vector<Dispatch> d;
  if (withQuant) d.push_back({"quant_rows", {A.input, A.quantized, A.rowScales, A.scaledSums}, p, {A.M / kTileM, 1, 1}});
  d.push_back({kernel, {A.quantized, L.weightsS, L.scales, L.biasesS, out, A.scaledSums, A.rowScales}, p, {A.M / kTileM, L.N / tileN, 1}});
  return d;
}

bool verify(Ctx &c) {
  std::mt19937 rng(7);
  const uint32_t M = 64, K = 1024, N = 512;
  Layer L = makeLayer(c, K, N, rng, true);
  Acts A = makeActs(c, M, K, rng, true);
  uint32_t groups = K / kGroup;
  id<MTLBuffer> outB = c.buffer(uint64_t(M) * N * 2), outI = c.buffer(uint64_t(M) * N * 2);
  submit(c, bf16Pass(A, L, "bf16_n128", 128, outB, true));
  submit(c, i8Pass(A, L, "i8_n128", 128, outI, true));
  // CPU references.
  std::vector<float> refExact(uint64_t(M) * N), refI8(uint64_t(M) * N);
  const auto *xq = (const int8_t *)A.quantized.contents; const auto *rs = (const float *)A.rowScales.contents;
  double maxAbsExact = 0, maxAbsI8 = 0, maxAbsQuantVsExact = 0, sumSq = 0, sumSqDiff = 0;
  double refScaleMax = 0;
  for (uint32_t m = 0; m < M; ++m) {
    // check quantizer
    float peak = 0; for (uint32_t k = 0; k < K; ++k) peak = std::max(peak, std::fabs(A.x[uint64_t(m) * K + k]));
    float s = peak / 127.0f;
    if (std::fabs(rs[m] - s) > 1e-6f * s) { printf("row scale mismatch row %u: %g vs %g\n", m, rs[m], s); return false; }
    for (uint32_t n = 0; n < N; ++n) {
      double exact = 0, i8 = 0;
      for (uint32_t g = 0; g < groups; ++g) {
        double dot = 0, sum = 0, dotq = 0;
        for (uint32_t k = g * kGroup; k < (g + 1) * kGroup; ++k) {
          float xv = A.x[uint64_t(m) * K + k]; int code = L.codes[uint64_t(n) * K + k];
          dot += double(xv) * code; sum += xv;
          dotq += double(xq[uint64_t(m) * K + k]) * (code - 8);
        }
        float sc = L.scale[uint64_t(n) * groups + g], bi = L.bias[uint64_t(n) * groups + g];
        float biS = fromBf16(toBf16(bi + 8.0f * sc));
        exact += dot * sc + sum * bi;
        i8 += dotq * sc + (sum / rs[m]) * biS;
      }
      i8 *= rs[m];
      refExact[uint64_t(m) * N + n] = float(exact); refI8[uint64_t(m) * N + n] = float(i8);
      float gb = fromBf16(((bf16 *)outB.contents)[uint64_t(m) * N + n]);
      float gi = fromBf16(((bf16 *)outI.contents)[uint64_t(m) * N + n]);
      maxAbsExact = std::max(maxAbsExact, std::fabs(double(gb) - exact));
      maxAbsI8 = std::max(maxAbsI8, std::fabs(double(gi) - i8));
      maxAbsQuantVsExact = std::max(maxAbsQuantVsExact, std::fabs(double(gi) - double(gb)));
      refScaleMax = std::max(refScaleMax, std::fabs(exact));
      sumSq += exact * exact; sumSqDiff += (double(gi) - exact) * (double(gi) - exact);
    }
  }
  printf("verify M=%u K=%u N=%u\n", M, K, N);
  printf("  bf16 kernel vs fp64 reference   : max abs %.4g (outputs up to %.3g)\n", maxAbsExact, refScaleMax);
  printf("  int8 kernel vs fp64 int8 model  : max abs %.4g  (layout/sign fold check)\n", maxAbsI8);
  printf("  int8 kernel vs bf16 kernel      : max abs %.4g, rel RMS %.4g\n", maxAbsQuantVsExact, std::sqrt(sumSqDiff / sumSq));
  bool ok = maxAbsExact < 0.02 * refScaleMax && maxAbsI8 < 0.02 * refScaleMax;
  printf("  %s\n", ok ? "OK" : "MISMATCH");
  return ok;
}

void bench(Ctx &c) {
  std::mt19937 rng(1);
  struct Shape { const char *name; uint32_t K, N; };
  const Shape shapes[] = {{"ffn_up 5120->17408", 5120, 17408}, {"ffn_down 17408->5120", 17408, 5120},
                          {"attn_in 5120->14336", 5120, 14336}, {"gdn_in 5120->16640", 5120, 16640},
                          {"mixer_out 6144->5120", 6144, 5120}};
  const uint32_t rowsList[] = {512, 2048, 8192};
  printf("\n%-22s %5s | %8s %8s %8s %8s | %8s %8s %8s %8s | %6s %6s | %8s %8s\n", "shape", "M",
         "bf16n128", "bf16n256", "i8n128", "i8n256", "rawB128", "rawB256", "rawI128", "rawI256",
         "best x", "raw x", "sums", "quant");
  printf("%-22s %5s | %8s %8s %8s %8s | %8s %8s %8s %8s | %6s %6s | %8s %8s\n", "", "",
         "ms", "ms", "ms", "ms", "ms", "ms", "ms", "ms", "", "", "ms", "ms");
  for (const Shape &s : shapes) {
    Layer L = makeLayer(c, s.K, s.N, rng, false);
    for (uint32_t M : rowsList) {
      Acts A = makeActs(c, M, s.K, rng, false);
      id<MTLBuffer> out = c.buffer(uint64_t(M) * s.N * 2);
      // producers once so the buffers are valid; timed separately
      double sums = timeIt(c, {{"sums32", {A.input, A.sums}, Params{s.N, s.K}, {M / kTileM, 1, 1}}});
      double quant = timeIt(c, {{"quant_rows", {A.input, A.quantized, A.rowScales, A.scaledSums}, Params{s.N, s.K}, {M / kTileM, 1, 1}}});
      double b128 = timeIt(c, bf16Pass(A, L, "bf16_n128", 128, out, false));
      double b256 = timeIt(c, bf16Pass(A, L, "bf16_n256", 256, out, false));
      double i128 = timeIt(c, i8Pass(A, L, "i8_n128", 128, out, false));
      double i256 = timeIt(c, i8Pass(A, L, "i8_n256", 256, out, false));
      double rb128 = timeIt(c, bf16Pass(A, L, "bf16_n128_raw", 128, out, false));
      double rb256 = timeIt(c, bf16Pass(A, L, "bf16_n256_raw", 256, out, false));
      double ri128 = timeIt(c, i8Pass(A, L, "i8_n128_raw", 128, out, false));
      double ri256 = timeIt(c, i8Pass(A, L, "i8_n256_raw", 256, out, false));
      double bs256 = timeIt(c, bf16Pass(A, L, "bf16_n256_scale", 256, out, false));
      double is256 = timeIt(c, i8Pass(A, L, "i8_n256_scale", 256, out, false));
      double ig256 = timeIt(c, i8Pass(A, L, "i8_n256_gscale", 256, out, false));
      double bestB = std::min(b128, b256), bestI = std::min(i128, i256);
      double flops = 2.0 * M * double(s.K) * s.N;
      printf("%-22s %5u | %8.3f %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f %8.3f | %6.2f %6.2f | %8.3f %8.3f   bf16 %.1f TF, i8 %.1f TOPS, rawI %.1f TOPS\n",
             s.name, M, b128, b256, i128, i256, rb128, rb256, ri128, ri256, bestB / bestI,
             std::min(rb128, rb256) / std::min(ri128, ri256), sums, quant,
             flops / bestB / 1e9, flops / bestI / 1e9, flops / std::min(ri128, ri256) / 1e9);
      printf("%-22s %5s | epilogue variants (n256): bf16 scale-only %.3f ms (%.2fx vs bf16), i8 scale-only %.3f ms (%.2fx), i8 group-scale %.3f ms (%.2fx)\n",
             "", "", bs256, b256 / bs256, is256, b256 / is256, ig256, b256 / ig256);
    }
  }
}

// Sustained load: long command buffers of one kernel, alternating bf16 and
// int8 rounds so clock/thermal drift hits both. ffn_up at M=2048 (the
// production prefill chunk).
void sustain(Ctx &c) {
  std::mt19937 rng(3);
  const uint32_t M = 2048, K = 5120, N = 17408, perBuffer = 96, rounds = 8;
  Layer L = makeLayer(c, K, N, rng, false);
  Acts A = makeActs(c, M, K, rng, false);
  id<MTLBuffer> out = c.buffer(uint64_t(M) * N * 2);
  struct Variant { const char *name; std::vector<Dispatch> one; std::vector<double> perDispatch; };
  std::vector<Variant> variants = {
      {"bf16_n256", bf16Pass(A, L, "bf16_n256", 256, out, false), {}},
      {"i8_n256", i8Pass(A, L, "i8_n256", 256, out, false), {}},
      {"i8_n256_scale", i8Pass(A, L, "i8_n256_scale", 256, out, false), {}},
      {"bf16_n128_sg4-like: bf16_n128", bf16Pass(A, L, "bf16_n128", 128, out, false), {}},
  };
  printf("\nsustained ffn_up M=%u: %u dispatches per command buffer, %u rounds, alternating\n", M, perBuffer, rounds);
  for (uint32_t r = 0; r < rounds; ++r) {
    for (Variant &v : variants) {
      std::vector<Dispatch> many;
      for (uint32_t i = 0; i < perBuffer; ++i) many.push_back(v.one[0]);
      double ms = submit(c, many) / perBuffer;
      v.perDispatch.push_back(ms);
      printf("  round %u %-32s %.3f ms/dispatch\n", r, v.name, ms);
    }
  }
  for (Variant &v : variants) {
    std::sort(v.perDispatch.begin(), v.perDispatch.end());
    printf("sustained median %-32s %.3f ms  (min %.3f, max %.3f)\n", v.name,
           v.perDispatch[v.perDispatch.size() / 2], v.perDispatch.front(), v.perDispatch.back());
  }
  printf("sustained i8_n256 / bf16_n256 median ratio: %.2fx\n",
         variants[0].perDispatch[rounds / 2] / variants[1].perDispatch[rounds / 2]);
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: w4a8_bench <metallib> [--sustain]\n"); return 2; }
    const bool onlySustain = argc > 2 && std::string(argv[2]) == "--sustain";
    Ctx c; c.device = MTLCreateSystemDefaultDevice(); c.queue = [c.device newCommandQueue];
    NSError *error = nil;
    c.library = [c.device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!c.library) { fprintf(stderr, "metallib: %s\n", error.localizedDescription.UTF8String); return 1; }
    printf("device=%s\n", c.device.name.UTF8String);
    if (onlySustain) { sustain(c); return 0; }
    bool ok = verify(c);
    bench(c);
    sustain(c);
    return ok ? 0 : 1;
  }
}
