/*
Profile an NCNN network's inference time, feeding it random input data --
C++ counterpart to profile_ncnn.py's --threads/--thread-instances timing
paths. There is no --processes here (a shell loop already does that fine,
in either language) and no static parameter/MACs analysis (use
profile_ncnn.py for that -- it's a plain-text .param parse, unrelated to
timing, and not worth duplicating).

Why this exists: ncnn's Python bindings do not release the Global
Interpreter Lock in Net::load_param/load_model or Extractor::input/extract
(confirmed both empirically -- a busy Python loop running concurrently
with extract() calls drops to ~12% of its normal throughput -- and by
inspecting ncnn's own python/src/main.cpp, where none of those .def(...)
bindings carry a py::call_guard<py::gil_scoped_release>()). So
profile_ncnn.py's --thread-instances serializes on the GIL and shows no
real speedup over --threads 1, even though --processes (separate OS
processes, separate GILs) does. This program has no interpreter and no GIL
to contend with, so its --thread-instances genuinely runs concurrently on
separate cores -- it exists to demonstrate that, and to give real
concurrent-throughput numbers when that's what's actually wanted.

Compile (from the repo root):

    ./build_profile_ncnn.sh

Usage:
    ./profile_ncnn <name> [options]

<name> is the network's .param/.bin pair, without the extension, e.g.

    ./profile_ncnn ncnn/imx662_32_2221e1221_1936x1104_fp16

Options mirror profile_ncnn.py's --width/--height/--channels/--threads/
--warmup/--loops/--fp32/--input/--output/--thread-instances; run with
--help for details.
*/
#include <ncnn/cpu.h>
#include <ncnn/net.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Args
{
    std::string name;
    int width = 968;
    int height = 552;
    int channels = 4;
    int threads = ncnn::get_cpu_count();
    int warmup = 3;
    int loops = 10;
    bool fp32 = false;
    std::string input_name;
    std::string output_name;
    int thread_instances = 1;
};

static void print_usage(const char* argv0)
{
    std::printf(
        "Usage: %s <name> [options]\n\n"
        "  --width N              input width (default: 968)\n"
        "  --height N             input height (default: 552)\n"
        "  --channels N           input channels (default: 4)\n"
        "  --threads N            ncnn worker threads, per instance (default: all cores)\n"
        "  --warmup N             warm-up runs, discarded (default: 3)\n"
        "  --loops N              timed runs to average (default: 10)\n"
        "  --fp32                 disable fp16 storage/arithmetic\n"
        "  --input NAME           input blob name (default: autodetect)\n"
        "  --output NAME          output blob name (default: autodetect)\n"
        "  --thread-instances N   run this many independent ncnn instances concurrently,\n"
        "                         as std::thread workers of this one process, and report\n"
        "                         aggregate ms/call across all of them (default: 1)\n",
        argv0);
}

static bool parse_args(int argc, char** argv, Args& a)
{
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        auto next_int = [&](int& dst) {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", arg.c_str()); exit(2); }
            dst = std::atoi(argv[++i]);
        };
        auto next_str = [&](std::string& dst) {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", arg.c_str()); exit(2); }
            dst = argv[++i];
        };

        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); exit(0); }
        else if (arg == "--width") next_int(a.width);
        else if (arg == "--height") next_int(a.height);
        else if (arg == "--channels") next_int(a.channels);
        else if (arg == "--threads") next_int(a.threads);
        else if (arg == "--warmup") next_int(a.warmup);
        else if (arg == "--loops") next_int(a.loops);
        else if (arg == "--fp32") a.fp32 = true;
        else if (arg == "--input") next_str(a.input_name);
        else if (arg == "--output") next_str(a.output_name);
        else if (arg == "--thread-instances") next_int(a.thread_instances);
        else if (!arg.empty() && arg[0] == '-') { std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); return false; }
        else positional.push_back(arg);
    }
    if (positional.size() != 1) { std::fprintf(stderr, "expected exactly one <name> argument\n"); return false; }
    a.name = positional[0];
    return true;
}

// Result of one instance's run: per-call times in ms, plus the blob names
// and output shape it resolved (for the single-instance report).
struct RunResult
{
    std::vector<double> times_ms;
    std::string input_name, output_name;
    int out_w = 0, out_h = 0, out_c = 0;
};

// Loads its own net, runs warmup then (after release()) the timed loop.
// release() is called once, right after warmup -- the caller supplies
// either a no-op (single-instance) or a barrier wait (thread-instances),
// exactly mirroring profile_ncnn.py's run_inference()/pre_loop_hook.
template <typename ReleaseFn>
static RunResult run_inference(const Args& a, ReleaseFn release)
{
    ncnn::Net net;
    net.opt.num_threads = a.threads;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = !a.fp32;
    net.opt.use_fp16_storage = !a.fp32;
    net.opt.use_fp16_arithmetic = !a.fp32;

    if (net.load_param((a.name + ".param").c_str()) != 0)
        throw std::runtime_error("failed to load " + a.name + ".param");
    if (net.load_model((a.name + ".bin").c_str()) != 0)
        throw std::runtime_error("failed to load " + a.name + ".bin");

    RunResult r;
    r.input_name = !a.input_name.empty() ? a.input_name : net.input_names()[0];
    r.output_name = !a.output_name.empty() ? a.output_name : net.output_names()[0];

    ncnn::Mat in_mat(a.width, a.height, a.channels);
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float* p = (float*)in_mat.data;
        for (size_t i = 0; i < in_mat.total(); i++) p[i] = dist(rng);
    }

    ncnn::Mat out;
    auto run_once = [&]() {
        ncnn::Extractor ex = net.create_extractor();
        ex.input(r.input_name.c_str(), in_mat);
        int ret = ex.extract(r.output_name.c_str(), out);
        if (ret != 0) throw std::runtime_error("extract() failed (" + std::to_string(ret) + ")");
    };

    for (int i = 0; i < a.warmup; i++) run_once();

    release();

    r.times_ms.reserve(a.loops);
    for (int i = 0; i < a.loops; i++)
    {
        auto t0 = std::chrono::steady_clock::now();
        run_once();
        r.times_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    r.out_w = out.w; r.out_h = out.h; r.out_c = out.c;
    return r;
}

static void print_stats(const std::vector<double>& times_in)
{
    std::vector<double> times = times_in;
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    std::printf("min=%.2f median=%.2f max=%.2f mean=%.2f",
                times.front(), times[times.size() / 2], times.back(), sum / times.size());
}

static void single_main(const Args& a)
{
    RunResult r = run_inference(a, [] {});

    std::printf("Network:  %s\n", a.name.c_str());
    std::printf("Input:    '%s' %dx%dx%d\n", r.input_name.c_str(), a.width, a.height, a.channels);
    std::printf("Output:   '%s'\n", r.output_name.c_str());
    std::printf("Threads:  %d   fp16: %s\n", a.threads, a.fp32 ? "false" : "true");
    std::printf("Output:   %dx%dx%d\n", r.out_w, r.out_h, r.out_c);

    std::printf("\n%d runs, ms: ", a.loops);
    print_stats(r.times_ms);
    std::printf("\n");
}

// Manual barrier (portable pre-C++20): every instance loads its model and
// runs warmup, then blocks here, so all N timed loops start together --
// mirroring profile_ncnn.py's threading.Barrier synchronization.
class Barrier
{
public:
    explicit Barrier(int n) : n_(n), count_(n) {}
    void wait()
    {
        std::unique_lock<std::mutex> lock(m_);
        int gen = gen_;
        if (--count_ == 0) { count_ = n_; gen_++; cv_.notify_all(); }
        else cv_.wait(lock, [&] { return gen != gen_; });
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    int n_, count_, gen_ = 0;
};

static void thread_main(const Args& a)
{
    int n = a.thread_instances;
    std::printf("Network:  %s\n", a.name.c_str());
    std::printf("Input:    %dx%dx%d\n", a.width, a.height, a.channels);
    std::printf("Threads:  %d concurrent ncnn instances in this process (each threads=%d, fp16=%s)\n",
                n, a.threads, a.fp32 ? "false" : "true");
    std::printf("Per instance: %d warmup + %d timed runs\n", a.warmup, a.loops);

    Barrier barrier(n);
    std::vector<RunResult> results(n);
    std::vector<std::string> errors(n);

    auto worker = [&](int i) {
        try { results[i] = run_inference(a, [&] { barrier.wait(); }); }
        catch (const std::exception& e) { errors[i] = e.what(); }
    };

    std::vector<std::thread> threads;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();
    double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    int ok = 0;
    double concurrent_s = 0.0;
    size_t total_calls = 0;
    for (int i = 0; i < n; i++)
    {
        if (!errors[i].empty()) { std::printf("  instance %d: FAILED: %s\n", i, errors[i].c_str()); continue; }
        ok++;
        double sum = std::accumulate(results[i].times_ms.begin(), results[i].times_ms.end(), 0.0);
        concurrent_s = std::max(concurrent_s, sum / 1000.0);
        total_calls += results[i].times_ms.size();
    }
    if (ok < n)
        std::printf("\n%d/%d thread instance(s) failed -- aggregate below only reflects the %d that succeeded\n",
                    n - ok, n, ok);

    std::printf("\n%d concurrent thread instances completed, %zu total calls, load+warmup+timed wall=%.1fms\n",
                ok, total_calls, wall_ms);
    if (total_calls > 0 && concurrent_s > 0)
        std::printf("Aggregate throughput: %.2f ms/call (= %.1f calls/s across all %d instances)\n",
                    concurrent_s * 1000.0 / total_calls, total_calls / concurrent_s, ok);

    for (int i = 0; i < n; i++)
    {
        if (!errors[i].empty()) continue;
        std::printf("  instance %d: ", i);
        print_stats(results[i].times_ms);
        std::printf("\n");
    }
}

int main(int argc, char** argv)
{
    Args a;
    if (!parse_args(argc, argv, a)) { print_usage(argv[0]); return 2; }

    std::printf("threads=%d  thread-instances=%d\n", a.threads, a.thread_instances);

    try
    {
        if (a.thread_instances > 1) thread_main(a);
        else single_main(a);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
