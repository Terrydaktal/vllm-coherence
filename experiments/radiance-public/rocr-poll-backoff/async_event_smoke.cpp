// Synthetic HSA signals only: no model data, GPU kernels, or device allocations.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <time.h>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
struct Event {
  hsa_signal_t signal{};
  std::atomic<unsigned> count{0};
  std::atomic<bool> stop{false};
  Clock::time_point observed;
};
static void check_status(hsa_status_t status, const char *operation) {
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "HSA failure " << status << " in " << operation << '\n';
    std::exit(1);
  }
}
#define check(operation) check_status((operation), #operation)
static bool callback(hsa_signal_value_t, void *opaque) {
  auto &event = *static_cast<Event *>(opaque);
  event.observed = Clock::now();
  hsa_signal_store_screlease(event.signal, 1);
  const bool keep = !event.stop.load();
  event.count.fetch_add(1, std::memory_order_release);
  return keep;
}
static double cpu_seconds() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value)) std::abort();
  return value.tv_sec + value.tv_nsec * 1e-9;
}
static double trigger(Event &event) {
  unsigned next = event.count.load(std::memory_order_acquire) + 1;
  auto started = Clock::now();
  hsa_signal_store_screlease(event.signal, 0);
  while (event.count.load(std::memory_order_acquire) < next) {
    if (Clock::now() - started > std::chrono::seconds(5)) {
      std::cerr << "lost callback\n";
      std::exit(2);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  return std::chrono::duration<double, std::micro>(event.observed - started).count();
}
int main() {
  // Fork before either process initializes HSA. Importing a signal from a
  // separate process creates an IPCSignal accepted by the async-handler API.
  int handles[2], done[2];
  if (pipe(handles) || pipe(done)) std::abort();
  pid_t child = fork();
  if (child < 0) std::abort();
  if (child == 0) {
    close(handles[0]); close(done[1]);
    check(hsa_init());
    hsa_signal_t signal;
    check(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_IPC, &signal));
    hsa_amd_ipc_signal_t handle;
    check(hsa_amd_ipc_signal_create(signal, &handle));
    if (write(handles[1], &handle, sizeof(handle)) != sizeof(handle)) std::abort();
    char byte;
    if (read(done[0], &byte, 1) != 1) std::abort();
    check(hsa_signal_destroy(signal));
    check(hsa_shut_down());
    _exit(0);
  }
  close(handles[1]); close(done[0]);
  check(hsa_init());
  Event polling, interrupt;
  hsa_amd_ipc_signal_t handle;
  if (read(handles[0], &handle, sizeof(handle)) != sizeof(handle)) std::abort();
  check(hsa_amd_ipc_signal_attach(&handle, &polling.signal));
  check(hsa_signal_create(1, 0, nullptr, &interrupt.signal));
  for (auto *event : {&polling, &interrupt})
    check(hsa_amd_signal_async_handler(event->signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                      callback, event));
  // Let registration settle before measuring the long idle wait.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto started = Clock::now();
  double cpu_start = cpu_seconds();
  std::this_thread::sleep_for(std::chrono::seconds(2));
  double cpu_elapsed = cpu_seconds() - cpu_start;
  double wall = std::chrono::duration<double>(Clock::now() - started).count();
  std::cout << "{\"idle_cpu_percent\":" << cpu_elapsed / wall * 100;
  for (auto *event : {&polling, &interrupt}) {
    std::vector<double> latencies;
    for (int i = 0; i < 200; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      latencies.push_back(trigger(*event));
    }
    std::sort(latencies.begin(), latencies.end());
    std::cout << ",\"" << (event == &polling ? "polling" : "interrupt")
              << "\":{\"callbacks\":" << event->count.load()
              << ",\"p50_us\":" << latencies[100]
              << ",\"p99_us\":" << latencies[198]
              << ",\"max_us\":" << latencies.back() << '}';
  }
  for (auto *event : {&polling, &interrupt}) {
    event->stop.store(true);
    trigger(*event);
  }
  for (auto *event : {&polling, &interrupt}) check(hsa_signal_destroy(event->signal));
  check(hsa_shut_down());
  if (write(done[1], "x", 1) != 1) std::abort();
  close(done[1]); close(handles[0]);
  int status;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status))
    std::abort();
  std::cout << ",\"passed\":true}\n";
}
