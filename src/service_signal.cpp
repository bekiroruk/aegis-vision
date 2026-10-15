#include "aegisvision/service.hpp"
#include <csignal>

namespace aegisvision {
namespace {
std::atomic_bool signal_requested{false}, signal_owned{false};
static_assert(std::atomic_bool::is_always_lock_free,
              "Shutdown signal handler needs a lock-free flag");
extern "C" void shutdown_signal_handler(int) {
    signal_requested.store(true, std::memory_order_relaxed);
}
class SignalScope {
public:
    SignalScope() {
        if (signal_owned.exchange(true)) throw std::logic_error("Shutdown handlers already owned");
        signal_requested.store(false);
#ifdef _WIN32
        old_int_ = std::signal(SIGINT, shutdown_signal_handler);
        if (old_int_ == SIG_ERR) { signal_owned = false; throw std::runtime_error("Cannot install SIGINT handler"); }
        old_term_ = std::signal(SIGTERM, shutdown_signal_handler);
        if (old_term_ == SIG_ERR) {
            std::signal(SIGINT, old_int_); signal_owned = false;
            throw std::runtime_error("Cannot install SIGTERM handler");
        }
#else
        struct sigaction action{};
        action.sa_handler = shutdown_signal_handler;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, &old_int_) != 0) {
            signal_owned = false; throw std::runtime_error("Cannot install SIGINT handler");
        }
        if (sigaction(SIGTERM, &action, &old_term_) != 0) {
            sigaction(SIGINT, &old_int_, nullptr); signal_owned = false;
            throw std::runtime_error("Cannot install SIGTERM handler");
        }
#endif
    }
    ~SignalScope() {
#ifdef _WIN32
        std::signal(SIGINT, old_int_); std::signal(SIGTERM, old_term_);
#else
        sigaction(SIGINT, &old_int_, nullptr); sigaction(SIGTERM, &old_term_, nullptr);
#endif
        signal_owned = false;
    }
    SignalScope(const SignalScope&) = delete;
    SignalScope& operator=(const SignalScope&) = delete;
private:
#ifdef _WIN32
    using Handler = void (*)(int);
    Handler old_int_{}, old_term_{};
#else
    struct sigaction old_int_{}, old_term_{};
#endif
};
}
bool LocalService::listen_until_signal() {
    if (port_ <= 0) throw std::logic_error("Bind before listening for shutdown");
    SignalScope signals;
    std::jthread observer([this](std::stop_token token) {
        while (!token.stop_requested()) {
            // httplib::stop() before listen starts is a no-op. Wait for the
            // running state, so an early signal cannot be lost in that window.
            if (signal_requested.load(std::memory_order_relaxed) && http_.is_running()) {
                stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    const bool ok = listen();
    observer.request_stop();
    observer.join();
    return ok;
}
}
