#include "aegisvision/tracklets.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace aegisvision;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Expected rejection");
}
std::vector<float> axis(int dimension, float scale = 1) {
    std::vector<float> v(512); v[dimension] = scale; return v;
}
void tests() {
    TrackletCollector c("cam", 7, "osnet", 1);
    c.update(0, {{9, axis(0, 4)}, {1, axis(2)}});
    c.update(100, {{9, axis(1, 10)}});
    check(c.results().empty(), "Timeout fired too early");
    c.update(200, {});
    check(c.results().size() == 1 && c.results()[0].tracklet.local_id == 1, "Wrong timeout track");
    c.update(300, {});
    check(c.results().size() == 2, "Timeout did not close");
    const auto& t = c.results()[1];
    check(t.closure == "missing_timeout" && t.observations == 2, "Wrong closure");
    check(t.tracklet.start_ms == 0 && t.tracklet.end_ms == 100, "End time includes missing frames");
    check(t.tracklet.epoch == 7 && t.tracklet.camera == "cam" && t.tracklet.embedding_space == "osnet", "Namespace lost");
    check(std::abs(t.tracklet.embedding[0] - std::sqrt(.5F)) < 1e-6 &&
          std::abs(t.tracklet.embedding[1] - std::sqrt(.5F)) < 1e-6, "Mean is not unit-feature mean");
    rejects([&] { c.update(400, {{9, axis(0)}}); });
    c.update(400, {{10, axis(0)}});
    c.finish("frame_limit");
    check(c.results().back().closure == "frame_limit", "Truncation not censored");
    rejects([&] { c.update(500, {}); });
    rejects([&] { c.finish("end_of_input"); });

    TrackletCollector e("cam", 1, "space", 0);
    rejects([&] { e.update(-1, {}); });
    rejects([&] { e.update(0, {{1, axis(0)}, {1, axis(1)}}); });
    rejects([&] { e.update(0, {{0, axis(0)}}); });
    rejects([&] { e.update(0, {{1, {1, 2}}}); });
    auto nan = axis(0); nan[5] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { e.update(0, {{1, nan}}); });
    e.update(0, {{1, axis(0)}}); // invalid frames did not mutate state
    rejects([&] { e.update(0, {}); });
    e.finish("end_of_input");
    check(e.results().front().closure == "end_of_input", "EOF not censored");
    rejects([] { TrackletCollector invalid("../camera", 1, "space"); });
    rejects([] { TrackletCollector invalid("cam", 0, "space"); });
    rejects([] { TrackletCollector invalid("cam", 1, ""); });

    TrackletCollector bounded("cam", 1, "space", 0);
    std::vector<PersonObservation> samples;
    for (std::uint64_t id = 1; id <= 512; ++id) samples.push_back({id, axis(0)});
    bounded.update(0, samples);
    bounded.update(1, {});
    rejects([&] { bounded.update(2, {{513, axis(0)}}); });
    bounded.finish("end_of_input");
    check(bounded.results().size() == 512, "Capacity boundary lost tracks");

    TrackletCollector empty("cam", 1, "space");
    empty.update(0, {}); empty.finish("end_of_input");
    check(empty.results().empty(), "Empty frame produced a person");
}
}
int main() {
    try { tests(); std::cout << "Tracklet collector tests passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
