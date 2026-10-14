#pragma once
#include "aegisvision/multicamera.hpp"
#include <map>

namespace aegisvision {
struct PersonObservation {
    std::uint64_t local_id{};
    std::vector<float> embedding;
};
struct ExtractedTracklet {
    CameraTracklet tracklet;
    std::size_t observations{};
    // missing_timeout is a tracker closure, NOT evidence of a physical exit.
    std::string closure;
};
// Consumes every consecutive frame, including empty frames. IDs must not be
// reused within an epoch. A prototype is the normalized mean of unit features.
class TrackletCollector {
public:
    TrackletCollector(std::string camera, std::uint64_t epoch, std::string space,
                      std::uint32_t max_missed_frames = 20);
    void update(std::int64_t relative_ms, const std::vector<PersonObservation>& observations);
    void finish(const std::string& reason); // end_of_input or frame_limit: censored
    [[nodiscard]] const std::vector<ExtractedTracklet>& results() const { return results_; }
private:
    struct Active {
        CameraTracklet tracklet;
        std::vector<double> sum;
        std::size_t count{}, last_frame{};
    };
    void close(std::map<std::uint64_t, Active>::iterator it, const std::string& reason);
    std::string camera_, space_;
    std::uint64_t epoch_;
    std::uint32_t max_missed_;
    std::size_t frame_{};
    std::int64_t last_ms_{-1};
    bool finished_{};
    std::map<std::uint64_t, Active> active_;
    std::vector<ExtractedTracklet> results_;
};
} // namespace aegisvision
