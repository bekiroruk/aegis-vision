#pragma once
#include "aegisvision/yolo.hpp"
#include <memory>
#include <nlohmann/json.hpp>

namespace aegisvision::vision {
// Offline builder: static YOLOv8 detection only, no plugins or downloaded engines.
// Output directory must not exist. FP16 permits mixed precision; I/O remains FP32.
// Nonempty calibration manifest enables INT8-mixed and requires fp16=true.
void build_yolo_tensorrt(const std::filesystem::path& onnx,
    const std::filesystem::path& new_bundle, bool fp16,
    const std::filesystem::path& calibration_manifest = {});

// Trust only engines built locally/from a trusted source. Hashes detect mutation,
// not malicious plans. One instance per worker; device 0, synchronous detect().
class TensorRtYoloDetector final : public IDetector {
public:
    explicit TensorRtYoloDetector(const std::filesystem::path& bundle, YoloConfig config = {});
    ~TensorRtYoloDetector() override;
    TensorRtYoloDetector(const TensorRtYoloDetector&) = delete;
    TensorRtYoloDetector& operator=(const TensorRtYoloDetector&) = delete;
    [[nodiscard]] std::vector<Detection> detect(const Frame& frame) override;
    [[nodiscard]] nlohmann::json metadata() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aegisvision::vision
