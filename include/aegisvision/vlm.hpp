#pragma once
#include <opencv2/core.hpp>
#include <atomic>
#include <string>
#include <vector>
namespace aegisvision::vision {
struct SceneDescription {std::string summary;};
struct VlmConfig {int port{8092};int timeout_ms{60000};};
class IVlm {
public:
    virtual ~IVlm()=default;
    virtual SceneDescription describe(const cv::Mat& image,const std::atomic_bool* cancel=nullptr)=0;
};
// Local trusted llama.cpp only; no arbitrary hosts, prompts, tools or image URLs.
class LocalVlm final : public IVlm {
public:
    explicit LocalVlm(VlmConfig config={});
    SceneDescription describe(const cv::Mat& image,const std::atomic_bool* cancel=nullptr) override;
private: VlmConfig config_;
};
// Strict response contract; schema validity does not imply factual correctness.
SceneDescription parse_scene_response(const std::string& response);
}
