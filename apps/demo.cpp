#include "aegisvision/adapters.hpp"
#include "aegisvision/pipeline.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/vector_store.hpp"

#include <iostream>
#include <vector>

int main() {
    using namespace aegisvision;
    PayloadDetector detector;
    IoUTracker tracker;
    HashEmbedder embedder;
    InMemoryVectorStore store;
    AnalysisPipeline pipeline{PipelineConfig{}, detector, &tracker, &embedder, nullptr, &store};

    const std::vector<Frame> frames{
        Frame{"0001", "camera-1", 0, {
            Detection{BoundingBox{10, 20, 110, 220}, "person", 0.94F},
            Detection{BoundingBox{250, 80, 390, 190}, "car", 0.88F},
        }},
        Frame{"0002", "camera-1", 40, {
            Detection{BoundingBox{14, 20, 114, 220}, "person", 0.96F},
        }},
    };

    for (const auto& frame : frames) {
        const auto result = pipeline.analyze(frame);
        std::cout << "frame=" << result.frame_id
                  << " detections=" << result.detections.size()
                  << " tracks=" << result.tracks.size() << '\n';
        for (const auto& track : result.tracks) {
            std::cout << "  track_id=" << track.track_id << " label=" << track.label
                      << " score=" << track.score << " age=" << track.age << '\n';
        }
    }
    const auto matches = store.search(embedder.embed_text("person"), 3);
    std::cout << "indexed_matches=" << matches.size() << '\n';
    return 0;
}

