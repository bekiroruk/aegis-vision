#pragma once

#include "aegisvision/config.hpp"
#include "aegisvision/contracts.hpp"

namespace aegisvision {

class AnalysisPipeline {
public:
    AnalysisPipeline(
        PipelineConfig config,
        IDetector& detector,
        ITracker* tracker,
        IEmbedder* embedder,
        ITextExtractor* text_extractor,
        IVectorStore* vector_store);
    [[nodiscard]] AnalysisResult analyze(const Frame& frame);

private:
    void validate_components() const;
    PipelineConfig config_;
    IDetector& detector_;
    ITracker* tracker_;
    IEmbedder* embedder_;
    ITextExtractor* text_extractor_;
    IVectorStore* vector_store_;
};

}  // namespace aegisvision

