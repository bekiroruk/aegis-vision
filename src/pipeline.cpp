#include "aegisvision/pipeline.hpp"

#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace aegisvision {

AnalysisPipeline::AnalysisPipeline(
    PipelineConfig config,
    IDetector& detector,
    ITracker* tracker,
    IEmbedder* embedder,
    ITextExtractor* text_extractor,
    IVectorStore* vector_store)
    : config_(config),
      detector_(detector),
      tracker_(tracker),
      embedder_(embedder),
      text_extractor_(text_extractor),
      vector_store_(vector_store) {
    validate_components();
}

AnalysisResult AnalysisPipeline::analyze(const Frame& frame) {
    auto detections = detector_.detect(frame);
    std::vector<std::string> indexed_items;
    std::vector<std::string> extracted_text;

    for (std::size_t index = 0; index < detections.size(); ++index) {
        auto& detection = detections[index];
        if (config_.enable_embeddings) {
            detection.embedding = embedder_->embed_image(frame, detection);
            if (config_.index_embeddings) {
                const auto item_id = frame.source_id + ":" + frame.frame_id + ":" +
                    std::to_string(index);
                vector_store_->upsert(
                    item_id,
                    detection.embedding,
                    {
                        {"source_id", frame.source_id},
                        {"frame_id", frame.frame_id},
                        {"label", detection.label},
                        {"timestamp_ms", std::to_string(frame.timestamp_ms)},
                    });
                indexed_items.push_back(item_id);
            }
        }
        if (config_.enable_ocr) {
            auto text = text_extractor_->extract(frame, detection);
            extracted_text.insert(
                extracted_text.end(),
                std::make_move_iterator(text.begin()),
                std::make_move_iterator(text.end()));
        }
    }

    auto tracks = config_.enable_tracking
        ? tracker_->update(detections)
        : std::vector<Track>{};
    return AnalysisResult{
        frame.frame_id,
        std::move(detections),
        std::move(tracks),
        std::move(extracted_text),
        std::move(indexed_items),
    };
}

void AnalysisPipeline::validate_components() const {
    if (config_.enable_tracking && tracker_ == nullptr) {
        throw std::invalid_argument("Tracking is enabled but tracker is null");
    }
    if (config_.enable_embeddings && embedder_ == nullptr) {
        throw std::invalid_argument("Embeddings are enabled but embedder is null");
    }
    if (config_.index_embeddings && vector_store_ == nullptr) {
        throw std::invalid_argument("Embedding indexing is enabled but vector store is null");
    }
    if (config_.enable_ocr && text_extractor_ == nullptr) {
        throw std::invalid_argument("OCR is enabled but text extractor is null");
    }
}

}  // namespace aegisvision

