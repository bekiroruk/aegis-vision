#pragma once
#include "aegisvision/contracts.hpp"
#include <memory>

namespace aegisvision {
struct QdrantConfig {
    std::string host{"127.0.0.1"};
    int port{6333};
    std::string collection{"aegis_clip"};
    std::size_t dimension{512};
    std::string embedding_space;
    int timeout_seconds{10};
};
// Local HTTP only; external/TLS/auth deployment is intentionally not supported yet.
// Constructor checks/creates a collection without deleting/replacing existing data.
class QdrantVectorStore final : public IVectorStore {
public:
    explicit QdrantVectorStore(QdrantConfig config, bool create_if_missing=false);
    ~QdrantVectorStore();
    void upsert(std::string,std::vector<float>,std::map<std::string,std::string>) override;
    [[nodiscard]] std::vector<SearchResult> search(const std::vector<float>&,std::size_t limit=10) const override;
    [[nodiscard]] std::size_t point_count() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Stable UUIDv8 from SHA256(namespace, embedding space, item ID).
[[nodiscard]] std::string qdrant_point_id(std::string_view space,std::string_view item);
}
