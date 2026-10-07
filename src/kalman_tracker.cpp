#include "aegisvision/kalman_tracker.hpp"
#include "aegisvision/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace aegisvision {
namespace {
using Mean = std::array<double, 8>;
using Covariance = std::array<double, 64>;
using Measurement = std::array<double, 4>;
using Matrix4 = std::array<double, 16>;
constexpr double position_weight = 1.0 / 20.0;
constexpr double velocity_weight = 1.0 / 160.0;
constexpr double max_coordinate = 1'000'000.0;

Measurement measurement(const BoundingBox& box) {
    const double width = static_cast<double>(box.x2) - box.x1;
    const double height = static_cast<double>(box.y2) - box.y1;
    return {static_cast<double>(box.x1) + width * 0.5,
            static_cast<double>(box.y1) + height * 0.5, width, height};
}

std::array<double, 4> scales(const Mean& mean) {
    const double width = std::max(1.0, mean[2]);
    const double height = std::max(1.0, mean[3]);
    return {width, height, width, height};
}

void initialize(Mean& mean, Covariance& covariance, const BoundingBox& box) {
    mean.fill(0.0);
    const auto observed = measurement(box);
    std::copy(observed.begin(), observed.end(), mean.begin());
    covariance.fill(0.0);
    const auto scale = scales(mean);
    // Position/size uncertainty starts at twice measurement noise, and velocity
    // at ten times its process noise. The constants are fixed before evaluation.
    // Each x/width component scales with width, each y/height with height; a
    // one-pixel floor keeps arbitrarily small positive input boxes well-defined.
    for (std::size_t i = 0; i < 4; ++i) {
        const double position_std = 2.0 * position_weight * scale[i];
        const double velocity_std = 10.0 * velocity_weight * scale[i];
        covariance[i * 8 + i] = position_std * position_std;
        covariance[(i + 4) * 8 + i + 4] = velocity_std * velocity_std;
    }
}

bool valid_filter(const Mean& mean, const Covariance& covariance) {
    for (double value : mean) {
        if (!std::isfinite(value) || std::abs(value) > 2.0 * max_coordinate) return false;
    }
    if (mean[2] <= 0.0 || mean[3] <= 0.0 ||
        mean[2] > max_coordinate || mean[3] > max_coordinate) return false;
    for (double value : covariance) {
        if (!std::isfinite(value) || std::abs(value) > 1e30) return false;
    }
    for (std::size_t i = 0; i < 8; ++i) {
        if (covariance[i * 8 + i] <= 0.0) return false;
    }
    return true;
}

bool predict(Mean& mean, Covariance& covariance) {
    const auto old = covariance;
    const auto scale = scales(mean);
    for (std::size_t i = 0; i < 4; ++i) mean[i] += mean[i + 4];
    // Width/height must remain positive even during a prolonged unmatched
    // shrinking observation. A boundary hit stops its size velocity only.
    for (std::size_t i : {std::size_t{2}, std::size_t{3}}) {
        if (mean[i] < 1e-3) { mean[i] = 1e-3; mean[i + 4] = 0.0; }
    }
    // F = [I I; 0 I]. Expand F P F^T directly instead of a general matrix API.
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            double value = old[i * 8 + j];
            if (i < 4) value += old[(i + 4) * 8 + j];
            if (j < 4) value += old[i * 8 + j + 4];
            if (i < 4 && j < 4) value += old[(i + 4) * 8 + j + 4];
            covariance[i * 8 + j] = value;
        }
    }
    for (std::size_t i = 0; i < 4; ++i) {
        const double position_std = position_weight * scale[i];
        const double velocity_std = velocity_weight * scale[i];
        covariance[i * 8 + i] += position_std * position_std;
        covariance[(i + 4) * 8 + i + 4] += velocity_std * velocity_std;
    }
    return valid_filter(mean, covariance);
}

BoundingBox predicted_box(const Mean& mean) {
    return {static_cast<float>(mean[0] - mean[2] * 0.5),
            static_cast<float>(mean[1] - mean[3] * 0.5),
            static_cast<float>(mean[0] + mean[2] * 0.5),
            static_cast<float>(mean[1] + mean[3] * 0.5)};
}

bool innovation(const Mean& mean, const Covariance& covariance,
                Matrix4& lower, std::array<double, 4>& noise) {
    const auto scale = scales(mean);
    Matrix4 projected{};
    for (std::size_t i = 0; i < 4; ++i) {
        const double standard_deviation = position_weight * scale[i];
        noise[i] = standard_deviation * standard_deviation;
        for (std::size_t j = 0; j < 4; ++j) projected[i * 4 + j] = covariance[i * 8 + j];
        projected[i * 4 + i] += noise[i];
    }
    lower.fill(0.0);
    // Cholesky factorization uses solves, never an explicitly formed inverse.
    // The positive measurement-noise floor prevents a singular valid matrix.
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = projected[i * 4 + j];
            for (std::size_t k = 0; k < j; ++k) value -= lower[i * 4 + k] * lower[j * 4 + k];
            if (i == j) {
                if (!std::isfinite(value) || value <= 1e-18) return false;
                lower[i * 4 + j] = std::sqrt(value);
            } else {
                lower[i * 4 + j] = value / lower[j * 4 + j];
                if (!std::isfinite(lower[i * 4 + j])) return false;
            }
        }
    }
    return true;
}

Measurement solve(const Matrix4& lower, Measurement right) {
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < i; ++j) right[i] -= lower[i * 4 + j] * right[j];
        right[i] /= lower[i * 4 + i];
    }
    for (std::size_t i = 4; i-- > 0;) {
        for (std::size_t j = i + 1; j < 4; ++j) right[i] -= lower[j * 4 + i] * right[j];
        right[i] /= lower[i * 4 + i];
    }
    return right;
}

double squared_distance(const Mean& mean, const Matrix4& lower, const Measurement& observed,
                        KalmanGateMode mode) {
    if (mode == KalmanGateMode::CenterOnly) {
        // The leading Cholesky block factors the marginal (cx,cy) innovation
        // covariance. Whiten only its two residuals: a full four-dimensional
        // solve with zero size residuals would instead use conditional center
        // uncertainty and is not this gate.
        const double x = (observed[0] - mean[0]) / lower[0];
        const double y = (observed[1] - mean[1] - lower[4] * x) / lower[5];
        return x * x + y * y;
    }
    Measurement residual{};
    for (std::size_t i = 0; i < 4; ++i) residual[i] = observed[i] - mean[i];
    const auto solution = solve(lower, residual);
    double distance = 0.0;
    for (std::size_t i = 0; i < 4; ++i) distance += residual[i] * solution[i];
    return distance;
}

bool correct(Mean& mean, Covariance& covariance, const Measurement& observed) {
    Matrix4 lower{};
    std::array<double, 4> noise{};
    if (!innovation(mean, covariance, lower, noise)) return false;
    std::array<double, 32> gain{};
    for (std::size_t i = 0; i < 8; ++i) {
        Measurement right{};
        for (std::size_t j = 0; j < 4; ++j) right[j] = covariance[i * 8 + j];
        const auto row = solve(lower, right);
        for (std::size_t j = 0; j < 4; ++j) gain[i * 4 + j] = row[j];
    }
    Measurement residual{};
    for (std::size_t i = 0; i < 4; ++i) residual[i] = observed[i] - mean[i];
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 4; ++j) mean[i] += gain[i * 4 + j] * residual[j];
    }
    // Joseph form: (I-KH) P (I-KH)^T + K R K^T preserves covariance symmetry
    // and positive semidefiniteness better than subtracting K S K^T.
    Covariance adjustment{}, temporary{}, next{};
    for (std::size_t i = 0; i < 8; ++i) {
        adjustment[i * 8 + i] = 1.0;
        for (std::size_t j = 0; j < 4; ++j) adjustment[i * 8 + j] -= gain[i * 4 + j];
    }
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            for (std::size_t k = 0; k < 8; ++k) temporary[i * 8 + j] += adjustment[i * 8 + k] * covariance[k * 8 + j];
        }
    }
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            for (std::size_t k = 0; k < 8; ++k) next[i * 8 + j] += temporary[i * 8 + k] * adjustment[j * 8 + k];
            for (std::size_t k = 0; k < 4; ++k) next[i * 8 + j] += gain[i * 4 + k] * noise[k] * gain[j * 4 + k];
        }
    }
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = i; j < 8; ++j) {
            const double symmetric = 0.5 * (next[i * 8 + j] + next[j * 8 + i]);
            covariance[i * 8 + j] = covariance[j * 8 + i] = symmetric;
        }
    }
    return valid_filter(mean, covariance);
}

std::uint32_t next_age(std::uint32_t age) {
    return age == std::numeric_limits<std::uint32_t>::max() ? age : age + 1;
}

void validate_detection(const Detection& detection) {
    if (detection.label.empty() || detection.label.size() > 256 ||
        detection.label.find('\0') != std::string::npos ||
        !std::isfinite(detection.score) || detection.score < 0.0F || detection.score > 1.0F) {
        throw std::invalid_argument("Invalid Kalman tracker detection label or score");
    }
    const auto& box = detection.bbox;
    for (float coordinate : {box.x1, box.y1, box.x2, box.y2}) {
        if (!std::isfinite(coordinate) || std::abs(static_cast<double>(coordinate)) > max_coordinate) {
            throw std::invalid_argument("Kalman tracker coordinates must be finite and within +/-1,000,000");
        }
    }
    const auto observed = measurement(box);
    if (observed[2] <= 0.0 || observed[3] <= 0.0 ||
        observed[2] > max_coordinate || observed[3] > max_coordinate) {
        throw std::invalid_argument("Kalman tracker box dimensions must be in (0,1,000,000]");
    }
}

std::vector<double> normalized_appearance(const std::vector<float>& embedding, std::size_t dimension) {
    if (embedding.size() != dimension) {
        throw std::invalid_argument("Kalman appearance embedding dimension mismatch");
    }
    std::vector<double> normalized(dimension);
    double squared_norm = 0.0;
    for (std::size_t i = 0; i < dimension; ++i) {
        const double value = embedding[i];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Kalman appearance embeddings must be finite");
        }
        normalized[i] = value;
        squared_norm += value * value;
    }
    // Finite float inputs with at most 1024 elements cannot overflow this
    // double-precision norm, including float subnormal/maximum-magnitude values.
    if (!std::isfinite(squared_norm) || squared_norm <= 0.0) {
        throw std::invalid_argument("Kalman appearance embeddings must have a nonzero norm");
    }
    const double norm = std::sqrt(squared_norm);
    for (auto& value : normalized) value /= norm;
    return normalized;
}

double cosine_similarity(const std::vector<double>& prototype, const std::vector<double>& observed) {
    double similarity = 0.0;
    for (std::size_t i = 0; i < prototype.size(); ++i) similarity += prototype[i] * observed[i];
    return std::clamp(similarity, -1.0, 1.0);
}

void update_appearance(std::vector<double>& prototype, const std::vector<double>& observed, double momentum) {
    double squared_norm = 0.0;
    for (std::size_t i = 0; i < prototype.size(); ++i) {
        prototype[i] = momentum * prototype[i] + (1.0 - momentum) * observed[i];
        squared_norm += prototype[i] * prototype[i];
    }
    // The active-gate ablation can admit antipodal features. At momentum .5
    // exact cancellation has no direction; retain the current observation.
    if (squared_norm == 0.0) { prototype = observed; return; }
    if (!std::isfinite(squared_norm)) {
        throw std::runtime_error("Kalman appearance prototype normalization failed");
    }
    const double norm = std::sqrt(squared_norm);
    for (auto& value : prototype) value /= norm;
}
}  // namespace

KalmanTracker::KalmanTracker(KalmanTrackerConfig config) : config_(config) {
    if ((config.relax_active_appearance || config.guard_active_appearance) && !config.use_appearance)
        throw std::invalid_argument("Active appearance ablation requires appearance");
    if (config.relax_active_appearance && config.guard_active_appearance)
        throw std::invalid_argument("Appearance ablation policies are mutually exclusive");
    for (float value : {config.low_threshold, config.high_threshold, config.new_track_threshold, config.match_iou}) {
        if (!std::isfinite(value) || value <= 0.0F || value > 1.0F) {
            throw std::invalid_argument("Kalman tracker thresholds must be finite and in (0,1]");
        }
    }
    if (config.low_threshold >= config.high_threshold || config.high_threshold > config.new_track_threshold ||
        config.max_missed_frames > 10000 || !std::isfinite(config.gating_threshold) ||
        config.gating_threshold <= 0.0 || config.gating_threshold > 1e6) {
        throw std::invalid_argument("Invalid Kalman tracker threshold order, lifetime or gating threshold");
    }
    if (config.gate_mode != KalmanGateMode::FullBox && config.gate_mode != KalmanGateMode::CenterOnly) {
        throw std::invalid_argument("Invalid Kalman tracker gate mode");
    }
    if (config.appearance_dimension == 0 || config.appearance_dimension > 1024 ||
        !std::isfinite(config.max_cosine_distance) || config.max_cosine_distance < 0.0 || config.max_cosine_distance > 1.0 ||
        !std::isfinite(config.appearance_weight) || config.appearance_weight < 0.0 || config.appearance_weight > 1.0 ||
        !std::isfinite(config.appearance_momentum) || config.appearance_momentum < 0.0 || config.appearance_momentum >= 1.0) {
        throw std::invalid_argument("Invalid Kalman appearance dimension, distance, weight or momentum");
    }
}

std::vector<Track> KalmanTracker::update(const std::vector<Detection>& detections) {
    if (detections.size() > max_detections) throw std::invalid_argument("Kalman tracker detection capacity exceeded");
    for (const auto& detection : detections) validate_detection(detection);
    std::vector<std::vector<double>> appearance;
    if (config_.use_appearance) {
        appearance.resize(detections.size());
        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].score >= config_.low_threshold) {
                appearance[i] = normalized_appearance(detections[i].embedding, config_.appearance_dimension);
            }
        }
    }

    // Work on bounded copies and commit only on success. Malformed observations,
    // assignment failure and allocation failure cannot partially age live state.
    auto working = states_;
    auto statistics = stats_;
    auto next_id = next_id_;
    std::vector<AssociationTrace> trace;
    std::vector<std::uint64_t> active, lost, all_ids;
    std::map<std::uint64_t, BoundingBox> predicted;
    std::map<std::uint64_t, Matrix4> factors;
    for (auto& [id, state] : working) {
        all_ids.push_back(id);
        (state.track.missed_frames == 0 ? active : lost).push_back(id);
        if (!predict(state.mean, state.covariance)) {
            initialize(state.mean, state.covariance, state.track.bbox);
            ++statistics.numerical_resets;
        }
        Matrix4 factor{};
        std::array<double, 4> noise{};
        if (!innovation(state.mean, state.covariance, factor, noise)) {
            initialize(state.mean, state.covariance, state.track.bbox);
            ++statistics.numerical_resets;
            if (!innovation(state.mean, state.covariance, factor, noise)) {
                throw std::runtime_error("Kalman innovation failed after a bounded reset");
            }
        }
        predicted.emplace(id, predicted_box(state.mean));
        factors.emplace(id, factor);
    }

    std::vector<std::size_t> high, low;
    // Count geometry-valid competition before any assignment. Include lost
    // states and low-score observations: stage ordering must not hide rivals.
    std::map<std::uint64_t, std::size_t> geometry_degree;
    std::vector<std::size_t> observation_degree(config_.guard_active_appearance ? detections.size() : 0);
    if (config_.guard_active_appearance) {
        for (const auto& [id, state] : working) {
            for (std::size_t j = 0; j < detections.size(); ++j) {
                const auto& d = detections[j];
                if (d.score < config_.low_threshold || d.label != state.track.label) continue;
                const auto overlap = predicted.at(id).iou(d.bbox);
                if (!std::isfinite(overlap) || overlap < config_.match_iou) continue;
                const auto distance = squared_distance(state.mean, factors.at(id), measurement(d.bbox), config_.gate_mode);
                if (!std::isfinite(distance) || distance < 0 || distance > config_.gating_threshold) continue;
                ++geometry_degree[id]; ++observation_degree[j];
            }
        }
    }
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].score >= config_.high_threshold) high.push_back(i);
        else if (detections[i].score >= config_.low_threshold) low.push_back(i);
    }
    const auto sort_scores = [&](auto& indices) {
        std::stable_sort(indices.begin(), indices.end(), [&](auto a, auto b) {
            return detections[a].score > detections[b].score;
        });
    };
    sort_scores(high); sort_scores(low);
    std::set<std::uint64_t> matched_ids;
    std::set<std::size_t> matched_detections;
    std::vector<Track> visible;
    const auto associate = [&](const std::vector<std::uint64_t>& ids,
                               const std::vector<std::size_t>& indices, bool low_stage, const char* stage) {
        std::vector<std::vector<double>> weights(ids.size(), std::vector<double>(indices.size(), -1.0));
        const auto trace_start = trace.size();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto& state = working.at(ids[i]);
            for (std::size_t j = 0; j < indices.size(); ++j) {
                const auto& detection = detections[indices[j]];
                AssociationTrace* event = nullptr;
                if (config_.trace_association) {
                    trace.push_back({ids[i], indices[j], stage, "class", {}, {}, {}, {}});
                    event = &trace.back();
                }
                if (state.track.label != detection.label) continue;
                const auto overlap = predicted.at(ids[i]).iou(detection.bbox);
                if (event) { event->outcome = "iou"; if (std::isfinite(overlap)) event->iou = overlap; }
                if (!std::isfinite(overlap) || overlap < config_.match_iou) continue;
                const double distance = squared_distance(state.mean, factors.at(ids[i]),
                                                         measurement(detection.bbox), config_.gate_mode);
                if (event) { event->outcome = "motion"; if (std::isfinite(distance)) event->motion_distance = distance; }
                if (!std::isfinite(distance) || distance < 0.0 || distance > config_.gating_threshold) {
                    ++statistics.gate_rejections;
                    continue;
                }
                if (config_.use_appearance) {
                    const double similarity = cosine_similarity(state.appearance, appearance[indices[j]]);
                    if (event) { event->outcome = "appearance"; if (std::isfinite(similarity)) event->cosine_distance = 1.0 - similarity; }
                    // Exact-match mode allows only rounding-sized norm error.
                    // Other hard-gated thresholds remain exact. The opt-in
                    // active ablation can admit negative cosine; nonpositive
                    // fused rewards are left unmatched by the solver.
                    const double tolerance = config_.max_cosine_distance == 0.0 ? 1e-12 : 0.0;
                    const bool guarded = config_.guard_active_appearance && !low_stage &&
                        state.track.missed_frames == 0 && detection.score >= config_.new_track_threshold &&
                        overlap >= guarded_appearance_iou && geometry_degree.at(ids[i]) == 1 &&
                        observation_degree[indices[j]] == 1;
                    const bool hard_gate = (!config_.relax_active_appearance || state.track.missed_frames > 0) && !guarded;
                    if (guarded && 1.0 - similarity > config_.max_cosine_distance + tolerance)
                        ++statistics.guarded_appearance_bypasses;
                    if (!std::isfinite(similarity) || (hard_gate &&
                        1.0 - similarity > config_.max_cosine_distance + tolerance)) {
                        ++statistics.appearance_rejections;
                        continue;
                    }
                    weights[i][j] = std::max(0.0, (1.0 - config_.appearance_weight) * overlap + config_.appearance_weight * similarity);
                } else {
                    weights[i][j] = overlap;
                }
                if (event) { event->outcome = "eligible"; event->reward = weights[i][j]; }
            }
        }
        for (const auto& [i, j] : assign_max_weight(weights)) {
            if (config_.trace_association) trace.at(trace_start + i * indices.size() + j).outcome = "matched";
            const auto id = ids[i];
            const auto index = indices[j];
            auto& state = working.at(id);
            const auto& detection = detections[index];
            if (!correct(state.mean, state.covariance, measurement(detection.bbox))) {
                initialize(state.mean, state.covariance, detection.bbox);
                ++statistics.numerical_resets;
            }
            if (config_.use_appearance) {
                ++statistics.appearance_matches;
                if (!low_stage) {
                    update_appearance(state.appearance, appearance[index], config_.appearance_momentum);
                    ++statistics.appearance_updates;
                }
            }
            if (low_stage) ++statistics.low_confidence_matches;
            if (state.track.missed_frames > 0) ++statistics.reactivations;
            state.track = {id, detection.bbox, detection.label, detection.score, next_age(state.track.age), 0};
            visible.push_back(state.track);
            matched_ids.insert(id);
            matched_detections.insert(index);
        }
    };
    associate(active, high, false, "active_high");
    std::vector<std::uint64_t> remaining_active;
    for (const auto id : active) if (!matched_ids.contains(id)) remaining_active.push_back(id);
    associate(remaining_active, low, true, "active_low");
    std::vector<std::size_t> remaining_high;
    for (const auto index : high) if (!matched_detections.contains(index)) remaining_high.push_back(index);
    associate(lost, remaining_high, false, "lost_high");

    for (const auto id : all_ids) {
        if (matched_ids.contains(id)) continue;
        auto& track = working.at(id).track;
        track.age = next_age(track.age);
        if (++track.missed_frames > config_.max_missed_frames) {
            working.erase(id);
            ++statistics.expired_tracks;
        }
    }
    for (const auto index : high) {
        const auto& detection = detections[index];
        if (matched_detections.contains(index) || detection.score < config_.new_track_threshold) continue;
        if (working.size() == max_tracks) { ++statistics.capacity_rejections; continue; }
        if (next_id == 0 || next_id == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Kalman tracker identity space exhausted");
        }
        State state;
        state.track = {next_id++, detection.bbox, detection.label, detection.score, 1, 0};
        if (config_.trace_association)
            trace.push_back({state.track.track_id, index, "birth", "created", {}, {}, {}, {}});
        initialize(state.mean, state.covariance, detection.bbox);
        if (config_.use_appearance) state.appearance = appearance[index];
        visible.push_back(state.track);
        working.emplace(state.track.track_id, std::move(state));
        ++statistics.created_tracks;
    }
    std::sort(visible.begin(), visible.end(), [](const auto& left, const auto& right) {
        return left.track_id < right.track_id;
    });
    states_ = std::move(working);
    stats_ = statistics;
    next_id_ = next_id;
    trace_ = std::move(trace);
    return visible;
}

}  // namespace aegisvision
