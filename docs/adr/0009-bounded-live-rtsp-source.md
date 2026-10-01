# ADR-0009: Bounded RTSP decode and isolated tracking sessions

Date: 2026-10-01. Status: accepted for local CPU CLI; not a production camera service.

File pipelines preserve every frame. Live sources cannot queue indefinitely when
CPU inference is slower than the camera. We use a single-thread-owned FFmpeg
VideoCapture with explicit open/read timeouts, feeding an owned, bounded,
drop-oldest queue. The consumer rejects stale decode-arrival frames. Both count
and pixel byte limits apply; decoder/model memory is outside those limits.

Capture is injectable for deterministic tests; the default adapter is explicitly
CAP_FFMPEG rather than a fallback backend that could ignore timeouts. Retry uses
bounded exponential backoff and a maximum outage. Stop joins the reader without
cross-thread capture.release(); one in-flight backend deadline can delay stop.

Pending frames are discarded on disconnect. The first decoded frame on a new
connection starts a new source session. Trackers reset on session change or a
large analyzed-arrival gap. IDs are epoch-scoped, not identities across cameras
or disconnects. Arrival time is monotonic, not camera PTS. Analyzed-frame MJPEG
is a diagnostic recording, not a faithful real-time recording across outages.

The existing durable HTTP queue handles finite local media jobs. RTSP is a separate
bounded CLI: do not replay infinite streams as file jobs, reuse file-derived
vector IDs, or promise PTS/seeking semantics that the adapter cannot provide.
Authentication/secrets, live web delivery, real camera timestamps, metric quality,
GStreamer and multi-camera lifecycle remain distinct future work.

Real local video/YOLO + MediaMTX tests verify recovered analysis and playable output.
Linux CI separately verifies real RTSP transport with a generated fixture and no
model; fake tests verify queue behavior and failure cleanup. None measures IDF1,
camera-to-result latency, or production availability.
