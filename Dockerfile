# --- build stage ---
FROM --platform=linux/amd64 golang:1.26-alpine AS builder
WORKDIR /src

# Pull in module + source. Two Go binaries get built from this tree:
#   - the long-lived api (root package)
#   - the one-shot build-index CLI under cmd/build-index/
COPY go.mod ./
COPY *.go ./
COPY internal/ ./internal/
COPY cmd/ ./cmd/
COPY resources/references.json.gz resources/mcc_risk.json resources/normalization.json ./resources/

# Build both binaries.
# default.pgo (if present at module root) feeds Profile-Guided Optimization
# into the api build; the build script in Makefile / scripts/pgo.sh generates
# it by running k6 against a non-PGO build and capturing /debug/pprof/profile.
# When default.pgo is absent the build proceeds without PGO.
# GOAMD64=v3 is intentionally NOT set yet — it changes FMA rounding in the
# autovectorized scalar distance loop, which shifts IVF cluster selection and
# regressed detection. Safe to re-enable once Option C replaces that loop
# with hand-written AVX2 assembly.
RUN --mount=type=bind,source=.,target=/ctx,ro \
    if [ -s /ctx/default.pgo ]; then cp /ctx/default.pgo ./default.pgo; fi && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -pgo=auto -ldflags="-s -w" -trimpath -o /out/api .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /out/build-index ./cmd/build-index
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /out/lb ./cmd/lb

# Train the IVFSQ index. Produces /out/index.bin (~84 MB). The full 168 MB raw
# dataset stays in this build stage and is dropped from the final image.
RUN /out/build-index --in /src/resources/references.json.gz --out /out/index.bin

# --- runtime stage ---
# alpine (not scratch) so we have wget for the docker-compose healthcheck.
FROM --platform=linux/amd64 alpine:3.20
RUN apk add --no-cache wget

COPY --from=builder /out/api /usr/local/bin/api
COPY --from=builder /out/lb /usr/local/bin/lb
COPY --from=builder /out/index.bin /index.bin

# Small JSON config files live in /resources; docker-compose mounts the host
# directory on top of this path for local dev, but the image baked-in versions
# make it self-contained for submission.
COPY --from=builder /src/resources/mcc_risk.json /resources/mcc_risk.json
COPY --from=builder /src/resources/normalization.json /resources/normalization.json

# Go runtime tuning for the 1-CPU, 350 MB cgroup budget.
#   GOMAXPROCS=1     — only one P, matches docker cpus=0.45 (no scheduler thrash)
#   GOGC=off         — disable triggered GC; let GOMEMLIMIT be the only trigger
#   GOMEMLIMIT=140MiB — soft memory ceiling; GC runs only as we approach it.
# Net effect: index (~85 MB) sits resident, per-request churn never causes
# GC until we'd exceed 140MiB — which we don't in steady state.
ENV GOMAXPROCS=1 \
    GOGC=off \
    GOMEMLIMIT=140MiB

ENTRYPOINT ["/usr/local/bin/api", "--resources-dir=/resources", "--index=/index.bin"]
