# Multi-arch image: produces both linux/amd64 and linux/arm64 from a single
# `docker buildx build --platform linux/amd64,linux/arm64 --push` invocation.
#
# Trick: the slow 22-min index-training step (RUN /out/build-index ...) runs
# once on the BUILD host's native arch via `--platform=$BUILDPLATFORM`,
# avoiding QEMU emulation. The resulting /out/index.bin is then COPYed into
# both per-arch runtime images unchanged, since the binary file format is
# endianness-portable (explicit little-endian header; both amd64 and arm64
# Linux are LE, so the unsafe.Slice zero-copy reads in LoadIndex work on
# either). Only the api binary is cross-compiled per target architecture,
# which Go does in seconds with GOOS/GOARCH env vars.

# --- index training stage (runs on the BUILD host's native arch) ---
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS indexer
WORKDIR /src
COPY go.mod ./
COPY *.go ./
COPY internal/ ./internal/
COPY cmd/ ./cmd/
COPY resources/references.json.gz resources/mcc_risk.json resources/normalization.json ./resources/

RUN CGO_ENABLED=0 \
    go build -ldflags="-s -w" -trimpath -o /out/build-index ./cmd/build-index

# Train the IVFSQ index once. Produces /out/index.bin (~84 MB). The 168 MB
# raw dataset stays in this build stage and is dropped from the final image.
RUN /out/build-index --in /src/resources/references.json.gz --out /out/index.bin

# --- api binary build stage (cross-compiles per target arch, no QEMU) ---
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS apibuilder
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src
COPY go.mod ./
COPY *.go ./
COPY internal/ ./internal/
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -ldflags="-s -w" -trimpath -o /out/api .

# --- runtime stage (per target arch) ---
# alpine (not scratch) so we have wget for the docker-compose healthcheck.
FROM alpine:3.20
RUN apk add --no-cache wget

COPY --from=apibuilder /out/api /usr/local/bin/api
COPY --from=indexer /out/index.bin /index.bin

# Small JSON config files live in /resources; docker-compose may mount the
# host directory on top of this path for local dev, but the image baked-in
# versions make it self-contained for submission.
COPY --from=indexer /src/resources/mcc_risk.json /resources/mcc_risk.json
COPY --from=indexer /src/resources/normalization.json /resources/normalization.json

ENTRYPOINT ["/usr/local/bin/api", "--resources-dir=/resources", "--index=/index.bin"]
