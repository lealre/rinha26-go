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
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /out/api .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /out/build-index ./cmd/build-index

# Train the IVFSQ index. Produces /out/index.bin (~84 MB). The full 168 MB raw
# dataset stays in this build stage and is dropped from the final image.
RUN /out/build-index --in /src/resources/references.json.gz --out /out/index.bin

# --- runtime stage ---
# alpine (not scratch) so we have wget for the docker-compose healthcheck.
FROM --platform=linux/amd64 alpine:3.20
RUN apk add --no-cache wget

COPY --from=builder /out/api /usr/local/bin/api
COPY --from=builder /out/index.bin /index.bin

# Small JSON config files live in /resources; docker-compose mounts the host
# directory on top of this path for local dev, but the image baked-in versions
# make it self-contained for submission.
COPY --from=builder /src/resources/mcc_risk.json /resources/mcc_risk.json
COPY --from=builder /src/resources/normalization.json /resources/normalization.json

ENTRYPOINT ["/usr/local/bin/api", "--resources-dir=/resources", "--index=/index.bin"]
