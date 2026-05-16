# --- build stage ---
FROM --platform=linux/amd64 golang:1.26-alpine AS builder
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /out/api .

# --- runtime stage ---
# alpine (not scratch) so we have wget for the docker-compose healthcheck.
FROM --platform=linux/amd64 alpine:3.20
RUN apk add --no-cache wget
COPY --from=builder /out/api /usr/local/bin/api
ENTRYPOINT ["/usr/local/bin/api", "--resources-dir=/resources"]
