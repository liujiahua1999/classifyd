FROM golang:1.22-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /out/classifyd ./cmd/classifyd

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/classifyd /usr/local/bin/classifyd
ENV CLASSIFYD_LISTEN=:8080 CLASSIFYD_DATA=/data
EXPOSE 8080
VOLUME ["/data"]
CMD ["classifyd"]
