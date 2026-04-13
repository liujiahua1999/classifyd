FROM golang:1.22-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/classifyd ./cmd/classifyd

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "dghs-imgutils[onnxruntime]" onnxruntime
COPY --from=build /out/classifyd /usr/local/bin/classifyd
COPY wd14_tagger.py /usr/local/bin/wd14_tagger.py
ENV CLASSIFYD_LISTEN=:8080 CLASSIFYD_DATA=/data PYTHON_BIN=/opt/venv/bin/python3
EXPOSE 8080
VOLUME ["/data"]
CMD ["classifyd"]
