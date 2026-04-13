package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
)

// VideoClassifier produces one JSON result per video path (stored on the job).
type VideoClassifier interface {
	ClassifyVideo(ctx context.Context, videoPath string) (json.RawMessage, error)
}

// FFProbeClassifier stores ffprobe JSON for each file (no ML tagging; extend this package for ONNX, etc.).
type FFProbeClassifier struct {
	FFprobe string
}

func (f *FFProbeClassifier) bin() string {
	if f.FFprobe != "" {
		return f.FFprobe
	}
	return "ffprobe"
}

// ClassifyVideo runs ffprobe and returns a small wrapper JSON with stream/format metadata.
func (f *FFProbeClassifier) ClassifyVideo(ctx context.Context, videoPath string) (json.RawMessage, error) {
	path := filepath.Clean(videoPath)
	cmd := exec.CommandContext(ctx, f.bin(),
		"-v", "error", "-print_format", "json", "-show_format", "-show_streams", path)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe: %w: %s", err, string(bytes.TrimSpace(stderr.Bytes())))
	}
	var probe any
	if err := json.Unmarshal(out, &probe); err != nil {
		return nil, err
	}
	wrap := map[string]any{
		"video_path": path,
		"tagger":     "ffprobe-metadata",
		"probe":      probe,
	}
	return json.Marshal(wrap)
}
