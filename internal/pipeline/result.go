package pipeline

import (
	"context"
	"encoding/json"
)

// ClassifyResult is the pipeline output for one video.
type ClassifyResult struct {
	Raw json.RawMessage
	// FrequencyTokens are deduplicated strings (per file) used for library-wide tag histograms.
	FrequencyTokens []string
	// Meta is optional structured fields stored on the job row for fast aggregates.
	Meta *ProbeMeta
}

// ProbeMeta is persisted on the job row for SQL aggregates (codecs, duration, resolution).
type ProbeMeta struct {
	DurationSec *float64
	VideoCodec  string
	AudioCodec  string
	Width       *int
	Height      *int
	Container   string
}

// VideoClassifier produces one result per video path (stored on the job).
type VideoClassifier interface {
	ClassifyVideo(ctx context.Context, videoPath string) (*ClassifyResult, error)
}
