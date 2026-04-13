package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"
)

// WD14Classifier calls the Python anime_video_classifier.py in --stdout-json-one mode.
type WD14Classifier struct {
	PythonBin  string   // default "python3"
	ScriptPath string   // path to anime_video_classifier.py
	ExtraArgs  []string // forwarded flags (e.g. --frames, --wd14-model, --character-llm)
}

func (w *WD14Classifier) python() string {
	if w.PythonBin != "" {
		return w.PythonBin
	}
	return "python3"
}

// ClassifyVideo invokes the Python tagger and parses its JSON output.
func (w *WD14Classifier) ClassifyVideo(ctx context.Context, videoPath string) (*ClassifyResult, error) {
	args := []string{w.ScriptPath, "--stdout-json-one"}
	args = append(args, w.ExtraArgs...)
	args = append(args, videoPath)

	cmd := exec.CommandContext(ctx, w.python(), args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("wd14 python: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	var row wd14Row
	if err := json.Unmarshal(out, &row); err != nil {
		return nil, fmt.Errorf("wd14 parse: %w (first 500 bytes: %s)", err, truncate(string(out), 500))
	}
	if row.Error != "" {
		return nil, fmt.Errorf("wd14 tagger: %s", row.Error)
	}

	meta := &ProbeMeta{
		DurationSec:   row.DurationSec,
		VideoCodec:    ns(row.VideoCodec),
		Container:     ns(row.ContainerFormat),
		Tagger:        row.Tagger,
		FramesSampled: row.FramesSampled,
	}
	if row.Width != nil && row.Height != nil {
		meta.Width, meta.Height = row.Width, row.Height
	}

	if row.WD14 != nil {
		if s := row.WD14.Summary; s != nil {
			meta.DominantRating = ns(s.DominantRating)
			meta.GeneralTagString = s.GeneralTagString
			meta.CharacterTagString = s.CharacterTagString
		}
	}

	tokens := buildWD14Tokens(&row)
	sort.Strings(tokens)

	return &ClassifyResult{
		Raw:             json.RawMessage(out),
		FrequencyTokens: tokens,
		Meta:            meta,
	}, nil
}

// --- JSON shapes matching the Python script output ---

type wd14Row struct {
	VideoPath       string      `json:"video_path"`
	DurationSec     *float64    `json:"duration_sec"`
	ContainerFormat *string     `json:"container_format"`
	VideoCodec      *string     `json:"video_codec"`
	Width           *int        `json:"width"`
	Height          *int        `json:"height"`
	FramesSampled   int         `json:"frames_sampled"`
	Tagger          string      `json:"tagger"`
	WD14            *wd14Block  `json:"wd14"`
	Error           string      `json:"error,omitempty"`
}

type wd14Block struct {
	Model        string       `json:"model"`
	FramesTagged int          `json:"frames_tagged"`
	Summary      *wd14Summary `json:"summary"`
	Rating       map[string]float64 `json:"rating"`
	GeneralTags  []wd14Tag    `json:"general_tags"`
	CharacterTags []wd14Tag   `json:"character_tags"`
}

type wd14Summary struct {
	OutputMinScore     float64  `json:"output_min_score"`
	DominantRating     *string  `json:"dominant_rating_guess"`
	GeneralTagString   string   `json:"general_tag_string"`
	CharacterTagString string   `json:"character_tag_string"`
	CombinedHigh       []string `json:"combined_high_confidence"`
	CharacterFNFallback []string `json:"character_filename_fallback"`
	CharacterLLMNames  []string `json:"character_llm_names"`
	CharacterLLMModel  string   `json:"character_llm_model"`
}

type wd14Tag struct {
	Name   string  `json:"name"`
	Score  float64 `json:"score"`
	Frames int     `json:"frames,omitempty"`
	Source string  `json:"source,omitempty"`
}

func buildWD14Tokens(row *wd14Row) []string {
	set := map[string]struct{}{}
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s != "" {
			set[s] = struct{}{}
		}
	}

	if row.VideoCodec != nil {
		add("video_codec:" + strings.ToLower(*row.VideoCodec))
	}
	if row.ContainerFormat != nil {
		add("container:" + strings.ToLower(*row.ContainerFormat))
	}
	if row.Width != nil && row.Height != nil {
		add(fmt.Sprintf("resolution:%dx%d", *row.Width, *row.Height))
	}

	if row.WD14 != nil {
		for rk := range row.WD14.Rating {
			add("rating:" + strings.ToLower(rk))
		}
		for _, t := range row.WD14.GeneralTags {
			add("general:" + strings.ToLower(t.Name))
		}
		for _, t := range row.WD14.CharacterTags {
			tok := "character:" + strings.ToLower(t.Name)
			if t.Source != "" {
				tok += ":" + t.Source
			}
			add(tok)
		}
		if s := row.WD14.Summary; s != nil {
			if s.DominantRating != nil && *s.DominantRating != "" {
				add("dominant_rating:" + strings.ToLower(*s.DominantRating))
			}
		}
	}
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	return out
}

func ns(sp *string) string {
	if sp == nil {
		return ""
	}
	return *sp
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
