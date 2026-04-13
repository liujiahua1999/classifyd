package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// WD14Classifier extracts frames via ffmpeg, runs WD14 ONNX inference via
// a minimal Python helper (wd14_tagger.py), then applies character identification.
type WD14Classifier struct {
	FFprobe        string
	PythonBin      string   // default "python3"
	TaggerScript   string   // path to wd14_tagger.py
	Frames         int      // frames to extract per video (default 12)
	MaxSide        int      // max dimension for extracted frames (default 1024)
	CharacterNames []string // from character_names.txt
	LLM            *LLMConfig
}

func (w *WD14Classifier) python() string {
	if w.PythonBin != "" {
		return w.PythonBin
	}
	return "python3"
}

func (w *WD14Classifier) ffprobe() string {
	if w.FFprobe != "" {
		return w.FFprobe
	}
	return "ffprobe"
}

func (w *WD14Classifier) nFrames() int {
	if w.Frames > 0 {
		return w.Frames
	}
	return 12
}

func (w *WD14Classifier) maxSide() int {
	if w.MaxSide > 0 {
		return w.MaxSide
	}
	return 1024
}

func (w *WD14Classifier) ClassifyVideo(ctx context.Context, videoPath string) (*ClassifyResult, error) {
	path := filepath.Clean(videoPath)

	// Step 1: ffprobe for metadata
	probeCmd := exec.CommandContext(ctx, w.ffprobe(),
		"-v", "error", "-print_format", "json", "-show_format", "-show_streams", path)
	var probeSE bytes.Buffer
	probeCmd.Stderr = &probeSE
	probeOut, err := probeCmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe: %w: %s", err, strings.TrimSpace(probeSE.String()))
	}
	var root ffprobeRoot
	if err := json.Unmarshal(probeOut, &root); err != nil {
		return nil, err
	}
	stats, meta := summarizeProbe(&root)

	// Step 2: extract frames via ffmpeg
	tmpDir, err := os.MkdirTemp("", "wd14_frames_")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	dur := 60.0
	if stats.DurationSec != nil && *stats.DurationSec > 0 {
		dur = *stats.DurationSec
	}
	n := w.nFrames()
	interval := dur / float64(n)
	if interval < 0.01 {
		interval = 0.01
	}
	pattern := filepath.Join(tmpDir, "f_%04d.jpg")
	ms := w.maxSide()
	scale := fmt.Sprintf("scale='min(%d,iw)':'-2':flags=lanczos", ms)

	ffmpegCmd := exec.CommandContext(ctx, "ffmpeg",
		"-hide_banner", "-loglevel", "error", "-y", "-i", path,
		"-vf", fmt.Sprintf("fps=1/%f,%s", interval, scale),
		"-frames:v", strconv.Itoa(n), "-q:v", "4", pattern)
	var ffmpegSE bytes.Buffer
	ffmpegCmd.Stderr = &ffmpegSE
	if err := ffmpegCmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %w: %s", err, strings.TrimSpace(ffmpegSE.String()))
	}

	frames, _ := filepath.Glob(filepath.Join(tmpDir, "f_*.jpg"))
	sort.Strings(frames)
	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames extracted from %s", path)
	}
	meta.FramesSampled = len(frames)

	// Step 3: WD14 ONNX inference via Python helper
	frameList := strings.Join(frames, "\n") + "\n"
	taggerCmd := exec.CommandContext(ctx, w.python(), w.TaggerScript)
	taggerCmd.Stdin = strings.NewReader(frameList)
	var taggerSE bytes.Buffer
	taggerCmd.Stderr = &taggerSE
	taggerOut, err := taggerCmd.Output()
	if err != nil {
		return nil, fmt.Errorf("wd14_tagger.py: %w: %s", err, strings.TrimSpace(taggerSE.String()))
	}

	var wd14 wd14Result
	if err := json.Unmarshal(taggerOut, &wd14); err != nil {
		return nil, fmt.Errorf("wd14 parse: %w (output: %s)", err, truncStr(string(taggerOut), 500))
	}
	if wd14.Error != "" {
		return nil, fmt.Errorf("wd14: %s", wd14.Error)
	}

	// Populate meta from WD14 result
	meta.Tagger = "wd14:" + wd14.Model
	meta.DominantRating = wd14.DominantRating
	meta.GeneralTagString = wd14.GeneralTagString
	meta.CharacterTagString = wd14.CharacterTagString

	// Step 4: character fallback — if WD14 found no characters, try filename + LLM
	var charTags []CharacterTag
	for _, ct := range wd14.CharacterTags {
		charTags = append(charTags, CharacterTag{Name: ct.Name, Score: ct.Score, Source: "wd14"})
	}

	basename := filepath.Base(path)
	stem := strings.TrimSuffix(basename, filepath.Ext(basename))

	if len(charTags) == 0 {
		if matched := MatchCharactersInFilename(stem, w.CharacterNames); len(matched) > 0 {
			for _, nm := range matched {
				charTags = append(charTags, CharacterTag{Name: nm, Score: 1.0, Source: "filename"})
			}
			meta.Tagger += "+filename"
		}
	}
	if len(charTags) == 0 && w.LLM != nil {
		if names, err := llmExtractCharacters(ctx, w.LLM, basename, stem); err == nil && len(names) > 0 {
			for _, nm := range names {
				charTags = append(charTags, CharacterTag{Name: nm, Score: 0.95, Source: "llm"})
			}
			meta.Tagger += "+llm"
		}
	}
	if len(charTags) > 0 {
		meta.CharacterTagString = danbooru(charTags)
	}

	// Build frequency tokens
	tokens := buildWD14FreqTokens(stats, &wd14, charTags)
	sort.Strings(tokens)
	tokens = dedupeSorted(tokens)

	// Build result JSON
	wrap := map[string]any{
		"video_path":           path,
		"tagger":               meta.Tagger,
		"video_stats":          stats,
		"wd14":                 wd14,
		"character_tags":       charTags,
		"frequency_tokens":     tokens,
	}
	raw, err := json.Marshal(wrap)
	if err != nil {
		return nil, err
	}
	return &ClassifyResult{
		Raw:             raw,
		FrequencyTokens: tokens,
		Meta:            meta,
	}, nil
}

// wd14Result is the JSON output from wd14_tagger.py.
type wd14Result struct {
	Model              string             `json:"model"`
	FramesTagged       int                `json:"frames_tagged"`
	DominantRating     string             `json:"dominant_rating"`
	Rating             map[string]float64 `json:"rating"`
	GeneralTagString   string             `json:"general_tag_string"`
	CharacterTagString string             `json:"character_tag_string"`
	GeneralTags        []wd14Tag          `json:"general_tags"`
	CharacterTags      []wd14Tag          `json:"character_tags"`
	Error              string             `json:"error,omitempty"`
}

type wd14Tag struct {
	Name   string  `json:"name"`
	Score  float64 `json:"score"`
	Frames int     `json:"frames,omitempty"`
}

func buildWD14FreqTokens(vs videoStats, wd *wd14Result, chars []CharacterTag) []string {
	set := map[string]struct{}{}
	add := func(s string) {
		if s = strings.TrimSpace(s); s != "" {
			set[s] = struct{}{}
		}
	}
	if vs.VideoCodec != "" {
		add("video_codec:" + strings.ToLower(vs.VideoCodec))
	}
	if vs.AudioCodec != "" {
		add("audio_codec:" + strings.ToLower(vs.AudioCodec))
	}
	if vs.FormatName != "" {
		add("container:" + strings.ToLower(firstFormatToken(vs.FormatName)))
	}
	if vs.Width != nil && vs.Height != nil {
		add(fmt.Sprintf("resolution:%dx%d", *vs.Width, *vs.Height))
	}
	for rk := range wd.Rating {
		add("rating:" + strings.ToLower(rk))
	}
	if wd.DominantRating != "" {
		add("dominant_rating:" + strings.ToLower(wd.DominantRating))
	}
	for _, t := range wd.GeneralTags {
		add("general:" + strings.ToLower(t.Name))
	}
	for _, ct := range chars {
		add("character:" + strings.ToLower(strings.ReplaceAll(ct.Name, " ", "_")))
	}
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	return out
}

func truncStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
