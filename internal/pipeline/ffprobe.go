package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// FFProbeClassifier stores ffprobe JSON and derived metadata for each file.
type FFProbeClassifier struct {
	FFprobe string
}

func (f *FFProbeClassifier) bin() string {
	if f.FFprobe != "" {
		return f.FFprobe
	}
	return "ffprobe"
}

// ClassifyVideo runs ffprobe and returns wrapper JSON with stats, metadata tags, and frequency tokens.
func (f *FFProbeClassifier) ClassifyVideo(ctx context.Context, videoPath string) (*ClassifyResult, error) {
	path := filepath.Clean(videoPath)
	cmd := exec.CommandContext(ctx, f.bin(),
		"-v", "error", "-print_format", "json", "-show_format", "-show_streams", path)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe: %w: %s", err, string(bytes.TrimSpace(stderr.Bytes())))
	}
	var root ffprobeRoot
	if err := json.Unmarshal(out, &root); err != nil {
		return nil, err
	}
	stats, meta := summarizeProbe(&root)
	metaTags := collectMetadataTags(&root)
	tokens := buildFrequencyTokens(&root, stats, metaTags)
	sort.Strings(tokens)
	tokens = dedupeSorted(tokens)

	wrap := map[string]any{
		"video_path":         path,
		"tagger":             "ffprobe-metadata",
		"video_stats":        stats,
		"metadata_tags":      metaTags,
		"frequency_tokens":   tokens,
		"probe":              json.RawMessage(out),
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

// --- ffprobe JSON shapes ---

type ffprobeRoot struct {
	Streams []ffStream `json:"streams"`
	Format  ffFormat   `json:"format"`
}

type ffFormat struct {
	Filename   string            `json:"filename"`
	FormatName string            `json:"format_name"`
	Duration   string            `json:"duration"`
	Size       string            `json:"size"`
	BitRate    string            `json:"bit_rate"`
	Tags       map[string]string `json:"tags"`
}

type ffStream struct {
	Index         int               `json:"index"`
	CodecType     string            `json:"codec_type"`
	CodecName     string            `json:"codec_name"`
	Width         int               `json:"width"`
	Height        int               `json:"height"`
	PixFmt        string            `json:"pix_fmt"`
	AvgFrameRate  string            `json:"avg_frame_rate"`
	RFrameRate    string            `json:"r_frame_rate"`
	BitRate       string            `json:"bit_rate"`
	Tags          map[string]string `json:"tags"`
	Disposition   map[string]int    `json:"disposition"`
}

type videoStats struct {
	DurationSec *float64 `json:"duration_sec,omitempty"`
	SizeBytes   *int64   `json:"size_bytes,omitempty"`
	BitRate     *int64   `json:"bit_rate,omitempty"`
	FormatName  string   `json:"format_name,omitempty"`
	NumStreams  int      `json:"num_streams"`
	VideoCodec  string   `json:"video_codec,omitempty"`
	AudioCodec  string   `json:"audio_codec,omitempty"`
	Width       *int     `json:"width,omitempty"`
	Height      *int     `json:"height,omitempty"`
	PixFmt      string   `json:"pix_fmt,omitempty"`
	AvgFPS      *float64 `json:"avg_fps,omitempty"`
}

type metadataTag struct {
	Scope string `json:"scope"` // format | video | audio | subtitle | stream:N
	Key   string `json:"key"`
	Value string `json:"value"`
}

func summarizeProbe(root *ffprobeRoot) (videoStats, *ProbeMeta) {
	var vs videoStats
	vs.NumStreams = len(root.Streams)
	if root.Format.FormatName != "" {
		vs.FormatName = root.Format.FormatName
	}
	if d, ok := parseFloat(root.Format.Duration); ok {
		vs.DurationSec = &d
	}
	if sz, ok := parseInt64(root.Format.Size); ok {
		vs.SizeBytes = &sz
	}
	if br, ok := parseInt64(root.Format.BitRate); ok {
		vs.BitRate = &br
	}

	var vStream, aStream *ffStream
	for i := range root.Streams {
		s := &root.Streams[i]
		switch strings.ToLower(s.CodecType) {
		case "video":
			if vStream == nil {
				vStream = s
			}
		case "audio":
			if aStream == nil {
				aStream = s
			}
		}
	}
	meta := &ProbeMeta{Container: firstFormatToken(root.Format.FormatName)}
	if vStream != nil {
		vs.VideoCodec = vStream.CodecName
		if vStream.Width > 0 && vStream.Height > 0 {
			w, h := vStream.Width, vStream.Height
			vs.Width, vs.Height = &w, &h
		}
		if vStream.PixFmt != "" {
			vs.PixFmt = vStream.PixFmt
		}
		if fps, ok := parseFPS(vStream.AvgFrameRate); ok {
			vs.AvgFPS = &fps
		}
		meta.VideoCodec = vStream.CodecName
		meta.Width, meta.Height = vs.Width, vs.Height
	}
	if aStream != nil {
		vs.AudioCodec = aStream.CodecName
		meta.AudioCodec = aStream.CodecName
	}
	if vs.DurationSec != nil {
		meta.DurationSec = vs.DurationSec
	}

	return vs, meta
}

func collectMetadataTags(root *ffprobeRoot) []metadataTag {
	var out []metadataTag
	add := func(scope, k, v string) {
		k = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		if k == "" || v == "" {
			return
		}
		out = append(out, metadataTag{Scope: scope, Key: k, Value: v})
	}
	for k, v := range root.Format.Tags {
		add("format", k, v)
	}
	for i := range root.Streams {
		s := &root.Streams[i]
		scope := "stream:" + strconv.Itoa(s.Index)
		switch strings.ToLower(s.CodecType) {
		case "video":
			scope = "video"
		case "audio":
			scope = "audio"
		case "subtitle":
			scope = "subtitle"
		}
		for k, v := range s.Tags {
			add(scope, k, v)
		}
	}
	return out
}

func buildFrequencyTokens(root *ffprobeRoot, vs videoStats, metaTags []metadataTag) []string {
	set := map[string]struct{}{}
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		set[s] = struct{}{}
	}

	if vs.VideoCodec != "" {
		add("video_codec:" + strings.ToLower(vs.VideoCodec))
	}
	if vs.AudioCodec != "" {
		add("audio_codec:" + strings.ToLower(vs.AudioCodec))
	}
	if vs.PixFmt != "" {
		add("pix_fmt:" + strings.ToLower(vs.PixFmt))
	}
	if vs.FormatName != "" {
		add("container:" + strings.ToLower(firstFormatToken(vs.FormatName)))
	}
	if vs.AvgFPS != nil && *vs.AvgFPS > 0 {
		add(fmt.Sprintf("fps:%.3f", *vs.AvgFPS))
	}
	if vs.Width != nil && vs.Height != nil {
		add(fmt.Sprintf("resolution:%dx%d", *vs.Width, *vs.Height))
	}

	for _, t := range metaTags {
		k := strings.ToLower(strings.TrimSpace(t.Key))
		v := strings.TrimSpace(t.Value)
		if k == "" || v == "" {
			continue
		}
		// Normalize long hex / binary-ish values to reduce cardinality noise.
		vn := v
		if len(vn) > 120 {
			vn = vn[:117] + "..."
		}
		add(fmt.Sprintf("%s:%s=%s", t.Scope, k, vn))
	}
	_ = root // reserved if we add more signals later
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	return out
}

func dedupeSorted(in []string) []string {
	if len(in) <= 1 {
		return in
	}
	out := in[:0]
	prev := ""
	for _, s := range in {
		if s == prev {
			continue
		}
		out = append(out, s)
		prev = s
	}
	return out
}

func firstFormatToken(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.IndexByte(s, ','); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}

func parseFloat(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

func parseInt64(s string) (int64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

func parseFPS(avg string) (float64, bool) {
	avg = strings.TrimSpace(avg)
	if avg == "" || avg == "0/0" {
		return 0, false
	}
	parts := strings.SplitN(avg, "/", 2)
	if len(parts) != 2 {
		return parseFloat(avg)
	}
	num, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	den, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err1 != nil || err2 != nil || den == 0 {
		return 0, false
	}
	return num / den, true
}
