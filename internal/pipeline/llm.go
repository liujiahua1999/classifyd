package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// LLMConfig holds OpenAI-compatible API settings for character extraction from filenames.
type LLMConfig struct {
	APIKey     string
	BaseURL    string // e.g. https://xiaoai.plus/v1
	Model      string // e.g. gpt-4o-mini
	TimeoutSec int
	EncodeB64  bool // send filename as base64 to dodge content filters
}

// LoadAPIFile reads the two-line credential file: line 1 = API key, line 2 = base URL.
func LoadAPIFile(path string) (apiKey, baseURL string, err error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", "", err
	}
	var lines []string
	for _, line := range strings.Split(string(raw), "\n") {
		s := strings.TrimSpace(line)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		lines = append(lines, s)
	}
	if len(lines) < 2 {
		return "", "", fmt.Errorf("api file needs at least 2 lines (key, base url), got %d", len(lines))
	}
	return lines[0], lines[1], nil
}

// llmExtractCharacters calls the OpenAI-compatible chat API to extract character names.
func llmExtractCharacters(ctx context.Context, cfg *LLMConfig, basename, stem string) ([]string, error) {
	if cfg == nil || cfg.APIKey == "" || cfg.BaseURL == "" {
		return nil, nil
	}

	var system, user string
	if cfg.EncodeB64 {
		system = "You extract anime or game character names that a file label likely refers to. " +
			"The user message contains a Base64-encoded UTF-8 string. Decode it first, then read " +
			`the lines "File name (no path): ..." and "Stem only: ...". ` +
			"Use Danbooru-style tags: lowercase words joined by underscores, no spaces. " +
			`Reply with ONLY a JSON object: {"characters":["name_one","name_two"]}. ` +
			"Empty array if no recognizable character names."
		user = "Encoded file label (Base64, UTF-8):\n" + EncodeFilenameB64(basename, stem)
	} else {
		system = "You extract anime or game character names that the filename likely refers to. " +
			"Use Danbooru-style tags: lowercase words joined by underscores, no spaces. " +
			`Reply with ONLY a JSON object: {"characters":["name_one","name_two"]}. ` +
			"Empty array if no recognizable character names."
		user = "File name (no path): " + basename + "\nStem only: " + stem
	}

	timeout := time.Duration(cfg.TimeoutSec) * time.Second
	if timeout <= 0 {
		timeout = 45 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	payload := map[string]any{
		"model": cfg.Model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"temperature": 0.1,
		"max_tokens":  256,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		msg := string(respBody)
		if len(msg) > 500 {
			msg = msg[:500]
		}
		return nil, fmt.Errorf("llm HTTP %d: %s", resp.StatusCode, msg)
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("llm parse: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, nil
	}
	content := strings.TrimSpace(result.Choices[0].Message.Content)
	if content == "" {
		return nil, nil
	}

	content = stripJSONFence(content)

	var parsed struct {
		Characters []string `json:"characters"`
	}
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return nil, nil
	}

	seen := map[string]struct{}{}
	var out []string
	for _, name := range parsed.Characters {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		low := strings.ToLower(name)
		if _, ok := seen[low]; ok {
			continue
		}
		seen[low] = struct{}{}
		out = append(out, name)
	}
	return out, nil
}

func stripJSONFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	lines := strings.Split(s, "\n")
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "```" {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}
