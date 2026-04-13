package config

import (
	"os"
	"strconv"
	"strings"
)

type Config struct {
	ListenAddr         string
	DataDir            string
	WorkerConcurrency  int
	FFprobeBin         string
	CharacterNamesFile string
	LLMAPIFile         string
	LLMModel           string
	LLMB64             bool
	// WD14 settings
	PythonBin      string
	TaggerScript   string // path to wd14_tagger.py; non-empty enables WD14 pipeline
	WD14Frames     int
	WD14MaxSide    int
}

func getenv(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func getenvInt(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func getenvBool(key string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func Load() Config {
	return Config{
		ListenAddr:         getenv("CLASSIFYD_LISTEN", ":8080"),
		DataDir:            getenv("CLASSIFYD_DATA", "./data"),
		WorkerConcurrency:  getenvInt("CLASSIFYD_WORKERS", 1),
		FFprobeBin:         getenv("FFPROBE_BIN", "ffprobe"),
		CharacterNamesFile: getenv("CHARACTER_NAMES_FILE", ""),
		LLMAPIFile:         getenv("LLM_API_FILE", ""),
		LLMModel:           getenv("LLM_MODEL", "gpt-4o-mini"),
		LLMB64:             getenvBool("LLM_B64_FILENAME"),
		PythonBin:          getenv("PYTHON_BIN", "python3"),
		TaggerScript:       getenv("WD14_TAGGER_SCRIPT", ""),
		WD14Frames:         getenvInt("WD14_FRAMES", 12),
		WD14MaxSide:        getenvInt("WD14_MAX_SIDE", 1024),
	}
}

func (c *Config) UseWD14() bool {
	return strings.TrimSpace(c.TaggerScript) != ""
}
