package config

import (
	"os"
	"strconv"
	"strings"
)

// Config holds daemon settings.
type Config struct {
	ListenAddr        string
	DataDir           string
	WorkerConcurrency int
	FFprobeBin        string
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

func Load() Config {
	return Config{
		ListenAddr:        getenv("CLASSIFYD_LISTEN", ":8080"),
		DataDir:           getenv("CLASSIFYD_DATA", "./data"),
		WorkerConcurrency: getenvInt("CLASSIFYD_WORKERS", 1),
		FFprobeBin:        getenv("FFPROBE_BIN", "ffprobe"),
	}
}
