package config

import (
	"os"
	"strconv"
	"strings"
)

type Config struct {
	ListenAddr        string
	DataDir           string
	WorkerConcurrency int
	FFprobeBin        string
	// WD14 (Python subprocess) settings. If PythonScript is set, the WD14 classifier is used.
	PythonBin       string
	PythonScript    string
	PythonExtraArgs []string
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
	c := Config{
		ListenAddr:        getenv("CLASSIFYD_LISTEN", ":8080"),
		DataDir:           getenv("CLASSIFYD_DATA", "./data"),
		WorkerConcurrency: getenvInt("CLASSIFYD_WORKERS", 1),
		FFprobeBin:        getenv("FFPROBE_BIN", "ffprobe"),
		PythonBin:         getenv("PYTHON_BIN", "python3"),
		PythonScript:      getenv("PYTHON_SCRIPT", ""),
	}
	if extra := strings.TrimSpace(os.Getenv("PYTHON_EXTRA_ARGS")); extra != "" {
		c.PythonExtraArgs = strings.Fields(extra)
	}
	return c
}

func (c *Config) UseWD14() bool {
	return strings.TrimSpace(c.PythonScript) != ""
}
