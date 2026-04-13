package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"classifyd/internal/config"
	"classifyd/internal/pipeline"
	"classifyd/internal/server"
	"classifyd/internal/store"
	"classifyd/internal/worker"
)

func main() {
	cfg := config.Load()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.SetOutput(os.Stderr)

	st, err := store.Open(cfg.DataDir)
	if err != nil {
		log.Fatal(err)
	}
	defer st.Close()

	charNames := loadCharacterNames(cfg.CharacterNamesFile)
	llmCfg := loadLLMConfig(cfg)

	taggerScript := cfg.TaggerScript
	if taggerScript == "" {
		taggerScript = autoDiscover("wd14_tagger.py")
	}

	var classifier pipeline.VideoClassifier
	if taggerScript != "" {
		log.Printf("pipeline: WD14 (wd14_tagger.py=%s, frames=%d)", taggerScript, cfg.WD14Frames)
		classifier = &pipeline.WD14Classifier{
			FFprobe:        cfg.FFprobeBin,
			PythonBin:      cfg.PythonBin,
			TaggerScript:   taggerScript,
			Frames:         cfg.WD14Frames,
			MaxSide:        cfg.WD14MaxSide,
			CharacterNames: charNames,
			LLM:            llmCfg,
		}
	} else {
		log.Printf("pipeline: ffprobe-only (place wd14_tagger.py in cwd or set WD14_TAGGER_SCRIPT for WD14)")
		classifier = &pipeline.FFProbeClassifier{
			FFprobe:        cfg.FFprobeBin,
			CharacterNames: charNames,
			LLM:            llmCfg,
		}
	}

	if len(charNames) > 0 {
		log.Printf("character names: %d loaded", len(charNames))
	}
	if llmCfg != nil {
		log.Printf("LLM fallback: model=%s base=%s b64=%v", llmCfg.Model, llmCfg.BaseURL, llmCfg.EncodeB64)
	}

	wctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	wp := &worker.Pool{Store: st, Classifier: classifier, N: cfg.WorkerConcurrency}
	go wp.Run(wctx)

	h := &server.Handler{Store: st}
	mux := http.NewServeMux()
	h.Register(mux)

	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 15 * time.Second,
	}
	errCh := make(chan error, 1)
	go func() {
		log.Printf("listening on http://%s", cfg.ListenAddr)
		errCh <- srv.ListenAndServe()
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	select {
	case <-sig:
		log.Printf("shutting down…")
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
		cancel()
		return
	}
	shctx, shcancel := context.WithTimeout(context.Background(), 10*time.Second)
	_ = srv.Shutdown(shctx)
	shcancel()
	cancel()
}

func autoDiscover(name string) string {
	if _, err := os.Stat(name); err == nil {
		abs, _ := filepath.Abs(name)
		return abs
	}
	candidate := filepath.Join(filepath.Dir(os.Args[0]), name)
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return ""
}

func loadCharacterNames(explicit string) []string {
	path := explicit
	if path == "" {
		path = autoDiscover("character_names.txt")
	}
	if path == "" {
		return nil
	}
	names, err := pipeline.LoadCharacterNames(path)
	if err != nil {
		log.Printf("warning: character_names.txt: %v", err)
		return nil
	}
	log.Printf("loaded %d character names from %s", len(names), path)
	return names
}

func loadLLMConfig(cfg config.Config) *pipeline.LLMConfig {
	path := cfg.LLMAPIFile
	if path == "" {
		path = autoDiscover("api")
	}
	if path == "" {
		return nil
	}
	key, base, err := pipeline.LoadAPIFile(path)
	if err != nil {
		log.Printf("warning: api file: %v", err)
		return nil
	}
	return &pipeline.LLMConfig{
		APIKey:    key,
		BaseURL:   base,
		Model:     cfg.LLMModel,
		EncodeB64: cfg.LLMB64,
	}
}
