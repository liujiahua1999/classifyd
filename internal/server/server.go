package server

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
	"strconv"
	"strings"

	"classifyd/internal/store"
)

//go:embed static/*
var static embed.FS

type Handler struct {
	Store *store.Store
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("/health", h.health)
	mux.HandleFunc("/api/v1/stats", h.stats)
	mux.HandleFunc("/api/v1/jobs", h.jobs)
	mux.HandleFunc("/api/v1/jobs/", h.jobByID)
	mux.HandleFunc("/api/v1/scan", h.scan)
	mux.HandleFunc("/api/v1/library/roots", h.roots)
	mux.HandleFunc("/api/v1/retry", h.retry)
	mux.HandleFunc("/api/v1/purge", h.purge)

	sub, err := fs.Sub(static, "static")
	if err != nil {
		panic(err)
	}
	mux.Handle("/", http.FileServer(http.FS(sub)))
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"ok":true}`))
}

func (h *Handler) stats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	st, err := h.Store.Stats(r.Context())
	if err != nil {
		writeErr(w, err, 500)
		return
	}
	writeJSON(w, st)
}

func (h *Handler) jobs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	status := q.Get("status")
	limit, _ := strconv.Atoi(q.Get("limit"))
	offset, _ := strconv.Atoi(q.Get("offset"))
	if limit == 0 {
		limit = 100
	}
	list, err := h.Store.ListJobs(r.Context(), status, limit, offset)
	if err != nil {
		writeErr(w, err, 500)
		return
	}
	writeJSON(w, list)
}

func (h *Handler) jobByID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/jobs/")
	id = strings.Trim(id, "/")
	if id == "" {
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodGet:
		j, err := h.Store.GetJob(r.Context(), id)
		if err != nil {
			writeErr(w, err, 500)
			return
		}
		if j == nil {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, j)

	case http.MethodDelete:
		if err := h.Store.DeleteJob(r.Context(), id); err != nil {
			writeErr(w, err, 500)
			return
		}
		writeJSON(w, map[string]any{"deleted": id})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

type scanReq struct {
	Root string `json:"root"`
}

func (h *Handler) scan(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req scanReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, err, 400)
		return
	}
	req.Root = strings.TrimSpace(req.Root)
	if req.Root == "" {
		http.Error(w, `{"error":"root required"}`, 400)
		return
	}
	added, skipped, err := h.Store.ScanRoot(req.Root)
	if err != nil {
		writeErr(w, err, 500)
		return
	}
	_ = h.Store.AddLibraryRoot(req.Root)
	writeJSON(w, map[string]any{
		"root":    req.Root,
		"added":   added,
		"skipped": skipped,
	})
}

func (h *Handler) roots(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		list, err := h.Store.ListLibraryRoots()
		if err != nil {
			writeErr(w, err, 500)
			return
		}
		writeJSON(w, list)

	case http.MethodDelete:
		var req struct {
			Path string `json:"path"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, err, 400)
			return
		}
		if strings.TrimSpace(req.Path) == "" {
			http.Error(w, `{"error":"path required"}`, 400)
			return
		}
		if err := h.Store.RemoveLibraryRoot(req.Path); err != nil {
			writeErr(w, err, 500)
			return
		}
		writeJSON(w, map[string]any{"removed": req.Path})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

type retryReq struct {
	ID string `json:"id"`
}

func (h *Handler) retry(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req retryReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, err, 400)
		return
	}
	if strings.TrimSpace(req.ID) == "all" || strings.TrimSpace(req.ID) == "" {
		n, err := h.Store.RetryAllFailed(r.Context())
		if err != nil {
			writeErr(w, err, 500)
			return
		}
		writeJSON(w, map[string]any{"retried": n})
		return
	}
	if err := h.Store.RetryJob(r.Context(), req.ID); err != nil {
		writeErr(w, err, 500)
		return
	}
	writeJSON(w, map[string]any{"retried": req.ID})
}

type purgeReq struct {
	Status string `json:"status"`
}

func (h *Handler) purge(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req purgeReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, err, 400)
		return
	}
	st := store.JobStatus(strings.TrimSpace(strings.ToLower(req.Status)))
	if st != store.StatusDone && st != store.StatusFailed {
		http.Error(w, `{"error":"can only purge done or failed"}`, 400)
		return
	}
	n, err := h.Store.PurgeByStatus(r.Context(), st)
	if err != nil {
		writeErr(w, err, 500)
		return
	}
	writeJSON(w, map[string]any{"purged": n, "status": string(st)})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeErr(w http.ResponseWriter, err error, code int) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}
