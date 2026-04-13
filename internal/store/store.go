package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "modernc.org/sqlite"
)

type JobStatus string

const (
	StatusPending    JobStatus = "pending"
	StatusProcessing JobStatus = "processing"
	StatusDone       JobStatus = "done"
	StatusFailed     JobStatus = "failed"
)

type Job struct {
	ID         string    `json:"id"`
	Path       string    `json:"path"`
	Status     JobStatus `json:"status"`
	CreatedAt  time.Time `json:"created_at"`
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
	Error      string    `json:"error,omitempty"`
	ResultJSON string    `json:"result_json,omitempty"`
}

type Stats struct {
	Total      int64 `json:"total"`
	Pending    int64 `json:"pending"`
	Processing int64 `json:"processing"`
	Done       int64 `json:"done"`
	Failed     int64 `json:"failed"`
}

type Store struct {
	db *sql.DB
}

func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	dbPath := filepath.Join(dir, "classifyd.db")
	db, err := sql.Open("sqlite", dbPath+"?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) migrate() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS jobs (
			id TEXT PRIMARY KEY,
			path TEXT NOT NULL UNIQUE,
			status TEXT NOT NULL,
			created_at INTEGER NOT NULL,
			started_at INTEGER,
			finished_at INTEGER,
			err TEXT,
			result_json TEXT
		);`,
		`CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);`,
		`CREATE TABLE IF NOT EXISTS library_roots (
			path TEXT PRIMARY KEY,
			added_at INTEGER NOT NULL
		);`,
	}
	for _, q := range stmts {
		if _, err := s.db.Exec(q); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) AddLibraryRoot(path string) error {
	path = filepath.Clean(path)
	_, err := s.db.Exec(
		`INSERT OR REPLACE INTO library_roots(path, added_at) VALUES(?, ?)`,
		path, time.Now().Unix(),
	)
	return err
}

func (s *Store) RemoveLibraryRoot(path string) error {
	path = filepath.Clean(path)
	_, err := s.db.Exec(`DELETE FROM library_roots WHERE path = ?`, path)
	return err
}

func (s *Store) ListLibraryRoots() ([]string, error) {
	rows, err := s.db.Query(`SELECT path FROM library_roots ORDER BY path`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]string, 0)
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *Store) EnqueueVideo(absPath string) (string, bool, error) {
	absPath = filepath.Clean(absPath)
	var existing string
	err := s.db.QueryRow(`SELECT id FROM jobs WHERE path = ?`, absPath).Scan(&existing)
	if err == nil {
		return existing, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return "", false, err
	}
	id := uuid.NewString()
	_, err = s.db.Exec(
		`INSERT INTO jobs(id, path, status, created_at) VALUES(?, ?, ?, ?)`,
		id, absPath, string(StatusPending), time.Now().Unix(),
	)
	if err != nil {
		return "", false, err
	}
	return id, true, nil
}

func (s *Store) ClaimNext(ctx context.Context) (*Job, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()

	var j Job
	var created, started, finished sql.NullInt64
	var errMsg, res sql.NullString
	err = tx.QueryRowContext(ctx,
		`SELECT id, path, status, created_at, started_at, finished_at, err, result_json
		 FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1`,
		string(StatusPending),
	).Scan(&j.ID, &j.Path, &j.Status, &created, &started, &finished, &errMsg, &res)
	if errMsg.Valid {
		j.Error = errMsg.String
	}
	if res.Valid {
		j.ResultJSON = res.String
	}
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	j.CreatedAt = time.Unix(created.Int64, 0)
	if started.Valid {
		t := time.Unix(started.Int64, 0)
		j.StartedAt = &t
	}
	if finished.Valid {
		t := time.Unix(finished.Int64, 0)
		j.FinishedAt = &t
	}

	now := time.Now().Unix()
	_, err = tx.ExecContext(ctx,
		`UPDATE jobs SET status = ?, started_at = ? WHERE id = ?`,
		string(StatusProcessing), now, j.ID,
	)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	j.Status = StatusProcessing
	t := time.Unix(now, 0)
	j.StartedAt = &t
	return &j, nil
}

func (s *Store) MarkDone(ctx context.Context, id string, resultJSON string) error {
	now := time.Now().Unix()
	_, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET status = ?, finished_at = ?, result_json = ?, err = '' WHERE id = ?`,
		string(StatusDone), now, resultJSON, id,
	)
	return err
}

func (s *Store) MarkFailed(ctx context.Context, id string, errMsg string) error {
	now := time.Now().Unix()
	_, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET status = ?, finished_at = ?, err = ? WHERE id = ?`,
		string(StatusFailed), now, errMsg, id,
	)
	return err
}

func (s *Store) RetryAllFailed(ctx context.Context) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET status = ?, started_at = NULL, finished_at = NULL, err = '' WHERE status = ?`,
		string(StatusPending), string(StatusFailed),
	)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

func (s *Store) RetryJob(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET status = ?, started_at = NULL, finished_at = NULL, err = '' WHERE id = ? AND status = ?`,
		string(StatusPending), id, string(StatusFailed),
	)
	return err
}

func (s *Store) DeleteJob(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM jobs WHERE id = ?`, id)
	return err
}

func (s *Store) PurgeByStatus(ctx context.Context, status JobStatus) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`DELETE FROM jobs WHERE status = ?`, string(status),
	)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

func (s *Store) Stats(ctx context.Context) (Stats, error) {
	var st Stats
	q := `SELECT status, COUNT(*) FROM jobs GROUP BY status`
	rows, err := s.db.QueryContext(ctx, q)
	if err != nil {
		return st, err
	}
	defer rows.Close()
	for rows.Next() {
		var status string
		var n int64
		if err := rows.Scan(&status, &n); err != nil {
			return st, err
		}
		switch JobStatus(status) {
		case StatusPending:
			st.Pending = n
		case StatusProcessing:
			st.Processing = n
		case StatusDone:
			st.Done = n
		case StatusFailed:
			st.Failed = n
		}
		st.Total += n
	}
	return st, rows.Err()
}

func (s *Store) ListJobs(ctx context.Context, status string, limit, offset int) ([]Job, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	var rows *sql.Rows
	var err error
	status = strings.TrimSpace(strings.ToLower(status))
	if status != "" && status != "all" {
		rows, err = s.db.QueryContext(ctx,
			`SELECT id, path, status, created_at, started_at, finished_at, err, result_json
			 FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?`,
			status, limit, offset)
	} else {
		rows, err = s.db.QueryContext(ctx,
			`SELECT id, path, status, created_at, started_at, finished_at, err, result_json
			 FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?`,
			limit, offset)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanJobs(rows)
}

func scanJobs(rows *sql.Rows) ([]Job, error) {
	list := make([]Job, 0)
	for rows.Next() {
		var j Job
		var created, started, finished sql.NullInt64
		var errMsg, res sql.NullString
		if err := rows.Scan(&j.ID, &j.Path, &j.Status, &created, &started, &finished, &errMsg, &res); err != nil {
			return nil, err
		}
		if errMsg.Valid {
			j.Error = errMsg.String
		}
		if res.Valid {
			j.ResultJSON = res.String
		}
		j.CreatedAt = time.Unix(created.Int64, 0)
		if started.Valid {
			t := time.Unix(started.Int64, 0)
			j.StartedAt = &t
		}
		if finished.Valid {
			t := time.Unix(finished.Int64, 0)
			j.FinishedAt = &t
		}
		list = append(list, j)
	}
	return list, rows.Err()
}

func (s *Store) GetJob(ctx context.Context, id string) (*Job, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, path, status, created_at, started_at, finished_at, err, result_json FROM jobs WHERE id = ?`, id)
	var j Job
	var created, started, finished sql.NullInt64
	var errMsg, res sql.NullString
	err := row.Scan(&j.ID, &j.Path, &j.Status, &created, &started, &finished, &errMsg, &res)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if errMsg.Valid {
		j.Error = errMsg.String
	}
	if res.Valid {
		j.ResultJSON = res.String
	}
	j.CreatedAt = time.Unix(created.Int64, 0)
	if started.Valid {
		t := time.Unix(started.Int64, 0)
		j.StartedAt = &t
	}
	if finished.Valid {
		t := time.Unix(finished.Int64, 0)
		j.FinishedAt = &t
	}
	return &j, nil
}

var VideoExtensions = map[string]struct{}{
	".mp4": {}, ".mkv": {}, ".webm": {}, ".mov": {}, ".avi": {}, ".m4v": {},
	".wmv": {}, ".flv": {}, ".ts": {}, ".m2ts": {}, ".mpg": {}, ".mpeg": {},
}

func IsVideo(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	_, ok := VideoExtensions[ext]
	return ok
}

// ScanRoot walks root and enqueues every video file using batched inserts.
func (s *Store) ScanRoot(root string) (added int, skipped int, err error) {
	root = filepath.Clean(root)
	st, err := os.Stat(root)
	if err != nil {
		return 0, 0, err
	}
	if !st.IsDir() {
		if !IsVideo(root) {
			return 0, 0, fmt.Errorf("not a video file: %s", root)
		}
		abs, err := filepath.Abs(root)
		if err != nil {
			return 0, 0, err
		}
		_, created, err := s.EnqueueVideo(abs)
		if err != nil {
			return 0, 0, err
		}
		if created {
			return 1, 0, nil
		}
		return 0, 1, nil
	}

	// Collect all video paths first, then batch-insert in one transaction.
	var paths []string
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if !IsVideo(path) {
			return nil
		}
		abs, absErr := filepath.Abs(path)
		if absErr != nil {
			return absErr
		}
		paths = append(paths, abs)
		return nil
	})
	if err != nil {
		return 0, 0, err
	}

	const batchSize = 500
	now := time.Now().Unix()
	for i := 0; i < len(paths); i += batchSize {
		end := i + batchSize
		if end > len(paths) {
			end = len(paths)
		}
		batch := paths[i:end]

		tx, txErr := s.db.Begin()
		if txErr != nil {
			return added, skipped, txErr
		}
		insertStmt, prepErr := tx.Prepare(
			`INSERT INTO jobs(id, path, status, created_at) VALUES(?, ?, ?, ?)`)
		if prepErr != nil {
			_ = tx.Rollback()
			return added, skipped, prepErr
		}
		checkStmt, prepErr := tx.Prepare(`SELECT 1 FROM jobs WHERE path = ? LIMIT 1`)
		if prepErr != nil {
			_ = insertStmt.Close()
			_ = tx.Rollback()
			return added, skipped, prepErr
		}

		for _, p := range batch {
			var exists int
			scanErr := checkStmt.QueryRow(p).Scan(&exists)
			if scanErr == nil {
				skipped++
				continue
			}
			if !errors.Is(scanErr, sql.ErrNoRows) {
				_ = insertStmt.Close()
				_ = checkStmt.Close()
				_ = tx.Rollback()
				return added, skipped, scanErr
			}
			_, execErr := insertStmt.Exec(uuid.NewString(), p, string(StatusPending), now)
			if execErr != nil {
				_ = insertStmt.Close()
				_ = checkStmt.Close()
				_ = tx.Rollback()
				return added, skipped, execErr
			}
			added++
		}
		_ = insertStmt.Close()
		_ = checkStmt.Close()
		if commitErr := tx.Commit(); commitErr != nil {
			return added, skipped, commitErr
		}
	}
	return added, skipped, nil
}
