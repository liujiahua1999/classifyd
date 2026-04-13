package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
)

// TagFrequencyRow is one row in the tag histogram (frequency_tokens across done jobs).
type TagFrequencyRow struct {
	Tag         string  `json:"tag"`
	Count       int64   `json:"count"`
	PctOfDone   float64 `json:"pct_of_done_jobs"`
	Note        string  `json:"note,omitempty"`
}

// Bucket is a grouped count (e.g. video codec distribution).
type Bucket struct {
	Key   string  `json:"key"`
	Count int64   `json:"count"`
	Pct   float64 `json:"pct_of_done_jobs"`
}

// FFProbeAggregate summarizes stored probe columns for finished jobs.
type FFProbeAggregate struct {
	DoneJobs       int64    `json:"done_jobs"`
	VideoCodec     []Bucket `json:"video_codec"`
	AudioCodec     []Bucket `json:"audio_codec"`
	Container      []Bucket `json:"container"`
	AvgDuration    *float64 `json:"avg_duration_sec,omitempty"`
	DominantRating []Bucket `json:"dominant_rating"`
	Characters     []Bucket `json:"characters"`
	Taggers        []Bucket `json:"taggers"`
}

// CountDoneJobs returns how many jobs are in status done.
func (s *Store) CountDoneJobs(ctx context.Context) (int64, error) {
	var n int64
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM jobs WHERE status = ?`, string(StatusDone)).Scan(&n)
	return n, err
}

// TagFrequency lists tag token frequencies; pct is relative to done job count (denominator).
func (s *Store) TagFrequency(ctx context.Context, limit int, minCount int64, prefix string) ([]TagFrequencyRow, int64, error) {
	done, err := s.CountDoneJobs(ctx)
	if err != nil {
		return nil, 0, err
	}
	if limit <= 0 || limit > 2000 {
		limit = 500
	}
	if minCount < 1 {
		minCount = 1
	}
	prefix = strings.TrimSpace(prefix)

	var rows *sql.Rows
	if prefix != "" {
		rows, err = s.db.QueryContext(ctx,
			`SELECT tag, COUNT(*) AS c FROM job_tags WHERE tag LIKE ? GROUP BY tag HAVING c >= ? ORDER BY c DESC LIMIT ?`,
			prefix+"%", minCount, limit)
	} else {
		rows, err = s.db.QueryContext(ctx,
			`SELECT tag, COUNT(*) AS c FROM job_tags GROUP BY tag HAVING c >= ? ORDER BY c DESC LIMIT ?`,
			minCount, limit)
	}
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	out := make([]TagFrequencyRow, 0)
	denom := float64(done)
	if denom <= 0 {
		denom = 1
	}
	for rows.Next() {
		var tag string
		var c int64
		if err := rows.Scan(&tag, &c); err != nil {
			return nil, 0, err
		}
		pct := (float64(c) / denom) * 100
		note := ""
		switch {
		case pct >= 95:
			note = "appears on almost every finished job — often boilerplate metadata"
		case pct >= 70:
			note = "very common — low discriminative value for filtering"
		case pct <= 2 && c >= minCount:
			note = "rare — may be worth keeping if meaningful"
		}
		out = append(out, TagFrequencyRow{
			Tag:       tag,
			Count:     c,
			PctOfDone: pct,
			Note:      note,
		})
	}
	return out, done, rows.Err()
}

// FFProbeAggregates groups codec/container columns for done jobs.
func (s *Store) FFProbeAggregates(ctx context.Context) (*FFProbeAggregate, error) {
	done, err := s.CountDoneJobs(ctx)
	if err != nil {
		return nil, err
	}
	out := &FFProbeAggregate{DoneJobs: done}
	if done == 0 {
		return out, nil
	}
	denom := float64(done)

	var avg sql.NullFloat64
	if err := s.db.QueryRowContext(ctx,
		`SELECT AVG(duration_sec) FROM jobs WHERE status = ? AND duration_sec IS NOT NULL`,
		string(StatusDone),
	).Scan(&avg); err != nil {
		return nil, err
	}
	if avg.Valid {
		v := avg.Float64
		out.AvgDuration = &v
	}

	buckets := func(col string) ([]Bucket, error) {
		var q string
		switch col {
		case "video_codec":
			q = `SELECT video_codec, COUNT(*) FROM jobs WHERE status = ? AND video_codec IS NOT NULL AND TRIM(video_codec) != '' GROUP BY video_codec ORDER BY COUNT(*) DESC LIMIT 50`
		case "audio_codec":
			q = `SELECT audio_codec, COUNT(*) FROM jobs WHERE status = ? AND audio_codec IS NOT NULL AND TRIM(audio_codec) != '' GROUP BY audio_codec ORDER BY COUNT(*) DESC LIMIT 50`
		case "container_fmt":
			q = `SELECT container_fmt, COUNT(*) FROM jobs WHERE status = ? AND container_fmt IS NOT NULL AND TRIM(container_fmt) != '' GROUP BY container_fmt ORDER BY COUNT(*) DESC LIMIT 50`
		case "dominant_rating":
			q = `SELECT dominant_rating, COUNT(*) FROM jobs WHERE status = ? AND dominant_rating IS NOT NULL AND TRIM(dominant_rating) != '' GROUP BY dominant_rating ORDER BY COUNT(*) DESC LIMIT 50`
		case "tagger":
			q = `SELECT tagger, COUNT(*) FROM jobs WHERE status = ? AND tagger IS NOT NULL AND TRIM(tagger) != '' GROUP BY tagger ORDER BY COUNT(*) DESC LIMIT 50`
		default:
			return nil, fmt.Errorf("unknown column %q", col)
		}
		rows, err := s.db.QueryContext(ctx, q, string(StatusDone))
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		var list []Bucket
		for rows.Next() {
			var k string
			var c int64
			if err := rows.Scan(&k, &c); err != nil {
				return nil, err
			}
			list = append(list, Bucket{
				Key:   k,
				Count: c,
				Pct:   (float64(c) / denom) * 100,
			})
		}
		return list, rows.Err()
	}

	vc, err := buckets("video_codec")
	if err != nil {
		return nil, err
	}
	ac, err := buckets("audio_codec")
	if err != nil {
		return nil, err
	}
	cf, err := buckets("container_fmt")
	if err != nil {
		return nil, err
	}
	out.VideoCodec = vc
	out.AudioCodec = ac
	out.Container = cf

	dr, err := buckets("dominant_rating")
	if err != nil {
		return nil, err
	}
	out.DominantRating = dr

	tg, err := buckets("tagger")
	if err != nil {
		return nil, err
	}
	out.Taggers = tg

	chars, err := s.characterFrequency(ctx, denom)
	if err != nil {
		return nil, err
	}
	out.Characters = chars
	return out, nil
}

func (s *Store) characterFrequency(ctx context.Context, denom float64) ([]Bucket, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT tag, COUNT(*) AS c FROM job_tags WHERE tag LIKE 'character:%' GROUP BY tag ORDER BY c DESC LIMIT 100`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Bucket
	for rows.Next() {
		var tag string
		var c int64
		if err := rows.Scan(&tag, &c); err != nil {
			return nil, err
		}
		name := strings.TrimPrefix(tag, "character:")
		idx := strings.LastIndex(name, ":")
		if idx > 0 {
			name = name[:idx]
		}
		out = append(out, Bucket{Key: name, Count: c, Pct: (float64(c) / denom) * 100})
	}
	return out, rows.Err()
}

// ListJobTags returns frequency tokens stored for a job.
func (s *Store) ListJobTags(ctx context.Context, jobID string) ([]string, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT tag FROM job_tags WHERE job_id = ? ORDER BY tag`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]string, 0)
	for rows.Next() {
		var t string
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// RebuildJobTags repopulates job_tags from stored result_json (frequency_tokens). Use after upgrades or DB repair.
func (s *Store) RebuildJobTags(ctx context.Context) (jobsUpdated int, tagsWritten int, err error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, `DELETE FROM job_tags`); err != nil {
		return 0, 0, err
	}
	rows, err := tx.QueryContext(ctx, `SELECT id, result_json FROM jobs WHERE status = ? AND result_json IS NOT NULL AND result_json != ''`, string(StatusDone))
	if err != nil {
		return 0, 0, err
	}
	defer rows.Close()

	ins, err := tx.PrepareContext(ctx, `INSERT OR IGNORE INTO job_tags(job_id, tag) VALUES(?, ?)`)
	if err != nil {
		return 0, 0, err
	}
	defer ins.Close()

	type wrap struct {
		Tokens []string `json:"frequency_tokens"`
	}
	for rows.Next() {
		var id, js string
		if err := rows.Scan(&id, &js); err != nil {
			return 0, 0, err
		}
		var w wrap
		if err := json.Unmarshal([]byte(js), &w); err != nil || len(w.Tokens) == 0 {
			continue
		}
		jobsUpdated++
		for _, t := range w.Tokens {
			t = strings.TrimSpace(t)
			if t == "" {
				continue
			}
			if _, err := ins.ExecContext(ctx, id, t); err != nil {
				return jobsUpdated, tagsWritten, err
			}
			tagsWritten++
		}
	}
	if err := rows.Err(); err != nil {
		return jobsUpdated, tagsWritten, err
	}
	if err := tx.Commit(); err != nil {
		return jobsUpdated, tagsWritten, err
	}
	return jobsUpdated, tagsWritten, nil
}
