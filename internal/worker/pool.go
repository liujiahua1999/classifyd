package worker

import (
	"context"
	"log"
	"sync"
	"time"

	"classifyd/internal/pipeline"
	"classifyd/internal/store"
)

type Pool struct {
	Store      *store.Store
	Classifier pipeline.VideoClassifier
	N          int
}

func (p *Pool) Run(ctx context.Context) {
	n := p.N
	if n < 1 {
		n = 1
	}
	log.Printf("worker pool: starting %d worker(s)", n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			p.loop(ctx, id)
		}(i)
	}
	<-ctx.Done()
	wg.Wait()
	log.Printf("worker pool: all workers stopped")
}

const (
	minPoll = 200 * time.Millisecond
	maxPoll = 5 * time.Second
)

func (p *Pool) loop(ctx context.Context, id int) {
	backoff := minPoll
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		job, err := p.Store.ClaimNext(ctx)
		if err != nil {
			log.Printf("worker %d: claim error: %v", id, err)
			sleep(ctx, backoff)
			backoff = grow(backoff)
			continue
		}
		if job == nil {
			sleep(ctx, backoff)
			backoff = grow(backoff)
			continue
		}
		backoff = minPoll
		t0 := time.Now()
		log.Printf("worker %d: processing %s", id, job.Path)
		res, err := p.Classifier.ClassifyVideo(ctx, job.Path)
		elapsed := time.Since(t0)
		if err != nil {
			_ = p.Store.MarkFailed(ctx, job.ID, err.Error())
			log.Printf("worker %d: FAIL %s (%v): %v", id, job.Path, elapsed.Round(time.Millisecond), err)
			continue
		}
		var dm *store.DoneMeta
		if res.Meta != nil {
			m := res.Meta
			dm = &store.DoneMeta{
				DurationSec: m.DurationSec,
				VideoCodec:  m.VideoCodec,
				AudioCodec:  m.AudioCodec,
				Width:       m.Width,
				Height:      m.Height,
				Container:   m.Container,
			}
		}
		if err := p.Store.MarkDone(ctx, job.ID, string(res.Raw), res.FrequencyTokens, dm); err != nil {
			log.Printf("worker %d: mark-done error %s: %v", id, job.ID, err)
		} else {
			log.Printf("worker %d: done %s (%v)", id, job.Path, elapsed.Round(time.Millisecond))
		}
	}
}

func grow(d time.Duration) time.Duration {
	d = d * 3 / 2
	if d > maxPoll {
		return maxPoll
	}
	return d
}

func sleep(ctx context.Context, d time.Duration) {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	}
}
