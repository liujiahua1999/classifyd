package pipeline

import "testing"

func TestParseFPS(t *testing.T) {
	f, ok := parseFPS("30000/1001")
	if !ok {
		t.Fatal("expected ok")
	}
	if f < 29.9 || f > 30.0 {
		t.Fatalf("fps: %v", f)
	}
}
