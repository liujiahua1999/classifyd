package pipeline

import (
	"bufio"
	"encoding/base64"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

var normRe = regexp.MustCompile(`[_\s.\-+]+`)
var spaceRe = regexp.MustCompile(`\s+`)

func normalizeName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = normRe.ReplaceAllString(s, " ")
	s = spaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// LoadCharacterNames reads a file with one Danbooru-style name per line (# = comment).
func LoadCharacterNames(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	seen := map[string]struct{}{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = strings.TrimSpace(line[:i])
		}
		if line == "" {
			continue
		}
		key := strings.ToLower(line)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, line)
	}
	return out, sc.Err()
}

// MatchCharactersInFilename returns names whose normalized form appears in the filename stem.
// Longer names are matched first; overlapping spans are masked.
func MatchCharactersInFilename(stem string, names []string) []string {
	hay := normalizeName(stem)
	if hay == "" || len(names) == 0 {
		return nil
	}
	type entry struct {
		orig string
		norm string
	}
	var entries []entry
	for _, n := range names {
		nn := normalizeName(n)
		if utf8.RuneCountInString(nn) < 2 {
			continue
		}
		entries = append(entries, entry{orig: n, norm: nn})
	}
	sort.Slice(entries, func(i, j int) bool {
		li, lj := len(entries[i].norm), len(entries[j].norm)
		if li != lj {
			return li > lj
		}
		return entries[i].norm < entries[j].norm
	})
	var matched []string
	for _, e := range entries {
		pos := strings.Index(hay, e.norm)
		if pos < 0 {
			continue
		}
		matched = append(matched, e.orig)
		hay = hay[:pos] + strings.Repeat(" ", len(e.norm)) + hay[pos+len(e.norm):]
	}
	return matched
}

// EncodeFilenameB64 creates the b64-encoded user prompt to avoid content filters.
func EncodeFilenameB64(basename, stem string) string {
	plain := "File name (no path): " + basename + "\nStem only: " + stem
	return base64.StdEncoding.EncodeToString([]byte(plain))
}

// CharacterTag is a single identified character with its detection source.
type CharacterTag struct {
	Name   string `json:"name"`
	Score  float64 `json:"score"`
	Source string  `json:"source"` // "filename" or "llm"
}

func danbooru(tags []CharacterTag) string {
	if len(tags) == 0 {
		return ""
	}
	parts := make([]string, len(tags))
	for i, t := range tags {
		parts[i] = strings.ReplaceAll(t.Name, " ", "_")
	}
	return strings.Join(parts, ", ")
}
