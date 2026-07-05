// loadgen stress-tests the inference fleet (ROADMAP Part 1).
//
// Open-loop load: requests are dispatched on a fixed schedule derived from
// -rps (optionally ramping up over -ramp), regardless of how slowly the
// server answers — so a struggling fleet shows up as rising latency and
// errors, not as a politely slowed-down client. A bounded worker pool
// (-concurrency) caps in-flight requests; dispatches that find no free
// worker are counted as "dropped" (the fleet fell behind the offered load).
//
// Chaos mode: -kill-after runs -kill-cmd mid-test (e.g. `spot-orchestrate
// fleet kill-worker --local`) and stamps the kill time into the report, so
// the per-second series shows the error blip and recovery.
//
// Usage:
//
//	go run . -url http://localhost:8000 -rps 20 -duration 60s \
//	  -kill-after 30s -kill-cmd "spot-orchestrate fleet kill-worker --local"
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

type config struct {
	url         string
	rps         float64
	concurrency int
	duration    time.Duration
	ramp        time.Duration
	promptsFile string
	maxTokens   int
	temperature float64
	timeout     time.Duration
	out         string
	killAfter   time.Duration
	killCmd     string
}

type result struct {
	at        time.Time
	latencyMs float64
	status    int
	failed    bool
	tokens    int
}

type secondBucket struct {
	Second    int     `json:"t"`
	Sent      int     `json:"sent"`
	OK        int     `json:"ok"`
	Errors    int     `json:"errors"`
	Dropped   int     `json:"dropped"`
	P99Ms     float64 `json:"p99_ms"`
	MeanMs    float64 `json:"mean_ms"`
	latencies []float64
}

type report struct {
	URL          string         `json:"url"`
	RPS          float64        `json:"rps"`
	Concurrency  int            `json:"concurrency"`
	DurationSec  float64        `json:"duration_s"`
	Requests     int            `json:"requests"`
	Succeeded    int            `json:"succeeded"`
	Failed       int            `json:"failed"`
	Dropped      int            `json:"dropped"`
	ErrorRate    float64        `json:"error_rate"`
	Tokens       int            `json:"completion_tokens"`
	TokensPerSec float64        `json:"tokens_per_second"`
	MeanMs       float64        `json:"mean_ms"`
	P50Ms        float64        `json:"p50_ms"`
	P90Ms        float64        `json:"p90_ms"`
	P95Ms        float64        `json:"p95_ms"`
	P99Ms        float64        `json:"p99_ms"`
	KillAtSec    float64        `json:"kill_at_s,omitempty"`
	KillCmd      string         `json:"kill_cmd,omitempty"`
	PerSecond    []secondBucket `json:"per_second"`
}

var defaultPrompts = []string{"ROMEO:", "JULIET:", "First Citizen:", "KING HENRY:", "\n"}

func main() {
	cfg := parseFlags()
	prompts := loadPrompts(cfg.promptsFile)

	client := &http.Client{Timeout: cfg.timeout}
	jobs := make(chan string) // unbuffered: a blocked send means no free worker
	results := make(chan result, 1024)
	var wg sync.WaitGroup

	for i := 0; i < cfg.concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for prompt := range jobs {
				results <- doRequest(client, cfg, prompt)
			}
		}()
	}

	// Collector owns all aggregation; done when the results channel closes.
	collectorDone := make(chan report)
	start := time.Now()
	go collect(results, start, collectorDone)

	if cfg.killAfter > 0 && cfg.killCmd != "" {
		go func() {
			time.Sleep(cfg.killAfter)
			fmt.Fprintf(os.Stderr, "[loadgen] chaos: running %q\n", cfg.killCmd)
			out, err := exec.Command("sh", "-c", cfg.killCmd).CombinedOutput()
			if err != nil {
				fmt.Fprintf(os.Stderr, "[loadgen] kill-cmd failed: %v: %s\n", err, out)
			}
		}()
	}

	dropped := dispatch(cfg, prompts, jobs, start)
	close(jobs)
	wg.Wait()
	close(results)
	rep := <-collectorDone

	rep.URL = cfg.url
	rep.RPS = cfg.rps
	rep.Concurrency = cfg.concurrency
	rep.Dropped = dropped
	if cfg.killAfter > 0 {
		rep.KillAtSec = cfg.killAfter.Seconds()
		rep.KillCmd = cfg.killCmd
	}
	printSummary(rep)
	if cfg.out != "" {
		writeReport(cfg.out, rep)
	}
	if rep.Failed > 0 {
		os.Exit(1) // scripts can assert the zero-visible-errors criterion
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.url, "url", "http://localhost:8000", "router base URL")
	flag.Float64Var(&cfg.rps, "rps", 10, "target requests per second (open loop)")
	flag.IntVar(&cfg.concurrency, "concurrency", 64, "max in-flight requests")
	flag.DurationVar(&cfg.duration, "duration", 60*time.Second, "test duration")
	flag.DurationVar(&cfg.ramp, "ramp", 0, "ramp RPS from 0 to target over this window")
	flag.StringVar(&cfg.promptsFile, "prompts", "", "file with one prompt per line (default: built-ins)")
	flag.IntVar(&cfg.maxTokens, "max-tokens", 64, "completion tokens per request")
	flag.Float64Var(&cfg.temperature, "temperature", 0.8, "sampling temperature")
	flag.DurationVar(&cfg.timeout, "timeout", 60*time.Second, "per-request timeout")
	flag.StringVar(&cfg.out, "out", "loadgen-report.json", "report path (empty to skip)")
	flag.DurationVar(&cfg.killAfter, "kill-after", 0, "run -kill-cmd after this delay (0 = off)")
	flag.StringVar(&cfg.killCmd, "kill-cmd", "", "shell command that kills a worker mid-test")
	flag.Parse()
	if cfg.rps <= 0 || cfg.concurrency <= 0 {
		fmt.Fprintln(os.Stderr, "loadgen: -rps and -concurrency must be positive")
		os.Exit(2)
	}
	return cfg
}

func loadPrompts(path string) []string {
	if path == "" {
		return defaultPrompts
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "loadgen: cannot read prompts: %v\n", err)
		os.Exit(2)
	}
	var prompts []string
	for _, line := range strings.Split(string(data), "\n") {
		if line = strings.TrimRight(line, "\r"); line != "" {
			prompts = append(prompts, line)
		}
	}
	if len(prompts) == 0 {
		return defaultPrompts
	}
	return prompts
}

// dispatch sends prompts to the pool on the open-loop schedule and returns
// how many dispatches were dropped because every worker was busy.
func dispatch(cfg config, prompts []string, jobs chan<- string, start time.Time) int {
	dropped, sent := 0, 0
	for {
		elapsed := time.Since(start)
		if elapsed >= cfg.duration {
			return dropped
		}
		rate := cfg.rps
		if cfg.ramp > 0 && elapsed < cfg.ramp {
			rate = cfg.rps * float64(elapsed) / float64(cfg.ramp)
			if rate < 1e-3 {
				rate = 1e-3
			}
		}
		// Next send time on the ideal schedule; sleep until then.
		next := start.Add(time.Duration(float64(sent) * float64(time.Second) / rate))
		if wait := time.Until(next); wait > 0 {
			time.Sleep(wait)
		}
		select {
		case jobs <- prompts[sent%len(prompts)]:
		default:
			dropped++ // pool saturated: the fleet fell behind the offered load
		}
		sent++
	}
}

func doRequest(client *http.Client, cfg config, prompt string) result {
	body, _ := json.Marshal(map[string]any{
		"prompt":      prompt,
		"max_tokens":  cfg.maxTokens,
		"temperature": cfg.temperature,
	})
	t0 := time.Now()
	resp, err := client.Post(cfg.url+"/v1/completions", "application/json", bytes.NewReader(body))
	lat := float64(time.Since(t0).Microseconds()) / 1000.0
	if err != nil {
		return result{at: t0, latencyMs: lat, failed: true}
	}
	defer resp.Body.Close()
	tokens := 0
	var payload struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	data, _ := io.ReadAll(resp.Body)
	if json.Unmarshal(data, &payload) == nil {
		tokens = payload.Usage.CompletionTokens
	}
	return result{
		at:        t0,
		latencyMs: lat,
		status:    resp.StatusCode,
		failed:    resp.StatusCode >= 400,
		tokens:    tokens,
	}
}

func collect(results <-chan result, start time.Time, done chan<- report) {
	buckets := map[int]*secondBucket{}
	var all []float64
	rep := report{}
	var sumMs float64
	for r := range results {
		sec := int(r.at.Sub(start).Seconds())
		b, ok := buckets[sec]
		if !ok {
			b = &secondBucket{Second: sec}
			buckets[sec] = b
		}
		b.Sent++
		rep.Requests++
		if r.failed {
			b.Errors++
			rep.Failed++
			continue
		}
		b.OK++
		rep.Succeeded++
		rep.Tokens += r.tokens
		b.latencies = append(b.latencies, r.latencyMs)
		all = append(all, r.latencyMs)
		sumMs += r.latencyMs
	}
	secs := make([]int, 0, len(buckets))
	for s := range buckets {
		secs = append(secs, s)
	}
	sort.Ints(secs)
	for _, s := range secs {
		b := buckets[s]
		b.P99Ms = percentile(b.latencies, 99)
		b.MeanMs = mean(b.latencies)
		b.latencies = nil
		rep.PerSecond = append(rep.PerSecond, *b)
	}
	rep.DurationSec = time.Since(start).Seconds()
	if rep.Requests > 0 {
		rep.ErrorRate = float64(rep.Failed) / float64(rep.Requests)
	}
	if rep.DurationSec > 0 {
		rep.TokensPerSec = float64(rep.Tokens) / rep.DurationSec
	}
	if len(all) > 0 {
		rep.MeanMs = sumMs / float64(len(all))
		rep.P50Ms = percentile(all, 50)
		rep.P90Ms = percentile(all, 90)
		rep.P95Ms = percentile(all, 95)
		rep.P99Ms = percentile(all, 99)
	}
	done <- rep
}

func percentile(v []float64, p float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	idx := int(math.Ceil(p/100.0*float64(len(s)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range v {
		sum += x
	}
	return sum / float64(len(v))
}

func printSummary(r report) {
	fmt.Printf("\n=== loadgen report ===\n")
	fmt.Printf("target        %s @ %.1f rps for %.0fs\n", r.URL, r.RPS, r.DurationSec)
	fmt.Printf("requests      %d ok=%d failed=%d dropped=%d (error rate %.2f%%)\n",
		r.Requests, r.Succeeded, r.Failed, r.Dropped, r.ErrorRate*100)
	fmt.Printf("latency ms    mean=%.1f p50=%.1f p90=%.1f p95=%.1f p99=%.1f\n",
		r.MeanMs, r.P50Ms, r.P90Ms, r.P95Ms, r.P99Ms)
	fmt.Printf("tokens        %d (%.1f tok/s)\n", r.Tokens, r.TokensPerSec)
	if r.KillCmd != "" {
		fmt.Printf("chaos         killed a worker at t=%.0fs (%q)\n", r.KillAtSec, r.KillCmd)
	}
}

func writeReport(path string, r report) {
	data, err := json.MarshalIndent(r, "", "  ")
	if err == nil {
		err = os.WriteFile(path, data, 0o644)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "loadgen: write report: %v\n", err)
		return
	}
	fmt.Printf("report        %s\n", path)
}
