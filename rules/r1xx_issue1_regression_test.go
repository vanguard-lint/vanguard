package rules

import (
	"testing"

	"github.com/vanguard-lint/vanguard/internal/ir"
)

// TestR1xx01Issue1Regression pins the exact reproduction reported in
// vanguard-lint/vanguard#1: a singular base segment with a read-by-id
// handler — @GetMapping("/order/{id}") on a Spring controller, the F-1
// false-negative candidate pre-declared in the w5-06 FN list.
//
// The predicate (case (i) of pluralCollectionCheck: any non-plural plain
// resource segment directly followed by a path variable) already fired on
// main when the issue was triaged, so the fix needs no production change —
// these cases pin the issue's shapes so the base-segment singular miss
// cannot silently return.
func TestR1xx01Issue1Regression(t *testing.T) {
	r := ruleByID(t, "R1xx-01")

	// The issue's example fires with the pluralized-path suggestion.
	got := requireFindings(t, r, methodOf(ir.VerbGet, "/order/{id}"), 1)
	if got[0].Suggestion != "/orders/{id}" {
		t.Errorf("suggestion = %q, want /orders/{id}", got[0].Suggestion)
	}

	// The singular base still fires deeper in the path — the item
	// sub-resource of a singular root.
	requireFindings(t, r, methodOf(ir.VerbGet, "/order/{id}/items"), 1)

	// ...and the pluralized roots clear both shapes.
	requireFindings(t, r, methodOf(ir.VerbGet, "/orders/{id}"), 0)
	requireFindings(t, r, methodOf(ir.VerbGet, "/orders/{id}/items"), 0)
}
