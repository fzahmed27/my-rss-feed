# Signal Feed Founder Mode TODO

## MVP: Daily Decision Loop

- [x] Replace default endless feed with a bounded reading session.
- [x] Add priority tiers: Must Read, Worth Skimming, and collapsed More.
- [x] Show read progress for Must Read and Worth Skimming.
- [x] Add a caught-up state: "You're caught up. Go build."
- [x] Preserve search, filtering, bookmarks, source ingestion, parsing, and existing navigation.
- [x] Persist read, dismissed, feedback, muted-topic, and intent state locally.
- [x] Add a configurable reading budget with a 20-minute default.
- [x] Show planned review time against the configured reading budget.
- [ ] Stabilize completion state across refresh windows so already-read sessions do not reshuffle unless a material new signal arrives.
- [x] Define the urgency threshold for surfacing new critical items after completion.
- [x] Add a daily briefing completion marker and session history.

## Intent And Profile

- [x] Add visible intent selection near the top of the feed.
- [x] Persist selected intent locally.
- [x] Default to Build Controls AI.
- [x] Keep intent weights centralized in a configurable profile structure.
- [x] Add industrial AI founder preference weights for controls, PLCs, PID tuning, embedded AI, edge inference, robotics, sensors, predictive maintenance, developer tools, and commercialization.
- [x] Add penalties for generic consumer AI, AI drama, repetitive launches, weak funding announcements, generic robotics papers, and duplicate stories.
- [ ] Decide whether to keep the current five industrial-founder modes or migrate to the roadmap modes: Operate, Explore, Fundraise, Build.
- [x] Add editable founder profile fields for company, customers, product areas, competitors, priorities, and topics.
- [ ] Extend the founder profile with stage, geographies, stack, evidence preferences, risk tolerance, reading time, and notifications.
- [ ] Label inferred profile fields separately from explicitly configured fields.
- [x] Add export and reset controls for the founder profile.

## Explainable Ranking

- [x] Refactor ranking into explicit weighted components.
- [x] Use component weights: personal relevance 30%, actionability 25%, strategic importance 20%, source quality 10%, novelty 10%, recency 5%.
- [x] Normalize component values before combining them.
- [x] Keep scoring deterministic and testable.
- [x] Store component-level explanations with ranked articles.
- [x] Add explicit penalties for duplicate topic, generic announcement, low technical depth, ignored themes, and stale low-value content.
- [x] Show a concise "Why this matters" explanation on cards.
- [x] Keep detailed ranking components behind a disclosure/detail view.
- [x] Add confidence, uncertainty, evidence/source attribution, "Should I care?", and estimated reading time to the signal contract.
- [ ] Add source corroboration and source-diversity scoring.
- [ ] Add ranking factor logs suitable for offline evaluation.
- [ ] Build a "how ranking works" guide inside the app or README.

## Feedback And Learning

- [x] Keep thumbs up and thumbs down.
- [x] Add structured positive and negative feedback reasons.
- [x] Persist feedback reasons locally.
- [x] Apply bounded, gradual preference updates from feedback.
- [x] Add mute topic for 30 days.
- [x] Add undo feedback.
- [x] Make feedback inspectable and exportable.
- [ ] Attach feedback to source, topic, entity, event, summary, or ranking layer instead of only article keywords.
- [ ] Add immediate acknowledgement text explaining the expected effect of feedback.
- [ ] Add "Acted on", "Wrong summary", "Wrong priority", "Too speculative", "Too repetitive", "More like this", and "Less like this" actions from the roadmap.
- [ ] Add personalization-change inspection so the user can see which durable preferences changed.

## Founder Briefing

- [x] Preserve and upgrade the Briefing tab.
- [x] Generate exactly five category slots.
- [x] Include AI capability, industrial automation, developer-tool, startup/commercialization, and science/space items.
- [x] Avoid duplicate stories within the briefing.
- [x] Show an explicit empty state when no strong item exists for a category.
- [x] Include what happened, why it matters, what you should do, should-care, confidence, evidence, and reading time.
- [x] Allow sharing the briefing text.
- [x] Add briefing save/history as a first-class persisted object.
- [ ] Add two opportunity/opening slots.
- [ ] Add one risk or roadmap implication slot.
- [ ] Add one "safe to ignore" noisy-topic slot.
- [ ] Allow comparing how the briefing changes when intent changes.
- [ ] Add scheduled daily briefing generation or define on-open generation rules.

## Opportunity Engine

- [x] Preserve lightweight opportunity labels for market-moving, business opportunity, and AI launch signals.
- [x] Create a persisted Opportunity Hypothesis with problem, customer, trigger, wedge, evidence, confidence, and next test.
- [ ] Extend Opportunity Hypotheses with market, competition, moat, difficulty, and timing.
- [x] Label opportunity cards as hypotheses, not facts.
- [x] Link opportunity hypotheses to supporting articles and sources.
- [x] Add a persisted research queue for opportunity hypotheses.
- [ ] Add save-to-research-project behavior.
- [ ] Add evidence-for/evidence-against tracking.
- [ ] Add human-review rubric before surfacing stronger opportunity claims.

## Enrichment, Events, And Dedupe

- [x] Keep RSS/Atom ingestion and article parsing intact.
- [x] Preserve canonical URL cleanup and exact URL dedupe.
- [x] Add deterministic near-duplicate topic penalty.
- [x] Add event clustering so duplicate reports become one event card with expandable source coverage.
- [x] Prefer representative primary or highest-reputation sources within clusters.
- [ ] Store raw source provenance and derived-output pipeline versions.
- [ ] Add entity, topic, claim, and event extraction.
- [ ] Add confidence and generated-output version fields for every AI-generated enrichment.
- [ ] Add graceful pending-analysis states when enrichment is delayed.

## Knowledge, Trends, And Memory

- [ ] Add initial node types: article, event, topic, entity, product, person, technology, market, project, thesis, opportunity, decision, and memory item.
- [ ] Add initial relationships: mentions, competes with, supplies, funds, builds, depends on, supports, contradicts, affects, belongs to, and derived from.
- [ ] Build readable topic/entity pages before any graph visualization.
- [ ] Add trend cards based on source-adjusted mention volume, unique entities, releases, funding, hiring, adoption, and cross-topic connections.
- [ ] Show trend baseline, current value, likely drivers, evidence, confidence, caveats, and relevance.
- [ ] Add founder memory objects: ideas, notes, projects, companies, people, products, decisions, assumptions, theses, risks, and commitments.
- [ ] Link new signals to projects and theses.
- [ ] Detect evidence that supports, weakens, or contradicts saved theses.
- [ ] Add export, archive, delete, and privacy controls for memory.

## Instrumentation And Evaluation

- [ ] Define Weekly Useful Decisions events: acted on, saved to project, tracked, shared with context, or rated useful.
- [ ] Track briefing completion time and completion rate.
- [ ] Track Must Read useful rate.
- [ ] Track not-relevant and too-repetitive rates.
- [ ] Track mute and notification opt-out rates.
- [ ] Track summary correction and unsupported-claim rates once generated summaries exist.
- [ ] Build a 100-item evaluation set for relevance, duplicate-event precision, summary quality, and groundedness.
- [ ] Add offline replay for ranking changes.
- [ ] Add production dashboards for fetch success, parse errors, dedupe rate, latency, source health, and useful-item rate.

## Trust, Privacy, And Reliability

- [x] Preserve source URLs and source attribution in UI.
- [x] Keep opportunity language conservative and avoid financial-advice framing.
- [ ] Separate quoted evidence from generated interpretation.
- [ ] Add prompt-injection defenses for untrusted article content before any generative enrichment.
- [ ] Add profile and memory deletion/export/retention controls.
- [ ] Avoid logging unnecessary sensitive text.
- [ ] Add tenant isolation only when multi-user support is introduced.
- [ ] Make ingestion idempotent and observable.
- [ ] Support replay and re-enrichment when scoring or enrichment versions change.

## Testing

- [x] Add ranking tests for intent mode influence.
- [x] Add tests for industrial AI weighting.
- [x] Add duplicate penalty tests.
- [x] Add muted-topic tests.
- [x] Add feedback persistence tests.
- [x] Add reading-budget grouping tests.
- [x] Add Founder Briefing slot and empty-state tests.
- [x] Add score explanation and signal-contract tests.
- [x] Add presentation-settings default/migration test.
- [x] Add repository persistence tests for read state, muted topics, presentation settings, and feedback export.
- [ ] Add feedback-driven preference update tests at the store level.
- [x] Add UI previews/fixtures for fresh feed, caught-up feed, no high-quality articles, muted-topic state, long titles, and accessibility text sizes.
- [x] Add briefing snapshot persistence, deduplication, and saved-share tests.
- [x] Add deterministic event-clustering and persisted founder-profile ranking tests.
- [x] Add briefing-completion and opportunity-hypothesis persistence tests.
- [ ] Add accessibility audit tests for minimum tap targets, labels, contrast, and Dynamic Type.

## Product Decisions To Resolve

- [ ] Choose the first ideal customer: solo founder, founder plus chief of staff, or early product team.
- [ ] Pick the first two or three target domains for source quality and evaluation expertise.
- [ ] Decide scheduled briefing, on-open briefing, or hybrid generation.
- [ ] Define which actions count for Weekly Useful Decisions.
- [ ] Define the difference between urgent and strategically important but non-urgent.
- [ ] Decide how long behavior remains session-specific before becoming durable preference.
- [ ] Decide which memory objects require explicit confirmation before affecting ranking.
- [ ] Define the evidence standard for opportunity and trend surfacing.

## Suggested Next 5 Commits

1. Add feedback-driven preference update tests at the store level.
2. Add accessibility audit tests for tap targets, labels, contrast, and Dynamic Type.
3. Add two opportunity/opening slots to Founder Briefing.
4. Extend the founder profile with stage, geography, stack, evidence, risk, and notification fields.
5. Add source corroboration and source-diversity scoring.
