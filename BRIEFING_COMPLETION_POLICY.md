# Briefing Completion Policy

## Completion

A briefing is complete when the user explicitly marks its saved five-slot snapshot complete. The completion record stores the snapshot, generation date, completion time, intent, reading budget, and reviewed-slot count. Refreshing does not alter that completed record.

## New Critical Item Rule

After completion, a newly fetched article may create a new review delta only when all of these conditions are true:

- It was not present in the completed snapshot.
- Its normalized ranking score is at least 85.
- Its signal confidence is High.
- It has a Market Moving or Business Opportunity flag.

The completed briefing remains unchanged. A qualifying article must be presented as a separate new-critical-item review, not inserted into or used to reshuffle the completed session. This rule is documented for a later implementation; SF-006 persists completion history only.
