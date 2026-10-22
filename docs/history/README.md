# Public history provenance

Before publication, a credential-shaped value in a configuration example was
replaced with a placeholder throughout the Git history. The public repository
preserves the original branches, commit authors, dates, and experimental code,
with commit identifiers changed by that redaction.

[`commit-map.tsv`](commit-map.tsv) maps pre-publication commit identifiers to
their public counterparts. Use it when an experiment transcript refers to an
original commit. An all-zero public identifier denotes an empty commit removed
during rewriting. Commits added after the redaction do not need this mapping.

The private fork retains its original history and the merged cleanup PR.
