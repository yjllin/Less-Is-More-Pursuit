# Released Checkpoints

Store only curated public checkpoints in this directory.

Recommended layout:

```text
checkpoints/
  ours_stage5/
    policy.pt
    config.json
    metrics.jsonl
```

Release policy:

- `policy.pt`: required
- `config.json`: strongly recommended so evaluation matches training exactly
- `metrics.jsonl`: optional, useful for reproducibility
- large files should be published through GitHub Releases or Git LFS

Do not copy full `runs/` subtrees into this directory.
