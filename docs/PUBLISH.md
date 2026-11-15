# Publish to GitHub

Suggested repository name: **cross-dataset-fall-detection**.

Suggested description: **Skeleton-based fall detection with RTMPose, CTR-GCN and reproducible four-dataset generalization evaluation.**

Suggested topics: `fall-detection`, `pytorch`, `computer-vision`, `skeleton-based-action-recognition`, `graph-convolutional-network`, `domain-generalization`.

Create an empty GitHub repository with that name, without an auto-generated README or license. The local project already contains its README, ignore rules and upstream notices.

From the prepared project root:

```bash
# Only needed when starting from the extracted ZIP rather than the prepared workspace:
git init -b main
git add .

# Use your Git author identity if it is not configured on this machine.
git commit -m "Add cross-dataset fall detection benchmark"
git remote add origin https://github.com/YOUR_USERNAME/cross-dataset-fall-detection.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If Git requests your identity, set your own `user.name` and `user.email`, then repeat the commit. The ZIP contains the clean source files and hidden GitHub workflow/ignore files, without Git history. Extract it before uploading its contents.

The original `app/`, `research/` and `figures4papers/` folders, raw data, pose caches, model weights, environments and logs are excluded. The public snapshot contains ERM results only; the SWAD result status is explained in the README.

Example CV bullet grounded in the bundled results:

> Built a reproducible PyTorch fall-detection pipeline using RTMPose and CTR-GCN; evaluated 520 recordings across four held-out datasets and three seeds, achieving 0.839 macro Fall F1 with source-only model selection.
