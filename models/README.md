# Model Weights

Large binary files are NOT committed to git. Download them manually before running inference.

## Required for Sprint 2 (DINOv2 + SAM2 prototype)

### DINOv2
Downloaded automatically by `torch.hub` on first inference run.
Cached to `~/.cache/torch/hub/` — no manual step needed.

### SAM2 — sam2.1_hiera_small.pt
Place at: `models/sam2/sam2.1_hiera_small.pt`

```bash
mkdir -p models/sam2
curl -L -o models/sam2/sam2.1_hiera_small.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
```

Size: ~185MB. Source: https://github.com/facebookresearch/sam2/releases

The detector runs without SAM2 weights (bounding boxes only, no pixel masks)
and logs a warning. Masks are required for Sprint 3 fine-tuning.

### Query embedding — query_embedding.pt
Built by `perception/tools/build_query_embedding.py` from Laboro Tomato training set.
See REPRODUCE.md for the command.

### Negative embedding — negative_embedding.pt (optional)
Built with `--output-negative models/negative_embedding.pt` for contrastive scoring.
Suppresses leaf/stem confusers. Auto-loaded when present.

## Sprint 3 (after fine-tuning on Greene)

Fine-tuned weights are transferred from HPC to this directory via `model_sync.sh`.
They replace the SAM2 checkpoint above and are loaded by the same detector.
