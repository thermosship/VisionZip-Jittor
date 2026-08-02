# Sample images

Run the deterministic generator instead of committing binary image files:

```bash
/root/miniconda3/bin/python scripts/create_sample_images.py \
  --output-dir assets/sample_images
```

The generated `scene.png`, `text.png`, and `dense.png` are license-free
inputs for real CLIP feature extraction. You can also pass your own JPG/PNG/WebP
images to the pipeline.
