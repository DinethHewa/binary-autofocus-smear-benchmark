# Explainability Analysis

- Grad-CAM and occlusion-sensitivity panels were successfully generated for: `cnn`, `cnn_attention`, `transfer`.
- The saved-artifact pipeline attempted both the top custom CNN and the best saved transfer model; any missing model reflects a recoverability issue that is documented in `warnings_and_limitations.md`.
- Quantitative support fields were restricted to simple heatmap-mass heuristics (`foreground_focus_share`, `border_focus_share`) to avoid overclaiming from qualitative overlays.
- Across the selected examples, heatmaps were interpreted conservatively: foreground-heavy attention is compatible with attention to smear material, but the current artifacts do not justify strong claims about specific subcellular structures such as nuclei.

Observed heuristic summary:

- `cnn`: median foreground focus share `0.9794`, median border focus share `0.4664`.
- `cnn_attention`: median foreground focus share `1.0000`, median border focus share `0.4601`.
- `transfer`: median foreground focus share `0.0000`, median border focus share `0.0000`.

Interpretation:

- Higher foreground-focus share suggests that the explanation mass concentrated on non-background image regions.
- Elevated border-focus share in an error case should be discussed as a potential artifact-attention failure mode rather than as evidence of clinically meaningful localization.