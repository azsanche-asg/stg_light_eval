# Paper Assets Overview

## Tables

- **Tab1_synthetic.tex** (`\label{tab:synthetic}`)
  - Caption: "Synthetic evaluation (higher is better for Rule F1; arrows denote metric direction)."
  - Columns: `Method`, `Rule F1 (\uparrow)`, `Tree Edit (\downarrow)`, `Repeat Reg. (\downarrow)`, `MDL (\downarrow)`, `Persistence Err. (\downarrow)`, `Motion Err. (\downarrow)`

- **Tab2_real.tex** (`\label{tab:real}`)
  - Caption: "Real-data evaluation (\uparrow higher is better, \downarrow lower is better)."
  - Columns: `Method`, `Feature Cosine (\uparrow)`, `Multi-view Cons. (\uparrow)`, `Repeat Reg. (\downarrow)`, `Fourier Grid (\uparrow)`, `Depth RMSE (\downarrow)`, `Normal Agree. (\uparrow)`, `Track Persist. (\uparrow)`

- **Tab3_efficiency.tex** (`\label{tab:efficiency}`)
  - Caption: "Efficiency comparison (smaller is better)."
  - Columns: `Method`, `Artifact (MB) (\downarrow)`, `Peak GPU (MB) (\downarrow)`, `Steps (k) (\downarrow)`

## Figures

- **Fig1_grammar_overlays.png**
  - Caption: "Qualitative overlays on synthetic facades/streets and Cityscapes frames highlighting inferred grammar structure."

- **Fig2_persistence_motion.png**
  - Caption: "Node persistence curves and linear motion fits for a dynamic synthetic scene."

- **Fig3_edits_counterfactuals.png**
  - Caption: "Counterfactual edits: removing repeated façade elements and perturbing motion trajectories with CLIP-similarity deltas."
