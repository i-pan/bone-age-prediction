# Data and split assignments

`splits.csv` preserves the 14,036 image IDs, source subsets, and five-fold assignments used by the training recipe. It contains no radiographs, age labels, or sex labels. Folds 0, 1 and 2 supply the three exported model members; partitions 3 and 4 remain part of the training pool for those members.

Obtain the source dataset separately and provide a label table containing `pid,bone_age,female,subset` (or an explicit relative `imgfile`). The `prepare` command joins by `pid`, checks complete coverage and subset agreement, and writes paired uncropped/cropped rows with identical folds. Keep the 200-image challenge test set separate from training.
