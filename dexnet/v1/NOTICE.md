# DExNet v1 graph-only recipe attribution

This repository publishes the non-executable compact graph skeleton and a range map, not the learned checkpoint bytes. The graph was reproduced from DExNet at da949e2291ea758a191fc0c93990767348e6e4ab and TorchVision 0.19.1 VGG19 architecture. Learned payloads were removed and graph bytes were checked against an independently initialized export. The final assembled ONNX and original pretrained checkpoints are not included.

- DExNet authors and source: https://github.com/jjhuangcs/DExNet/tree/da949e2291ea758a191fc0c93990767348e6e4ab ; original Apache-2.0 source license in LICENSE-DExNet.
- TorchVision source: https://github.com/pytorch/vision/tree/v0.19.1 ; original BSD-3-Clause source license in LICENSE-TorchVision.
- Modification: static 128x128 RGB graph export, VGG19 hypercolumn integration, removal of learned byte ranges, compact skeleton encoding and reconstruction map.
- These source licenses do not establish rights in the original pretrained checkpoints. Confirm terms for local use/conversion and before external distribution. Only the original publisher endpoints are referenced.
