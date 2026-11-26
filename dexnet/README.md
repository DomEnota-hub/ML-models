# DExNet local assembly — development only

Owner decision 2026-11-26: investigate a local installer which obtains original components directly from their publishers and builds the already validated 128px ONNX on the user's device. This is not permission to redistribute checkpoints, extracted tensors or the combined ONNX. Review original terms for local use and conversion; public release remains gated.

## Repository contents

- recipe-v1.json: original locations, source revision, known digests, export and runtime contract. This is **research metadata**, not an executable installation manifest.
- No extracted trained DExNet/VGG19 tensor payloads or combined ONNX are stored here. The compact graph skeleton and offset/hash map are now available under dexnet/v1. They reconstruct the model only with the original checkpoint files.
- The known-good combined ONNX can still be imported with the existing exact size/SHA-256 check.
- VGG19 full SHA-256 and 574,673,361-byte original download verified in CI #45804969093. The exact prior ONNX was reconstructed from 1,854 contiguous source ranges. The compact skeleton publication is a separate audited operation; it contains no extracted trained tensor payloads.

## Implementation target

1. On a development workstation reproduce the existing pinned CPU export. Generate a weight-free ONNX template and a precise tensor mapping, checking the template contains no learned parameters or folded trained constants. Preserve original code notices.
2. Prototype an Android-compatible, bounded reader for the exact checkpoint serialization formats. Do not execute remote Python/pickle instructions. Download original files only after explicit user action and applicable terms are confirmed.
3. Populate the template locally, validate every tensor name/shape/type/size, exact output bytes/hash and numeric parity; test actual ARM64 resource use. Stage atomically, cancel safely, retain the previous working model, remove temporary files.
4. Only after validation, connect the existing model manager to a compiled, reviewed local assembler. Catalog data alone must never execute code or approve new weights.
5. Before external app distribution resolve source/checkpoint/VGG19 conditions and the separate signed model update mechanism.

Current status: byte-exact local reconstruction proven by CI #45804969093; graph skeleton and range map published in #45805495680. The Android assembler implementation is undergoing app CI #45806169039; no device validation yet. Verified import remains available.
