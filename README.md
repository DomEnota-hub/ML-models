# ML-models

Independent model distribution catalog for MediaLoader (DomEnota-hub/ML).

## Status

Initial catalog, NOT an authorized remote update release. The existing Android SHA-256 pins remain authoritative. No new weights are approved by this repository yet. The existing verified MIT SRNet artifact is published as base64 with its complete attribution. No changed model weights are authorized.

- `catalog.json` records existing, code-reviewed binary identities and upstream download sources.
- Model records explicitly distinguish upstream download, offline installation, and verified import.
- A changed upstream digest is not an approved update. A future published release must include an immutable artifact identity, exact size, SHA-256, compatibility, license evidence and test receipt.
- DExNet + VGG19 redistribution rights remain unconfirmed. No binary or download endpoint for that combined model is advertised.
- SRNet offers an immutable pinned download and retains its offline installation; license and attribution accompany the published file in srnet/1/NOTICE.md.
- Never place signing private keys, user photographs or application APKs here.

Source baseline: https://github.com/DomEnota-hub/ML/tree/baa81fdbd8c5ceb36eaa6d285df7f664e037522f

## Activation gate

The public repository is transport, not proof of compatibility. A future release catalog must be authenticated using an owner-controlled signing identity, with monotonic version, compatibility and expiry checks. Validate the entire model bundle before atomic activation, preserve the installed generation on failure, and require explicit installation/update. Current app pins and offline/import fallbacks remain in effect until the actual remote-update implementation is validated.

See https://github.com/DomEnota-hub/ML/blob/e846aa49c68eb3bded09b6879311fa6f3041a0ad/docs/models/distribution-gate.md for the release gate. 
