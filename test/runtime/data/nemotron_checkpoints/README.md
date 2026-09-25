# Nemotron checkpoint fixtures

These fixtures contain official configuration and safetensors metadata for the
3B, 8B and 14B instruct checkpoints. No tensor payloads or generated answers are
included. Each `provenance.json` records the repository, immutable revision,
source link, original header SHA-256 and tokenizer file SHA-256.

`weights_header.json` is the checkpoint's complete safetensors header, with
`__metadata__` removed and JSON reformatted for review. It was retrieved with an
HTTP Range request and is checked against the runtime's weight mapping and full
model geometry on the meta device. The recorded header hash refers to the
original header bytes, before reformatting.

`adapter_header.json` records the optional draft LoRA tensors in the same format;
tests check every layer's A/B dimensions against its attention output projection.

The three tokenizer files have the same SHA-256, so the input token sequences in
`../nemotron_ar_fixtures.json` can be shared across sizes. Expected generated
tokens and numeric tolerances must be measured separately for each checkpoint.

CPU metadata tests do not establish GPU numerical parity or GSM8K accuracy.
See [GPU validation commands](../../integration/nemotron_validation.md).
