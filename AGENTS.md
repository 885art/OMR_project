# Repository instructions for AI agents

Before changing training, datasets, inference, or MusicXML behavior, read
`CHATGPT_PROJECT_CONTEXT.md` completely. Treat it as the current handoff record,
but verify paths and executable state from the repository and environment.

For every substantive change, update `CHATGPT_PROJECT_CONTEXT.md` in the same
commit when any of the following changes:

- model architecture, class mapping, weights, or confidence policy;
- dataset source, annotation format, split, conversion, or augmentation;
- server environment, GPU strategy, commands, paths, or scheduler usage;
- inference/postprocessing or MusicXML acceptance rules;
- completed validation, known limitations, blockers, or next steps.

Add a dated entry to its change log. Do not claim server training is ready until
the relevant dataset has passed validation and the Linux/Slurm smoke job has
actually succeeded.

Never commit datasets, generated tiles, experiment runs, model weights, OCR
weights, credentials, tokens, or private server environment files. Commit only
code, templates, documentation, and small test fixtures.
