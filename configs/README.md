# Configuration files

- `books.yaml`, `disney.yaml`, `enron.yaml`, `reddit.yaml`, `weibo.yaml` — fixed hyperparameters for each PyGOD benchmark dataset used in the paper supplement.
- `universal_template.yaml` — optional template illustrating universal polarity fields; copy and edit `dataset` / `exp_tag` if you need a custom run.

Training reads **only** the chosen YAML file plus CLI flags (`--seed`, `--num_trial`, etc.). No hidden config paths.
