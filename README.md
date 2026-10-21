# DSN 2026 language ID

Fine-tunes `facebook/mms-300m` (first 12 layers) into a 13-class classifier for 12 Nigerian languages plus
`other`, then exports, checks, benchmarks and submits it.

## Run it

**Kaggle:** GPU on, Internet on, `HF_TOKEN` under Add-ons > Secrets. Upload this folder as a dataset (so
`dsn_lid/` is inside it), attach it to a new notebook, and import `DSN_LID_MMS300m.ipynb`.

**Locally:** `pip install -r requirements.txt`, log in with `huggingface-cli login` (or set `HF_TOKEN`), and
open `DSN_LID_MMS300m.ipynb` from this folder. Files go to `./workspace`.

Set `hub_username` and `run_name` in the configuration cell. Use a new `run_name` for every submission.

## Layout

| Module | Contents |
|---|---|
| `config.py` | Labels, limits, `ModelConfig`, `TrainingConfig`, `AugmentationConfig` |
| `environment.py` | Paths for Kaggle or local, Hugging Face login, device, dataset download |
| `data.py` | Manifests, audio loading, balanced sampling, batching, data audit |
| `augment.py` | Telephone, mu-law, noise, reverb and speed augmentation |
| `modeling.py` | Feature extractor and truncated MMS classifier |
| `training.py` | `Trainer` with resume and validation-based checkpoint selection |
| `evaluation.py`, `metrics.py` | Inference, language-weighted F1, efficiency and final score, bias tuning |
| `export.py` | fp16 export, model card, pre-submission checks |
| `benchmark.py` | Speed and memory in a fresh process (`python -m dsn_lid.benchmark DIR --device cpu`) |
| `submission.py` | Push to Hugging Face, write `submission.zip` |

## Before the first real run

- Not run on a GPU or on the real dataset. The whole pipeline was exercised on a tiny random model with
  synthetic audio (training, resume, evaluation, calibration, export, checks, benchmark, submission zip).
- `run_official_checker` assumes `check_model_folder.py` takes the model folder as its only argument.
- MMS is CC BY-NC 4.0, and the generated model card says so.
