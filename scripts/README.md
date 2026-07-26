# Running the Translation Pipeline

## 1. Check GPU availability

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
```

## 2. Clone the repository

```bash
git clone https://github.com/dianamikova/cs-en-idioms.git
cd cs-en-idioms
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Authenticate with Hugging Face

```python
from huggingface_hub import login

login()  # Paste your Hugging Face access token when prompted.
```

## 5. Run the translation script

```bash
python scripts/translate.py --limit 5
```

The `--limit` argument is optional and can be adjusted or omitted to process the entire dataset.
