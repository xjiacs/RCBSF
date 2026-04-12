
# Contract Template & Risk Dimension Generation Script

## 1. Configure OpenAI API Key

The script calls the API via the `OpenAI()` client. Please set the key before running:

### macOS / Linux

```bash
export OPENAI_API_KEY="your_API_Key"
````

### Windows PowerShell

```powershell
setx OPENAI_API_KEY "your_API_Key"
```

---

## 2. Directory Structure Requirements

### 2.1 Input Directory (`input_dir`)

* Reads **all files** under `input_dir`

Example:

```
data/contracts/
├── contract_001.txt
├── contract_002.txt
└── NDA_sample.txt
```

### 2.2 Output Directories

You need to provide two output directories (the script will create them automatically):

* `templates_out_dir`: Stage 1 output (templates)
* `enriched_out_dir`: Stage 2 output (templates + risk categories)

Example:

```
out/
├── stage1_templates/
│   ├── Software_License_01.json
│   └── Service_Contract_01.json
└── stage2_enriched/
    ├── Software_License_01.json
    └── Service_Contract_01.json
```

> Note: Output filenames are generated using **category + incremental index** to avoid overwriting.

---

## 3. Command-Line Parameters (CLI)

### 3.1 Required Parameters

| Parameter             | Type | Description                                     |
| --------------------- | ---: | ----------------------------------------------- |
| `--input_dir`         | Path | Input contracts directory                       |
| `--templates_out_dir` | Path | Stage 1 output directory (template JSON)        |
| `--enriched_out_dir`  | Path | Stage 2 output directory (template + risk JSON) |

### 3.2 Model Parameters

| Parameter          | Default | Description                                                      |
| ------------------ | ------- | ---------------------------------------------------------------- |
| `--template_model` | `gpt-5` | Used for **category identification** and **template generation** |
| `--risk_model`     | `gpt-5` | Used for **risk category generation**                            |

---

## 4. Run Example

```bash
python process_data.py \
  --input_dir ./data/contracts \
  --templates_out_dir ./out/stage1_templates \
  --enriched_out_dir ./out/stage2_enriched \
  --template_model gpt-5 \
  --risk_model gpt-5 \
```

```
```
