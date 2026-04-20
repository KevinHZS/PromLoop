# PromLoop
Integrating Pretrained Biological Representations for Identification and Generation of Industrial Microbial Promoters.

## 📁 Datasets

We have constructed high-quality and cross-species promoter datasets covering six representative industrial microorganisms. All promoter sequences are length-normalized to 81 bp (from –60 to +20 bp relative to the transcription start site), covering the core regulatory region. The dataset is strictly deduplicated (CD-HIT, 90% identity) and split into training / validation / test sets with a 7:1:2 ratio, maintaining an approximately 1:1 positive-negative balance.

👉 **All dataset files are available in the [`Datasets/`](./Datasets/) directory**, organized by species (e.g., `Escherichia coli/`, `Bacillus subtilis/`, etc.).

## 🧬 Model Usage

Our fine-tuned promoter identification models for six industrial microbial species are built upon the **LucaOne** biological foundation model. To perform inference with these models (i.e., to identify promoters in your own sequences), please refer to the official **LucaOneTasks** project, which provides a complete pipeline for loading fine-tuned LucaOne models and running predictions.

👉 **Inference Guide**: [https://github.com/LucaOne/LucaOneTasks](https://github.com/LucaOne/LucaOneTasks)

Follow their README to:
- Load the fine-tuned model from Hugging Face (our model IDs are listed below)
- Run promoter identification on your sequences
- Obtain output CSV files similar to the ones we provide in this repository (e.g., `E.coli_prediction_results.csv`)

### Model Zoo

All six fine-tuned models are available on Hugging Face Hub:

| Target Species | Hugging Face Model ID |
| :--- | :--- |
| *Escherichia coli* | [`lucaone-ecoli-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-ecoli-promoter-id) |
| *Bacillus subtilis* | [`lucaone-bsutilis-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-bsubtilis-promoter-id) |
| *Diphtheriae* | [`lucaone-diphtheria-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-diphtheria-promoter-id) |
| *Baumannii* | [`lucaone-baumanii-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-baumanii-promoter-id) |
| *Staphylococcus* | [`lucaone-staphylococcus-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-staphylococcus-promoter-id) |
| *Bradyrhizobium* | [`lucaone-bradyhizobium-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-bradyrhizobium-promoter-id) |


## 📊 Experimental Results – Reproducibility Data

To ensure full transparency and reproducibility of all results reported in our paper, we provide the complete prediction outputs for **five different experimental configurations** on the test sets of all six microbial species.

Each CSV file contains:
- `seq_id`: sequence identifier
- `seq`: 81 bp promoter sequence
- `prob`: predicted probability (0–1) of being a functional promoter
- `label`: ground truth (1 = promoter, 0 = non‑promoter)

👉 All result files are organized in the [`Results/`](./Results/) folder.


These results correspond to the following experiments in our paper:

| Folder | Paper Reference | Description |
| :--- | :--- | :--- |
| `1_lucaone_lucabase_ft50` | Table 2, Table 3, Table 4 | Our proposed method (LucaOne + LucaBase, fine-tuned 50 epochs) |
| `2_dnabert2_lucabase_ft50` | Table 2 | Baseline: DNABert2 embedding + LucaBase |
| `3_onehot_lucabase_ft50` | Table 2 | Baseline: Onehot encoding + LucaBase |
| `4_lucaone_linear_ft50` | Table 3 | Comparison: LucaOne + linear classifier |
| `5_lucaone_lucabase_zero_shot` | Table 4 | Zero‑shot transfer (no fine‑tuning) |

You can directly load any CSV file to reproduce our evaluation metrics (Accuracy, Precision, Recall, F1, MCC, AUC) or to compare with your own models.

