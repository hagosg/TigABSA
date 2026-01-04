# TigABSA

This repository contains the datasets, preprocessing pipelines, model implementations, training scripts, and evaluation utilities for the paper: TigABSA: A Knowledge-Guided Cross-Lingual Framework for Aspect-Based Sentiment Analysis in Low-Resource Tigrigna, Submitted at IEEE Transactions on Audio, Speech and Language Processing (IEEE). 
The goal of TigABSA is to enable aspect based sentiment analysis for low-resource Tigrigna through cross-lingual transfer learning, affective modeling, and parameter-efficient fine-tuning of large language models.

🧠 Overview of  the Model Architecture

TigABSA is a scalable cross-lingual framework that unifies multilingual pretrained backbones, sentiment-guided adaptation, and parameter-efficient fine-tuning to produce saspect based sentiment analysis in low-resource Tigrigna. The approach exploits cross-lingual transfer from high-resource languages (English, Amharic) while preserving semantic fidelity and emotional coherence.

Key features include:

- Multilingual backbones (XLM-R, mT5, LLaMA-3.1, EthioLLM-7B, and Qwen2.5)

- Hybrid Sentiment Fusion Module (additive / gated / concatenation)

- Low-Rank Adaptation (LoRA) for efficient fine-tuning

- Balanced evaluation with ATE, ASC, Aspect F1, Percision, Recall, and Acc

- Fully reproducible training, evaluation, and analysis pipelines

  
🔤 Data Preprocessing & Tokenization

python preprocessing/sentencepiece_train.py \
  -- input data/raw \
  -- vocab_size 32000 \
  -- model_prefix spm_tigABSA
  
  python preprocessing/build_multilingual_dataset.py

### TigABSA-800 Native Data


| Domain      | Sentences | Avg. Aspects/Sent | Positive | Negative | Neutral | Split (Train/Dev/Test) | Usage     |
|-------------|-----------|-------------------|----------|----------|---------|------------------------|-----------|
| Restaurant  | 250       | 3.1               | 46%      | 38%      | 14%     | 187 / 31 / 32          | Dev/Test  |
| Telecom     | 200       | 2.9               | 32%      | 54%      | 12%     | 150 / 25 / 25          | Dev/Test  |
| Politics    | 180       | 2.6               | 28%      | 61%      | 9%      | 135 / 23 / 22          | Dev/Test  |
| Product     | 170       | 2.7               | 41%      | 47%      | 10%     | 128 / 21 / 21          | Dev/Test  |
| **Total**   | **800**   | **2.84**          | **37%**  | **50%**  | **11%** | **600 / 100 / 100**    | Train/Dev/Test |

*Note:* Only the **Dev + Test** split (200 sentences) is used in zero-resource experiments.  
Supervised training is performed exclusively on auxiliary cross-lingual corpora.

  
### Cross-Lingual Data Sources for Build Multilingual Dataset
The selected three cross-lingual high-resource languages(English, Amharic, and Arabic), chosen for typological or cultural proximity are availiable on the following:
1.	https://github.com/ybai-nlp/MCLAS
2.	https://github.com/google-deepmind/rc-data
3.	https://github.com/Ethanscuter/gigaword 
5.	https://github.com/IsraelAbebe/An-Amharic-News-Text-classification-Dataset
   


## Usage

* Istallation

conda env create -f environment.yml
conda activate tigABSA

OR 
```bash
git clone https://github.com/yourusername/tigABSA.git
cd tigabsa
pip install -e .
```

🚀 Training

python training/train_tigabsa.py \
  --model TigABSA \
  --fusion hybrid \
  --lora_rank 8 \
  --epochs 5



📊 Evaluation


- ROUGE-L

- BERTScore

- Sentiment Preservation Rate (SPR)

- Emotional Consistency Index (ECI)

python test_tigABSA.py



### Configuration



| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 16 | Effective after gradient accumulation |
| **Learning Rate** | 2 × 10⁻⁵ | AdamW optimizer |
| **Epochs** | 8 | Early stopping with patience 2 |
| **Max Input Length** | 512 tokens | Summarization truncation |
| **Sentiment Weight (β)** | 0.4 | Balancing polarity regularization |
| **Fusion λ** | 0.7 | Encoder–sentiment tradeoff |
| **Hardware** | 4 × A6000 GPUs | FP16 precision training |
| **Training Time** | 5–14 h | Depending on model |

*Note:* Configuration for TigABSA model training across different architectures. All experiments used linear learning rate warm-up over 10% of training steps.


📚 Citation

@article{Gebremeskel2026TigABSA,
  title={TigABSA: A Knowledge-Guided Cross-Lingual Framework for Aspect-Based Sentiment Analysis in Low-Resource Tigrigna},
  author={ Hagos Gebremedhin Gebremeskel, Chong Feng, and et. al.},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2026}
}

📬 Contact

Hagos Gebremedhin Gebremeskel,
Beijing Institute of Technology
📧 hagosg81@bit.edu.cn
