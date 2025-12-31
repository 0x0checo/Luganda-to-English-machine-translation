# Fine-tuning Multilingual NMT for a Low-Resource, Morphologically Rich Language (Luganda → English)

This project studies **Luganda–English machine translation** under low-resource conditions. We evaluate several baselines (including **Google Translate**) and fine-tune two multilingual pretrained NMT models—**NLLB-600M** and **mBART-50**—on the **SALT Luganda–English** parallel corpus. 

## Dataset

We use the **SALT corpus** (≈ **25,000** Luganda–English sentence pairs), split into:

* Train: **23,497**
* Validation: **496**
* Test: **500** 

## Models & Setup

**Baselines**

* Copy baseline (returns the source as output)
* mBART-50 baseline (uses **Swahili `sw_KE`** as `src_lang` since Luganda is not in pretraining)
* NLLB-600M baseline (`lug_Latn` → `eng_Latn`)
* Google Translate

**Fine-tuning**

* Fine-tuned mBART-50 (training stopped after **2 epochs** due to disk quota limits)
* Fine-tuned NLLB-600M
* Fine-tuned NLLB-600M + **Back-Translation** (train en→lug, synthesize Luganda, then retrain lug→en) 

Evaluation uses **BLEU** on the same test set with consistent decoding settings. 

## Results (BLEU)

| Model                                   |      BLEU |   |
| --------------------------------------- | --------: | - |
| Copy Baseline                           |      2.84 |   |
| Google Translate                        |      9.24 |   |
| mBART-50 Baseline                       |      7.60 |   |
| NLLB-600M Baseline                      |     15.62 |   |
| Fine-tuned mBART-50                     |     25.93 |   |
| Fine-tuned NLLB-600M                    | **35.99** |   |
| Fine-tuned NLLB-600M + Back-Translation |     35.99 |   |

**Key takeaway:** fine-tuning yields large gains, and **NLLB-600M** performs best; back-translation did not further improve BLEU in this setup.

## Acknowledgements

* SALT dataset: Nabende et al. (2023)
* Models: mBART (Liu et al., 2020), NLLB (Team et al., 2022) 
