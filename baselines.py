#!/usr/bin/env python3
"""
SALT Project - Baseline Models Evaluation
Run 4 baseline models on test set: Copy, mBART-50, NLLB-200, Google Translate
"""

import os
import torch
import time
import requests
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import BLEU

def load_test_data():
    """Load test data for evaluation"""
    print("=== LOADING TEST DATA ===")
    
    # Check if files exist
    if not os.path.exists('salt_luganda/test.lu'):
        raise FileNotFoundError("salt_luganda/test.lu not found!")
    if not os.path.exists('salt_luganda/test.en'):
        raise FileNotFoundError("salt_luganda/test.en not found!")
    
    with open('salt_luganda/test.lu', 'r', encoding='utf-8') as f:
        test_lu = [line.strip() for line in f.readlines()]
    with open('salt_luganda/test.en', 'r', encoding='utf-8') as f:
        test_en = [line.strip() for line in f.readlines()]
    
    # Check alignment
    if len(test_lu) != len(test_en):
        raise ValueError(f"Data misalignment: {len(test_lu)} Luganda vs {len(test_en)} English")
    
    print(f"Test data loaded: {len(test_lu)} examples")
    return test_lu, test_en

def copy_baseline(test_lu, test_en):
    """Copy baseline: return source as translation"""
    print("\n=== COPY BASELINE ===")
    
    # Copy baseline: return source as translation
    predictions = test_lu
    references = [[ref] for ref in test_en]
    
    # Calculate BLEU score
    bleu = BLEU()
    score = bleu.corpus_score(predictions, references)
    
    print(f"Copy Baseline BLEU: {score.score:.2f}")
    
    # Show examples
    print("\n=== EXAMPLE TRANSLATIONS ===")
    for i in range(min(3, len(test_lu))):
        print(f"Example {i+1}:")
        print(f"Source:     {test_lu[i]}")
        print(f"Copy:       {predictions[i]}")
        print(f"Reference:  {test_en[i]}")
        print("-" * 60)
    
    return score.score

def mbart_baseline(test_lu, test_en):
    """mBART-50 baseline"""
    print("\n=== mBART-50 BASELINE ===")
    print("Loading mBART-50 model...")
    
    try:
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    except Exception as e:
        print(f"Error loading mBART model: {e}")
        return 0.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    
    # Set languages
    tokenizer.src_lang = "lu_UG"
    tokenizer.tgt_lang = "en_XX"
    
    predictions = []
    references = []
    
    forced_bos = tokenizer.lang_code_to_id["en_XX"]  # 强制输出英语
    
    print("Translating with mBART-50...")
    for i, source in enumerate(test_lu):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(test_lu)}: {source[:40]}...")
        
        try:
            inputs = tokenizer(source, return_tensors="pt", max_length=128, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    forced_bos_token_id=forced_bos
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(translation)
            references.append([test_en[i]])
        except Exception as e:
            print(f"Error translating example {i+1}: {e}")
            predictions.append("Translation failed")
            references.append([test_en[i]])
    
    # Calculate BLEU
    bleu = BLEU()
    score = bleu.corpus_score(predictions, references)
    
    print(f"mBART-50 Baseline BLEU: {score.score:.2f}")
    
    # Show examples
    print("\n=== EXAMPLE TRANSLATIONS ===")
    for i in range(min(3, len(test_lu))):
        print(f"Example {i+1}:")
        print(f"Source:     {test_lu[i]}")
        print(f"mBART-50:   {predictions[i]}")
        print(f"Reference:  {test_en[i]}")
        print("-" * 60)
    
    return score.score

def nllb_baseline(test_lu, test_en):
    """NLLB-200 baseline"""
    print("\n=== NLLB-200 BASELINE ===")
    print("Loading NLLB-200 model...")
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=True)
    except Exception as e:
        print(f"Error loading NLLB model: {e}")
        return 0.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    
    # Set languages
    tokenizer.src_lang = "lu_Latn"
    tokenizer.tgt_lang = "en_Latn"
    
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")  # ✅ 获取对应 token id
    
    predictions = []
    references = []
    
    print("Translating with NLLB-200...")
    for i, source in enumerate(test_lu):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(test_lu)}: {source[:40]}...")
        
        try:
            inputs = tokenizer(source, return_tensors="pt", max_length=128, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128, 
                    num_beams=4,
                    forced_bos_token_id=forced_bos
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(translation)
            references.append([test_en[i]])
        except Exception as e:
            print(f"Error translating example {i+1}: {e}")
            predictions.append("Translation failed")
            references.append([test_en[i]])
    
    # Calculate BLEU
    bleu = BLEU()
    score = bleu.corpus_score(predictions, references)
    
    print(f"NLLB-200 Baseline BLEU: {score.score:.2f}")
    
    # Show examples
    print("\n=== EXAMPLE TRANSLATIONS ===")
    for i in range(min(3, len(test_lu))):
        print(f"Example {i+1}:")
        print(f"Source:     {test_lu[i]}")
        print(f"NLLB-200:   {predictions[i]}")
        print(f"Reference:  {test_en[i]}")
        print("-" * 60)
    
    return score.score

def translate_with_google(text, source_lang="lg", target_lang="en"):
    """Translate text using Google Translate API"""
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': source_lang,  # lg = Luganda
        'tl': target_lang,  # en = English
        'dt': 't',
        'q': text
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        result = response.json()
        if result and len(result) > 0 and len(result[0]) > 0:
            return result[0][0][0]
        else:
            return "Translation failed"
    except Exception as e:
        print(f"Error translating '{text[:30]}...': {e}")
        return "Translation failed"

def google_translate_baseline(test_lu, test_en):
    """Google Translate baseline"""
    print("\n=== GOOGLE TRANSLATE BASELINE ===")
    print("Note: This will take 20-30 minutes due to API rate limits")
    print(f"Translating {len(test_lu)} examples...")
    
    predictions = []
    references = []
    
    print("Translating with Google Translate...")
    for i, source in enumerate(test_lu):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(test_lu)}: {source[:40]}...")
        
        translation = translate_with_google(source)
        predictions.append(translation)
        references.append([test_en[i]])
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
        
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{len(test_lu)} translations")
    
    # Calculate BLEU score
    bleu = BLEU()
    score = bleu.corpus_score(predictions, references)
    
    print(f"Google Translate BLEU: {score.score:.2f}")
    
    # Show examples
    print("\n=== EXAMPLE TRANSLATIONS ===")
    for i in range(min(3, len(test_lu))):
        print(f"Example {i+1}:")
        print(f"Source:     {test_lu[i]}")
        print(f"Google:     {predictions[i]}")
        print(f"Reference:  {test_en[i]}")
        print("-" * 60)
    
    return score.score

def main():
    """Main function to run all baselines"""
    print("=== SALT PROJECT - BASELINE EVALUATION ===")
    print("Running 4 baseline models on FULL test set (500 examples)")
    
    try:
        # Load test data
        test_lu, test_en = load_test_data()
        
        # Run baselines on ALL 500 examples
        copy_score = copy_baseline(test_lu, test_en)
        mbart_score = mbart_baseline(test_lu, test_en)
        nllb_score = nllb_baseline(test_lu, test_en)
        google_score = google_translate_baseline(test_lu, test_en)
        
        # Final results
        print(f"\n=== FINAL BASELINE RESULTS (500 examples) ===")
        print(f"Copy Baseline:      {copy_score:.2f} BLEU")
        print(f"mBART-50 Baseline:  {mbart_score:.2f} BLEU")
        print(f"NLLB-200 Baseline:  {nllb_score:.2f} BLEU")
        print(f"Google Translate:   {google_score:.2f} BLEU")
        
        # Save results to file
        with open('baseline_results.txt', 'w') as f:
            f.write("SALT Project - Baseline Results (500 examples)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Copy Baseline:      {copy_score:.2f} BLEU\n")
            f.write(f"mBART-50 Baseline:  {mbart_score:.2f} BLEU\n")
            f.write(f"NLLB-200 Baseline:  {nllb_score:.2f} BLEU\n")
            f.write(f"Google Translate:   {google_score:.2f} BLEU\n")
        
        print(f"\nResults saved to baseline_results.txt")
        print("Baseline evaluation completed!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())