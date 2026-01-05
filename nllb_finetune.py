def main():
    
    # 1. Load pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    
    tokenizer.src_lang = "lug_Latn"
    tokenizer.tgt_lang = "eng_Latn"
    
    # Load TRAINING data (23,947 examples)
    (train_lu, train_en), (val_lu, val_en), (test_lu, test_en) = load_data()
    
    # 3. Prepare datasets for training
    train_dataset = prepare_dataset(train_lu, train_en, tokenizer)
    val_dataset = prepare_dataset(val_lu, val_en, tokenizer)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 4.  Set up training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir="/proj/.../nllb_finetuned_luganda",
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=3e-5,                    # Learning rate
        per_device_train_batch_size=8,        # Batch size
        num_train_epochs=3,                    # Epochs
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
    )
    
    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,          #  Training data
        eval_dataset=val_dataset,             #  Validation data
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    # 6. TRAIN THE MODEL
    trainer.train()                           #  Trains on 23,947 examples
    
    # 7. Save finetuned model
    trainer.save_model("/proj/.../nllb_finetuned_luganda/final")
    
    # 8. Evaluate on test set
    test_score, predictions = evaluate_on_test(model, tokenizer, test_lu, test_en)
    
    return test_score
