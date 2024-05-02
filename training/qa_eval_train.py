import argparse
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
import pandas as pd

from dataset import QAEvalDataset
from trainer import Trainer

spacy.prefer_gpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--qa_eval_model", type=str, default="vinai/phobert-base")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./bert-base-cased-qa-evaluator")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--log_file", type=str, default="evaluating_log.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.qa_eval_model)
    
    train_set = QAEvalDataset(
        csv_file='sample/evaluation/eval_qa_train.csv',
        max_length=args.max_length,
        tokenizer=tokenizer
    )

    valid_set = QAEvalDataset(
        csv_file='sample/evaluation/eval_qa_valid.csv',
        max_length=args.max_length,
        tokenizer=tokenizer
    )
    
    log_file = 'evaluating_log.csv'

    model = AutoModelForSequenceClassification.from_pretrained(args.qa_eval_model)
    trainer = Trainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        train_batch_size=args.train_batch_size,
        train_set=train_set,
        valid_batch_size=args.valid_batch_size,
        valid_set=valid_set,
        log_file=log_file,
        evaluate_on_accuracy=True
    )
    trainer.train()
