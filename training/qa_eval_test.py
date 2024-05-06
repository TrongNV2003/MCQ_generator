import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

from dataset import QAEvalDataset
from trainer import Trainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./test-bert-base-cased")
    parser.add_argument("--log_file", type=str, default="test_qa_log.csv")
    parser.add_argument("--qa_eval_model", type=str, default="vinai/phobert-base")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.qa_eval_model)

    train_set = QAEvalDataset(
        csv_file='sample/evaluation/eval_qa_train.csv',
        max_length=args.max_length,
        tokenizer=tokenizer
    )

    test_set = QAEvalDataset(
        csv_file='sample/test/test.csv',
        max_length=args.max_length,
        tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.qa_eval_model)
    
    trainer = Trainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=1,  # Chỉ cần 1 epoch cho việc test
        learning_rate=0,  # Không cần learning rate khi test
        model=model,
        tokenizer=tokenizer,
        pin_memory=args.pin_memory,
        save_dir=args.save_dir,  # Không cần lưu trữ khi test
        train_batch_size=args.train_batch_size,
        train_set=train_set,  # Không cần tập dữ liệu huấn luyện khi test
        valid_batch_size=args.test_batch_size,  # Sử dụng test_batch_size cho valid_batch_size
        log_file=args.log_file,  # Sử dụng log_file cho việc lưu kết quả
        valid_set=test_set,  # Sử dụng test_set cho valid_set
        evaluate_on_accuracy=True  # Đánh giá dựa trên độ chính xác khi test
    )

    trainer.train()
