import os
import torch
from tqdm import tqdm
from src.tracker import MetricTracker
from src.utils import dict_to_device


class Trainer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        logger=None,
        num_beams=5,
        top_p=1,
        top_k=0,
        temperature=1,
        *arg,
        **kwarg,
        ):
        
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler
        # self.eval_func = RougeScore()
        self.tracker = MetricTracker()
        self.logger = logger

        # generation arguments
        self.num_beams = num_beams
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def train_step(self, batch_data, index):
        outputs = self.model(**batch_data)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        n = preds.shape[0]
        self.tracker.update("train/loss", loss / n, n)
        return loss

    # def valid_step(self, batch_data, index):
    #     generated_tokens = self.model.generate(
    #         input_ids=batch_data["input_ids"],
    #         attention_mask=batch_data["attention_mask"],
    #         max_length=MAX_TARGET_LEN,
    #         num_beams=self.num_beams,
    #         # do_sample=False,
    #         # top_p=self.top_p,
    #         # top_k=self.top_k,
    #         # temperature=self.temperature,
    #     )
        # generated_tokens = postprocess_func(
        #     self.tokenizer.batch_decode(
        #         generated_tokens, skip_special_tokens=True
        #     )
        # )
        # return generated_tokens, batch_data[SUMMARY_COL]

    def log(self, record):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record)
        return

    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.tracker.reset(keys=["train/loss"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss = self.train_step(batch_data, step)
            self.progress_bar.set_postfix({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        self.progress_bar.close()
        return

    # @torch.no_grad()
    # def valid_one_epoch(self):
    #     self.model.eval()
    #     self.progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
    #     self.tracker.reset(keys=["valid/rouge-1", "valid/rouge-2", "valid/rouge-l", "valid/rouge-total"])

    #     prediction_list = []
    #     ground_truth_list = []

    #     for step, batch_data in enumerate(self.progress_bar, start=1):
    #         batch_data = dict_to_device(batch_data, self.device)
    #         generated_tokens, ground_truth = self.valid_step(batch_data, step)
    #         prediction_list.extend(generated_tokens)
    #         ground_truth_list.extend(ground_truth)

    #     scores = self.eval_func.evaluate(prediction_list, ground_truth_list)
    #     print(scores)

    #     rouge_total = 0
    #     for rouge in ["rouge-1", "rouge-2", "rouge-l"]:
    #         self.tracker.update(f"valid/{rouge}", scores[rouge])
    #         rouge_total += scores[rouge]
    #     self.tracker.update(f"valid/rouge-total", rouge_total)

    #     self.log({"epoch": self.cur_ep, **self.tracker.result()})
    #     self.progress_bar.close()
    #     self.model.save_pretrained(
    #         os.path.join(
    #             CHECKPOINT_DIR,
    #             f"epoch={self.cur_ep}_rouge-total={self.tracker.result().get('valid/rouge-total', 0)}"
    #         )
    #     )
    #     return

    def fit(self, epoch):
        self.model.to(self.device)
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            # self.valid_one_epoch()
            self.model.save_pretrained(
                os.path.join(
                    "checkpoint",
                    f"epoch={self.cur_ep}_loss={self.tracker.result().get('train/loss', 0)}"
            )
        )
        return
