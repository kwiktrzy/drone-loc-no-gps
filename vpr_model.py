import os
import pytorch_lightning as pl
import torch
import pandas as pd
from torch.optim import lr_scheduler
import torch.nn.functional as F
from pathlib import Path

from models import abstract, utils
import numpy as np
from datetime import datetime

from models.utils.batch_attention import visualize_batch_attention


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        backbone_config={},
        # ---- Aggregator
        agg_arch="ConvAP",
        agg_config={},
        # ---- Train hyperparameters
        lr=0.03,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000,
        },
        # ----- Loss
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = (
            []
        )  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = abstract.get_backbone(backbone_arch, backbone_config)
        self.aggregator = abstract.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        # ----------------------------------

        self.is_loss_debug = True
        self.is_return_attention = True

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x, attn_map = self.aggregator(x)
        if self.is_return_attention:
            return x, attn_map
        return x

    # configure the optimizer
    def configure_optimizers(self):
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )

        if self.lr_sched.lower() == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_sched_args["milestones"],
                gamma=self.lr_sched_args["gamma"],
            )
        elif self.lr_sched.lower() == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, self.lr_sched_args["T_max"]
            )
        elif self.lr_sched.lower() == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args["start_factor"],
                end_factor=self.lr_sched_args["end_factor"],
                total_iters=self.lr_sched_args["total_iters"],
            )

        return [optimizer], [scheduler]

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        if self.is_loss_debug:
            return loss, miner_outputs
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        if self.is_return_attention:
            descriptors, attn_maps = self(images)
        else:
            descriptors = self(
                images
            )  # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError("NaNs in descriptors")

        loss, miner_outputs = self.loss_function(descriptors, labels)

        if self.is_loss_debug:
            if (
                self.current_epoch > 30
                and loss.item() > 0.04
                and miner_outputs is not None
            ):

                debug_root_dir = "debug_vis"
                os.makedirs(debug_root_dir, exist_ok=True)

                debug_file_path = os.path.join(debug_root_dir, "debug_hard_mining.csv")

                with torch.no_grad():
                    probs = F.softmax(descriptors, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)

                    norms = torch.norm(descriptors, p=2, dim=1)
                    stds = torch.std(descriptors, dim=1)

                    desc_norm = F.normalize(descriptors, p=2, dim=1)
                    pairs_to_log = []

                    if len(miner_outputs) == 3:
                        anchors, positives, negatives = miner_outputs
                        for i in range(len(anchors)):
                            pairs_to_log.append((anchors[i], positives[i], "HARD_POS"))
                            pairs_to_log.append((anchors[i], negatives[i], "HARD_NEG"))

                    elif len(miner_outputs) == 4:
                        a_pos, p, a_neg, n = miner_outputs
                        for i in range(len(a_pos)):
                            pairs_to_log.append((a_pos[i], p[i], "HARD_POS"))
                        for i in range(len(a_neg)):
                            pairs_to_log.append((a_neg[i], n[i], "HARD_NEG"))

                    csv_data = []
                    for idx_a_t, idx_b_t, pair_type in pairs_to_log:
                        idx_a = idx_a_t.item()
                        idx_b = idx_b_t.item()

                        sim = torch.dot(desc_norm[idx_a], desc_norm[idx_b]).item()

                        csv_data.append(
                            {
                                "epoch": self.current_epoch,
                                "batch_idx": batch_idx,
                                "loss": round(loss.item(), 5),
                                "pair_type": pair_type,
                                "similarity": round(sim, 4),
                                "imgA_idx": idx_a,
                                "imgA_label": labels[idx_a].item(),
                                "imgA_entropy": round(entropy[idx_a].item(), 4),
                                "imgA_norm": round(norms[idx_a].item(), 4),
                                "imgA_std": round(stds[idx_a].item(), 4),
                                "imgB_idx": idx_b,
                                "imgB_label": labels[idx_b].item(),
                                "imgB_entropy": round(entropy[idx_b].item(), 4),
                                "imgB_norm": round(norms[idx_b].item(), 4),
                            }
                        )
                    if csv_data:
                        df = pd.DataFrame(csv_data)
                        df.to_csv(
                            debug_file_path,
                            mode="a",
                            header=not os.path.exists(debug_file_path),
                            index=False,
                        )

                    if self.is_return_attention and attn_maps is not None:
                        visualize_batch_attention(
                            images_tensor=images,
                            attn_maps=attn_maps,
                            labels=labels,
                            batch_idx=batch_idx,
                            epoch=self.current_epoch,
                            output_dir="debug_vis",
                            limit=None,
                        )
                    else:
                        print(
                            "Warning: Loss spiked but attention maps strictly not returned by model."
                        )

        self.log("loss", loss.item(), logger=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # TODO LZ: Verify it
    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)

        # if we pass just one validation DataLoader the dataloader_idx is always None, which breaks the code...
        idx_to_use = dataloader_idx if dataloader_idx is not None else 0

        self.val_outputs[idx_to_use].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [
            [] for _ in range(len(self.trainer.datamodule.val_datasets))
        ]

    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        # if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
        #     val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.val_set_names, dm.val_datasets)
        ):

            short_val_name = Path(val_set_name).stem
            feats = torch.concat(val_step_outputs[i], dim=0)

            if "Shandan" in short_val_name:
                num_references = len(val_dataset.db_image_paths)
                positives = val_dataset.get_positives()

                variant = (
                    "v1"
                    if "v1" in short_val_name
                    else ("v2" if "v2" in short_val_name else "base")
                )

                print(f"\n Shandan ({variant}): {short_val_name}")
                print(f" Queries: {len(positives)}")
                print(f" References: {num_references}")

                cnts = np.array([len(p) for p in positives])
                zero_pos = (cnts == 0).sum()
                print(f" Q with 0 positives: {zero_pos} / {len(cnts)}")
                if len(cnts) > 0:
                    print(
                        " positives per Q (median/mean/90p):",
                        f"{np.median(cnts):.1f} / {cnts.mean():.1f} / {np.quantile(cnts, 0.9):.1f}",
                    )
            elif "Changjiang-23" in short_val_name:
                num_references = len(val_dataset.db_image_paths)
                positives = val_dataset.get_positives()

                variant = (
                    "v1"
                    if "v1" in short_val_name
                    else ("v2" if "v2" in short_val_name else "base")
                )

                print(f"\n Changjiang-23 ({variant}): {short_val_name}")
                print(f" Queries: {len(positives)}")
                print(f" References: {num_references}")

                cnts = np.array([len(p) for p in positives])
                zero_pos = (cnts == 0).sum()
                print(f" Q with 0 positives: {zero_pos} / {len(cnts)}")
                if len(cnts) > 0:
                    print(
                        " positives per Q (median/mean/90p):",
                        f"{np.median(cnts):.1f} / {cnts.mean():.1f} / {np.quantile(cnts, 0.9):.1f}",
                    )
            elif "Shandong-1" in short_val_name:
                num_references = len(val_dataset.db_image_paths)
                positives = val_dataset.get_positives()
                print(f"Queries: {len(positives)}")
                print(f"References: {num_references}")
            # elif 'msls' in val_set_name:
            #     # split to ref and queries
            #     num_references = val_dataset.num_references
            # positives = val_dataset.pIdx
            else:
                print(f"Please implement validation_epoch_end for {val_set_name}")
                raise NotImplemented

            r_list = feats[:num_references]
            q_list = feats[num_references:]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=short_val_name,
                faiss_gpu=self.faiss_gpu,
            )
            predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=False,
                dataset_name=short_val_name,
                faiss_gpu=self.faiss_gpu,
                testing=True,
            )
            self.val_debug_results(
                current_val_dataset=val_dataset,
                q_list=q_list,
                predictions=predictions,
                positives=positives,
                num_references=num_references,
                short_val_name=short_val_name,
            )
            del r_list, q_list, feats, num_references, positives

            metric_name_r1 = f"{short_val_name}/R1"
            metric_name_r5 = f"{short_val_name}/R5"
            print(
                f"Metrics: '{metric_name_r1}' = {pitts_dict[1]:.4f}, '{metric_name_r5}' = {pitts_dict[5]:.4f}"
            )

            self.log(f"{short_val_name}/R1", pitts_dict[1], prog_bar=False, logger=True)
            self.log(f"{short_val_name}/R5", pitts_dict[5], prog_bar=False, logger=True)
            self.log(
                f"{short_val_name}/R10", pitts_dict[10], prog_bar=False, logger=True
            )
        print("\n\n")

        # reset the outputs list
        self.val_outputs = []

    def val_debug_results(
        self,
        current_val_dataset,
        q_list,
        predictions,
        positives,
        num_references,
        short_val_name,
    ):
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        debug_file_path = f"debug_results_{short_val_name}.txt"
        with open(debug_file_path, "a") as f:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            f.write(f"Results for {short_val_name} \n")
            f.write(f"\n Date: {timestamp_str}")
            f.write("=" * 50 + "\n")
            for q_idx in range(len(q_list)):
                real_q_idx_in_dataset = (
                    num_references + q_idx
                )  # cuz queries are after db embeddings
                q_path = current_val_dataset.images[real_q_idx_in_dataset]
                top_k_indices = predictions[q_idx][:5]
                f.write(f"\nQUERY: {q_path}\n")
                f.write(f"Ground Truth Indices (z get_positives): {positives[q_idx]}\n")

                f.write("Predictions (Top 5):\n")
                for rank, db_idx in enumerate(top_k_indices):
                    pred_path = current_val_dataset.images[db_idx]

                    is_correct = db_idx in positives[q_idx]
                    marker = "[HIT]  " if is_correct else "[MISS] "

                    f.write(
                        f"  {rank+1}. {marker} Index: {db_idx} | Path: {pred_path}\n"
                    )
