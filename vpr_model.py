import os
import math
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

        self._shape_debug_printed = False

        self.debug_config = {
            "min_epoch": 35,
            "loss_threshold": 0.043,
            "top_k_triplets": 20,
            "save_attention": True,
            "save_triplet_images": True,
        }
        if hasattr(self.loss_fn, "distance") and self.loss_fn.distance is not None:
            print("LOSS DIST:", type(self.loss_fn.distance), "inv:", getattr(self.loss_fn.distance, "is_inverted", "Unknown"))
        else:
            print("LOSS DIST: NO DISTANCE")

        if self.miner is not None and hasattr(self.miner, "distance") and self.miner.distance is not None:
            print("MINER DIST:", type(self.miner.distance), "inv:", getattr(self.miner.distance, "is_inverted", "Unknown"))
        else:
            print("MINER DIST: NO MINER")

    # the forward pass of the lightning model
    # def forward(self, x):
    #     x = self.backbone(x)
    #     x, attn_map = self.aggregator(x)
    #     if self.is_return_attention:
    #         return x, attn_map
    #     return x

    def forward(self, x):
        do_print = (not self._shape_debug_printed)

        if do_print:
            print("training mode:", self.training)
            print("input shape:", x.shape)

        x = self.backbone(x)

        if do_print:
            print("backbone feat shape:", x.shape)

        x, attn_map = self.aggregator(x)

        if do_print:
            print("agg output shape:", x.shape)
            print("attn_map shape:", attn_map.shape)
            self._shape_debug_printed = True

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
        elif self.optimizer.lower() in ["adamw", "adam"]:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not found")

        if self.lr_sched.lower() == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_sched_args["milestones"],
                gamma=self.lr_sched_args["gamma"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif self.lr_sched.lower() == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_sched_args["T_max"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif self.lr_sched.lower() == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args["start_factor"],
                end_factor=self.lr_sched_args["end_factor"],
                total_iters=self.lr_sched_args["total_iters"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.lr_sched.lower() == "cosine_step_eta_min":
            total_steps = self._get_total_steps()
            eta_min = float(self.lr_sched_args.get("eta_min", 0.0))

            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=eta_min,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.lr_sched.lower() in ["cosine_warm_restarts", "sgdr"]:
            steps_per_epoch = self._get_steps_per_epoch()

            T_0_epochs = int(self.lr_sched_args["T_0_epochs"])
            T_mult = int(self.lr_sched_args.get("T_mult", 2))
            eta_min = float(self.lr_sched_args.get("eta_min", 0.0))

            T_0_steps = max(1, T_0_epochs * steps_per_epoch)

            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0_steps,
                T_mult=T_mult,
                eta_min=eta_min,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        else:
            raise ValueError(f"LR scheduler {self.lr_sched} not found")

    def _get_total_steps(self):
        if "total_steps" in self.lr_sched_args:
            return int(self.lr_sched_args["total_steps"])

        if self.trainer is None:
            raise RuntimeError("Trainer is not attached yet; cannot infer total_steps.")

        return int(self.trainer.estimated_stepping_batches)

    
    def _get_steps_per_epoch(self):
        total_steps = self._get_total_steps()

        if self.trainer is None or self.trainer.max_epochs is None:
            raise RuntimeError("Trainer or max_epochs unavailable; cannot infer steps_per_epoch.")

        return max(1, math.ceil(total_steps / int(self.trainer.max_epochs)))

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    #     # warm up lr
    #     optimizer.step(closure=optimizer_closure)
    #     self.lr_schedulers().step()

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels, domains=None):
        """
        Compute the metric-learning loss for the current batch.
    
        Why this method exists:
        - it keeps mining and loss computation in one place
        - it logs how many examples are actually mined
        - it optionally adds domain-aware diagnostics for cross-domain training
    
        Important:
        The mined counts below tell us how much supervision the miner is producing,
        but they do not tell us whether those examples are useful or noisy.
        For that reason, similarity statistics are logged separately in
        `log_basic_mining_stats(...)`.
    
        Example:
        - many mined negatives + high negative similarity may indicate difficult or noisy hard negatives
        - few mined examples may indicate that the miner became too strict
        """
        
        miner_outputs = None
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)

            if len(miner_outputs) == 4:
                a_pos, p, a_neg, n = miner_outputs
                self.log("mined_pos_pairs", float(a_pos.numel()), prog_bar=True, logger=True)
                self.log("mined_neg_pairs", float(a_neg.numel()), prog_bar=True, logger=True)
            elif len(miner_outputs) == 3:
                a, p, n = miner_outputs
                self.log("mined_triplets", float(a.numel()), prog_bar=True, logger=True)


            self.log_basic_mining_stats(descriptors, miner_outputs)
            if domains is not None:
                self.log_domain_aware_mining(descriptors, domains, miner_outputs)
            
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
        domains = None  

        if len(batch) == 2:
            places, labels = batch
        elif len(batch) == 3:
            places, labels, domains = batch
        else:
            raise ValueError(f"Unexpected batch structure of length {len(batch)}")  

        BS, N, ch, h, w = places.shape  

        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)    

        if domains is not None:
            domains = domains.view(-1).to(labels.device)    

        attn_maps = None
        if self.is_return_attention:
            descriptors, attn_maps = self(images)
        else:
            descriptors = self(images)  

        if torch.isnan(descriptors).any():
            raise ValueError("NaNs in descriptors") 
        
        loss_out = self.loss_function(descriptors, labels, domains=domains) 

        if self.is_loss_debug:
            loss, miner_outputs = loss_out
            self.debug_step(
                descriptors, labels, images, loss, miner_outputs, attn_maps, batch_idx
            )
        else:
            loss = loss_out 

        self.log("loss", loss.item(), logger=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, logger=True)
        self.batch_acc = []

    # TODO LZ: Verify it
    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch

        if places.dim() == 5:
            BS, N, ch, h, w = places.shape
            images = places.view(BS * N, ch, h, w)
        elif places.dim() == 4:
            BS, ch, h, w = places.shape
            images = places
        else:
            raise ValueError(f"Unexpected places shape: {places.shape}")

        # forward (spójne z training_step - przepuszczamy spłaszczony batch)
        if self.is_return_attention:
            descriptors, _ = self(images)
        else:
            descriptors = self(images)

        if torch.isnan(descriptors).any():
            raise ValueError("NaNs in descriptors (val)")

        # TUTAJ ZMIANA: Usunięto .mean(dim=1), zachowujemy kształt (BS * N, D)
        descriptors_cpu = descriptors.detach().cpu()

        idx_to_use = 0 if dataloader_idx is None else dataloader_idx
        while len(self.val_outputs) <= idx_to_use:
            self.val_outputs.append([])

        self.val_outputs[idx_to_use].append(descriptors_cpu)
        return descriptors_cpu

    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [
            [] for _ in range(len(self.trainer.datamodule.val_datasets))
        ]

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.log("grad_norm", total_norm**0.5, prog_bar=True)

    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs 

        dm = self.trainer.datamodule    

        # ADDED: aggregate R1 over all validation datasets
        all_val_r1 = [] 

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

            # ADDED: geo-debug for top1 misses
            geo_stats = self.compute_val_top1_geodist_stats(
                val_dataset=val_dataset,
                predictions=predictions,
                positives=positives,
            )
            if geo_stats is not None:
                for stat_name, stat_value in geo_stats.items():
                    self.log(
                        f"{short_val_name}/{stat_name}",
                        stat_value,
                        prog_bar=False,
                        logger=True,
                    )
                self.save_val_geo_stats(short_val_name, geo_stats)  

                print(
                    f" Geo debug: valid_q={geo_stats['top1_geo_valid_queries']:.0f}, "
                    f"skip_zero_pos={geo_stats['top1_geo_skipped_zero_pos_queries']:.0f}, "
                    f"miss_mean_m={geo_stats['top1_miss_geodist_mean']:.2f}, "
                    f"miss_med_m={geo_stats['top1_miss_geodist_median']:.2f}"
                )   

            self.val_debug_results(
                current_val_dataset=val_dataset,
                q_list=q_list,
                predictions=predictions,
                positives=positives,
                num_references=num_references,
                short_val_name=short_val_name,
            )   

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

            # ADDED: collect per-dataset R1 for aggregate metrics
            all_val_r1.append(float(pitts_dict[1])) 

            del r_list, q_list, feats, num_references, positives    

        # ADDED: aggregate metrics over all validation datasets
        if len(all_val_r1) > 0:
            mean_r1_4sets = float(np.mean(all_val_r1))
            min_r1_4sets = float(np.min(all_val_r1))    

            self.log("val_mean_R1_4sets", mean_r1_4sets, prog_bar=True, logger=True)
            self.log("val_min_R1_4sets", min_r1_4sets, prog_bar=True, logger=True)  

            print(
                f"Aggregate metrics: val_mean_R1_4sets = {mean_r1_4sets:.4f}, "
                f"val_min_R1_4sets = {min_r1_4sets:.4f}"
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

    def debug_step(
        self,
        descriptors,
        labels,
        images,
        loss,
        miner_outputs,
        attn_maps,
        batch_idx,
    ):
    
        cfg = self.debug_config
        if not (
            self.current_epoch >= cfg["min_epoch"]
            and float(loss.item()) > cfg["loss_threshold"]
            and miner_outputs is not None
        ):
            return

        debug_dir = "debug_vis"
        os.makedirs(debug_dir, exist_ok=True)

        if miner_outputs is None:
            return

        dist = self.get_distance_obj(self.loss_fn, self.miner)
        is_similarity = getattr(dist, "is_inverted", False)

        with torch.no_grad():

            mat = dist(descriptors, descriptors).detach()
            closeness = mat if is_similarity else -mat
            neigh_ent = self.neighborhood_entropy_from_closeness(
                closeness, temperature=0.07
            )

            if len(miner_outputs) == 3:
                anchors, positives, negatives = miner_outputs
                if anchors.numel() == 0 or negatives.numel() == 0:
                    return

                results = self.analyze_triplet_miner(
                    anchors,
                    positives,
                    negatives,
                    mat,
                    is_similarity,
                    labels,
                    neigh_ent,
                )
            elif len(miner_outputs) == 4:
                a_pos, p, a_neg, n = miner_outputs
                # self.log("mined_pos_pairs", float(a_pos.numel()), prog_bar=True)
                # self.log("mined_neg_pairs", float(a_neg.numel()), prog_bar=True)
                
                if a_pos.numel() == 0 and a_neg.numel() == 0:
                    return
                results = self.analyze_pair_miner(
                    a_pos, p, a_neg, n, mat, is_similarity, labels, neigh_ent
                )
            else:
                raise ValueError("Oh no unexpected miner outputs")

            batch_stats = self.compute_batch_stats(
                dist_matrix=mat,
                labels=labels,
                is_similarity=is_similarity,
                neighborhood_entropy=neigh_ent,
            )
            self.save_results_to_csv(
                results=results,
                batch_stats=batch_stats,
                loss_value=loss.item(),
                batch_idx=batch_idx,
                debug_dir=debug_dir,
                is_similarity=is_similarity,
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

    def neighborhood_entropy_from_closeness(self, closeness, temperature=0.07):
        # High neighborughood entropy: anchor is too similar for many samples(model “dont know” which one to choose) -> confusion / discrimination issue / collapse.
        # Low neighborughood entropy: anchor have some neighbours (mostly positives) →> more discriminative embedding.
        B = closeness.size(0)
        c = closeness.clone()
        c.fill_diagonal_(-float("inf"))
        probs = torch.softmax(c / temperature, dim=1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        return ent  # [B]

    def save_results_to_csv(
        self,
        results,
        batch_stats,
        loss_value,
        batch_idx,
        debug_dir,
        is_similarity,
    ):

        if results:
            top_k = self.debug_config.get("top_k_triplets", 20)
            rows = []

            for rank, item in enumerate(results[:top_k]):
                rows.append(
                    {
                        "epoch": self.current_epoch,
                        "batch_idx": batch_idx,
                        "batch_loss": round(loss_value, 5),
                        "rank": rank,
                        "metric_type": "similarity" if is_similarity else "distance",
                        **item,
                    }
                )

            df = pd.DataFrame(rows)
            path = os.path.join(debug_dir, "debug_triplets.csv")
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        batch_row = {
            "epoch": self.current_epoch,
            "batch_idx": batch_idx,
            "batch_loss": round(loss_value, 5),
            "metric_type": "similarity" if is_similarity else "distance",
            **batch_stats,
        }

        df = pd.DataFrame([batch_row])
        path = os.path.join(debug_dir, "debug_batch_stats.csv")
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        print(
            f"[DEBUG] Epoch {self.current_epoch}, Batch {batch_idx}: "
            f"loss={loss_value:.4f}, separation={batch_stats['class_separation']:.4f}, "
            f"entropy={batch_stats['entropy_mean']:.4f}"
        )

    def get_distance_obj(self, loss_fn, miner=None):
        # Get the same distance based on selected loss or miner..
        if hasattr(loss_fn, "distance") and loss_fn.distance is not None:
            return loss_fn.distance
        if (
            miner is not None
            and hasattr(miner, "distance")
            and miner.distance is not None
        ):
            return miner.distance

        raise ValueError(f"Get miner/loss distance has not been added!")

    def analyze_triplet_miner(
        self,
        anchors,
        positives,
        negatives,
        distance_matrix,
        is_similarity,
        labels,
        neigh_ent,
    ):
        """

        Triplet loss: positive distance nearer to anchor than negative

        For DISTANCE (is_similarity=False):
            loss = max(0, d(a,p) - d(a,n) + margin)
            Goal: d(a,p) < d(a,n)

        For SIMILARITY (is_similarity=True):
            loss = max(0, s(a,n) - s(a,p) + margin)
            Goal: s(a,p) > s(a,n)
        """
        results = []
        margin = float(self.get_margin())

        for i in range(len(anchors)):
            a_idx = anchors[i].item()
            p_idx = positives[i].item()
            n_idx = negatives[i].item()

            ap_value = distance_matrix[a_idx, p_idx].item()  # anchor-positive
            an_value = distance_matrix[a_idx, n_idx].item()  # anchor-negative

            if is_similarity:
                violation = an_value - ap_value + margin
            else:
                violation = ap_value - an_value + margin

            triplet_loss = max(0.0, violation)

            anchor_entropy = neigh_ent[a_idx].item()

            problem = self.diagnose_problem(
                ap_value, an_value, is_similarity, triplet_loss
            )

            results.append(
                {
                    "type": "TRIPLET",
                    # Anchor info
                    "anchor_idx": a_idx,
                    "anchor_label": labels[a_idx].item(),
                    "anchor_entropy": round(anchor_entropy, 4),
                    # Positive info
                    "positive_idx": p_idx,
                    "positive_label": labels[p_idx].item(),
                    "positive_entropy": round(neigh_ent[p_idx].item(), 4),
                    "negative_idx": n_idx,
                    "negative_label": labels[n_idx].item(),
                    "negative_entropy": round(neigh_ent[n_idx].item(), 4),
                    "ap_value": round(ap_value, 4),
                    "an_value": round(an_value, 4),
                    "violation": round(violation, 4),
                    "triplet_loss": round(triplet_loss, 5),
                    "margin": margin,
                    "problem": problem,
                }
            )

        results.sort(key=lambda x: x["triplet_loss"], reverse=True)

        return results

    def analyze_pair_miner(
        self, a_pos, p, a_neg, n, mat, is_similarity, labels, neigh_ent
    ):
        results = []
        top_k = self.debug_config.get("top_k_triplets", 20)

        # --- positives: should be close to anchor
        if a_pos.numel() > 0:
            pos_vals = mat[a_pos, p]
            # similarity: worst = lowest; distance: worst = biggest
            order = torch.argsort(pos_vals, descending=(not is_similarity))
            for j in order[: min(top_k, order.numel())].tolist():
                ai = int(a_pos[j])
                bi = int(p[j])
                results.append(
                    {
                        "type": "POS_PAIR_BAD",
                        "anchor_idx": ai,
                        "anchor_label": int(labels[ai]),
                        "anchor_entropy": round(float(neigh_ent[ai]), 4),
                        "other_idx": bi,
                        "other_label": int(labels[bi]),
                        "other_entropy": round(float(neigh_ent[bi]), 4),
                        "value": round(float(pos_vals[j]), 4),
                    }
                )

        # --- negatives: should be far away from anchor
        if a_neg.numel() > 0:
            neg_vals = mat[a_neg, n]
            # similarity: worst = biggest; distance: worst = lowest
            order = torch.argsort(neg_vals, descending=is_similarity)
            for j in order[: min(top_k, order.numel())].tolist():
                ai = int(a_neg[j])
                bi = int(n[j])
                results.append(
                    {
                        "type": "NEG_PAIR_BAD",
                        "anchor_idx": ai,
                        "anchor_label": int(labels[ai]),
                        "anchor_entropy": round(float(neigh_ent[ai]), 4),
                        "other_idx": bi,
                        "other_label": int(labels[bi]),
                        "other_entropy": round(float(neigh_ent[bi]), 4),
                        "value": round(float(neg_vals[j]), 4),
                    }
                )

        return results

    def get_margin(self):
        if hasattr(self.loss_fn, "margin"):
            return self.loss_fn.margin
        if hasattr(self.loss_fn, "m"):
            return self.loss_fn.m
        return 0.1

    def compute_batch_stats(
        self,
        dist_matrix,
        labels,
        is_similarity,
        neighborhood_entropy,
    ):
        N = dist_matrix.size(0)
        unique_labels = torch.unique(labels)

        intra_values = []
        inter_values = []

        for label in unique_labels:
            mask = labels == label
            indices = torch.where(mask)[0]
            other_indices = torch.where(~mask)[0]

            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        intra_values.append(dist_matrix[indices[i], indices[j]].item())

            if len(other_indices) > 0 and len(indices) > 0:
                sample_size = min(5, len(other_indices))
                sampled = other_indices[
                    torch.randperm(len(other_indices))[:sample_size]
                ]
                for idx in indices[:3]:
                    for other in sampled:
                        inter_values.append(dist_matrix[idx, other].item())

        intra = np.array(intra_values) if intra_values else np.array([0.0])
        inter = np.array(inter_values) if inter_values else np.array([0.0])

        # Class separation
        if is_similarity:
            # Similarity: intra should be High, inter Low
            separation = intra.mean() - inter.mean()
        else:
            # Distance: intra Low, inter should be High,
            separation = inter.mean() - intra.mean()

        return {
            "intra_class_mean": round(intra.mean(), 4),
            "intra_class_std": round(intra.std(), 4),
            "inter_class_mean": round(inter.mean(), 4),
            "inter_class_std": round(inter.std(), 4),
            "class_separation": round(separation, 4),  # Higer = better!
            "entropy_mean": round(neighborhood_entropy.mean().item(), 4),
            "entropy_std": round(neighborhood_entropy.std().item(), 4),
            "entropy_min": round(neighborhood_entropy.min().item(), 4),
            "entropy_max": round(neighborhood_entropy.max().item(), 4),
            "num_samples": N,
            "num_classes": len(unique_labels),
        }

    def diagnose_problem(self, ap_value, an_value, is_similarity, triplet_loss):
        if triplet_loss == 0:
            return "OK"

        if is_similarity:
            pos_bad = ap_value < 0.5
            neg_bad = an_value > 0.7
        else:
            pos_bad = ap_value > 1.0
            neg_bad = an_value < 0.8

        if pos_bad and neg_bad:
            return "BOTH_BAD"
        elif pos_bad:
            return "POSITIVE_TOO_FAR"
        elif neg_bad:
            return "NEGATIVE_TOO_CLOSE"
        return "MARGINAL"


    def log_basic_mining_stats(self, descriptors, miner_outputs):
        """
        Log similarity statistics for mined examples.   

        Why this method exists:
        Mined counts alone are not enough. Two runs may mine the same number of examples,
        but one run may mine clean/useful examples while another may mine noisy ones.   

        What we log:
        - mean similarity of mined positive pairs
        - mean similarity of mined negative pairs
        - for triplets: mean anchor-positive and anchor-negative similarity 

        How to read it:
        - higher positive similarity is usually good
        - very high negative similarity means the miner is finding hard confusers
        - if negative similarity stays too high for too long, the model may struggle to separate places 

        Example:
        - mined_neg_sim_mean rising together with many mined negatives can indicate
          aggressive mining pressure or possible false negatives
        - mined_pos_sim_mean rising over training usually means positives are becoming tighter
        """
        if miner_outputs is None:
            return  

        dist = self.get_distance_obj(self.loss_fn, self.miner)  

        with torch.no_grad():
            mat = dist(descriptors, descriptors).detach()   

            if len(miner_outputs) == 4:
                a_pos, p, a_neg, n = miner_outputs  

                if a_pos.numel() > 0:
                    self.log(
                        "mined_pos_sim_mean",
                        float(mat[a_pos, p].mean().item()),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    )   

                if a_neg.numel() > 0:
                    self.log(
                        "mined_neg_sim_mean",
                        float(mat[a_neg, n].mean().item()),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    )   

            elif len(miner_outputs) == 3:
                a, p, n = miner_outputs 

                if a.numel() > 0:
                    self.log(
                        "triplet_pos_sim_mean",
                        float(mat[a, p].mean().item()),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    )
                    self.log(
                        "triplet_neg_sim_mean",
                        float(mat[a, n].mean().item()),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    )   

    def log_domain_aware_mining(self, descriptors, domains, miner_outputs):
        """
        Domain-aware mining analysis.

        Domains:
        - 0 = UAV
        - 1 = SAT

        We split mined pairs into:
        - UAV-UAV
        - SAT-SAT
        - UAV-SAT

        Interpretation:
        - high mined_pos_uav_sat_share:
            miner still sees useful cross-domain positives
        - high mined_neg_uav_sat_share with high similarity:
            model is confused across domains or training is over-pushing cross-domain negatives
        """
        dist = self.get_distance_obj(self.loss_fn, self.miner)

        with torch.no_grad():
            mat = dist(descriptors, descriptors).detach()

            if len(miner_outputs) == 4:
                a_pos, p, a_neg, n = miner_outputs
                self._log_pair_group("mined_pos", a_pos, p, domains, mat)
                self._log_pair_group("mined_neg", a_neg, n, domains, mat)

            elif len(miner_outputs) == 3:
                a, p, n = miner_outputs
                self._log_pair_group("triplet_pos", a, p, domains, mat)
                self._log_pair_group("triplet_neg", a, n, domains, mat)


    def _log_pair_group(self, prefix, idx_a, idx_b, domains, sim_matrix):
        """
        Log how mined examples are distributed across domain combinations.

        This helper is shared by pair-based and triplet-based miners.

        Example:
        - prefix='mined_neg' produces logs such as:
          mined_neg_uav_uav_share, mined_neg_sat_sat_share, mined_neg_uav_sat_share

        These logs are intended mainly for epoch-level analysis, not step-level debugging.
        """
        if idx_a.numel() == 0:
            return

        da = domains[idx_a]
        db = domains[idx_b]
        vals = sim_matrix[idx_a, idx_b]

        masks = {
            "uav_uav": (da == 0) & (db == 0),
            "sat_sat": (da == 1) & (db == 1),
            "uav_sat": da != db,
        }

        total = max(int(idx_a.numel()), 1)

        for name, mask in masks.items():
            count = int(mask.sum().item())
            share = float(count / total)

            self.log(
                f"{prefix}_{name}_count",
                count,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{prefix}_{name}_share",
                share,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            if count > 0:
                self.log(
                    f"{prefix}_{name}_sim_mean",
                    float(vals[mask].mean().item()),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )


    def compute_val_top1_geodist_stats(self, val_dataset, predictions, positives):
        """
        Analyze top-1 retrieval misses by geographic distance.
    
        Why this matters:
        Recall@1 treats every miss equally, but not every miss is equally severe.
        A prediction just outside the positive radius may indicate a near-correct retrieval,
        while a prediction far away is a clear localization failure.
    
        This diagnostic helps separate:
        - threshold / annotation effects
        - genuine retrieval errors
    
        Distance buckets:
        - <100 m: suspicious if counted as a miss, may indicate GT/indexing issues
        - 100-150 m: near-miss just outside the positive radius
        - 150-250 m: moderate miss
        - >250 m: clear miss
    
        Important:
        Queries with zero positives are skipped, otherwise broken or unmatched validation
        samples would artificially inflate the miss statistics.
        """
        if not hasattr(val_dataset, "q_utm_np") or not hasattr(val_dataset, "db_utm_np"):
            return None

        q_utm = np.asarray(val_dataset.q_utm_np)
        db_utm = np.asarray(val_dataset.db_utm_np)

        if len(q_utm) != len(predictions):
            print(
                f"[WARN] q_utm length {len(q_utm)} != num predictions {len(predictions)}. "
                "Skipping geo debug."
            )
            return None

        miss_under_100 = 0
        miss_100_150 = 0
        miss_150_250 = 0
        miss_over_250 = 0

        miss_dists = []
        hit_count = 0
        valid_queries = 0
        skipped_zero_positive_queries = 0

        for q_idx in range(len(predictions)):
            if len(positives[q_idx]) == 0:
                skipped_zero_positive_queries += 1
                continue

            if len(predictions[q_idx]) == 0:
                continue

            valid_queries += 1

            top1_db_idx = int(predictions[q_idx][0])
            is_hit = top1_db_idx in positives[q_idx]

            dist_m = float(np.linalg.norm(q_utm[q_idx] - db_utm[top1_db_idx]))

            if is_hit:
                hit_count += 1
            else:
                miss_dists.append(dist_m)

                if dist_m < 100.0:
                    miss_under_100 += 1
                elif dist_m < 150.0:
                    miss_100_150 += 1
                elif dist_m < 250.0:
                    miss_150_250 += 1
                else:
                    miss_over_250 += 1

        total_misses = len(miss_dists)

        if valid_queries == 0:
            return {
                "top1_geo_valid_queries": 0.0,
                "top1_geo_skipped_zero_pos_queries": float(skipped_zero_positive_queries),
                "top1_hit_count": 0.0,
                "top1_miss_count": 0.0,
                "top1_hit_rate_valid": 0.0,
                "top1_miss_under_100_share": 0.0,
                "top1_miss_100_150_share": 0.0,
                "top1_miss_150_250_share": 0.0,
                "top1_miss_over_250_share": 0.0,
                "top1_miss_geodist_mean": 0.0,
                "top1_miss_geodist_median": 0.0,
            }

        if total_misses == 0:
            return {
                "top1_geo_valid_queries": float(valid_queries),
                "top1_geo_skipped_zero_pos_queries": float(skipped_zero_positive_queries),
                "top1_hit_count": float(hit_count),
                "top1_miss_count": 0.0,
                "top1_hit_rate_valid": float(hit_count / valid_queries),
                "top1_miss_under_100_share": 0.0,
                "top1_miss_100_150_share": 0.0,
                "top1_miss_150_250_share": 0.0,
                "top1_miss_over_250_share": 0.0,
                "top1_miss_geodist_mean": 0.0,
                "top1_miss_geodist_median": 0.0,
            }

        miss_dists_np = np.array(miss_dists, dtype=np.float32)

        return {
            "top1_geo_valid_queries": float(valid_queries),
            "top1_geo_skipped_zero_pos_queries": float(skipped_zero_positive_queries),
            "top1_hit_count": float(hit_count),
            "top1_miss_count": float(total_misses),
            "top1_hit_rate_valid": float(hit_count / valid_queries),
            "top1_miss_under_100_share": float(miss_under_100 / total_misses),
            "top1_miss_100_150_share": float(miss_100_150 / total_misses),
            "top1_miss_150_250_share": float(miss_150_250 / total_misses),
            "top1_miss_over_250_share": float(miss_over_250 / total_misses),
            "top1_miss_geodist_mean": float(miss_dists_np.mean()),
            "top1_miss_geodist_median": float(np.median(miss_dists_np)),
        }


    def save_val_geo_stats(self, short_val_name, geo_stats, debug_dir="debug_vis"):
        os.makedirs(debug_dir, exist_ok=True)

        row = {
            "epoch": self.current_epoch,
            "dataset": short_val_name,
            **geo_stats,
        }

        path = os.path.join(debug_dir, "debug_val_geo_stats.csv")
        df = pd.DataFrame([row])
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)