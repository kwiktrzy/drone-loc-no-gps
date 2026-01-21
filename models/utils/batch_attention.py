import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os


@staticmethod
def visualize_batch_attention(
    images_tensor, attn_maps, labels, batch_idx, epoch, output_dir, limit=None
):
    MEAN = np.array([0.485, 0.456, 0.406])  # todo: remember about that
    STD = np.array([0.229, 0.224, 0.225])

    dump_dir = os.path.join(
        output_dir, f"spike_epoch_{epoch:03d}_batch_{batch_idx:04d}"
    )
    os.makedirs(dump_dir, exist_ok=True)

    count = images_tensor.size(0)
    if limit:
        count = min(count, limit)

    for i in range(count):

        img = images_tensor[i].detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
        img = img * STD + MEAN
        img = np.clip(img, 0, 1)
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        attn = attn_maps[i].detach()
        if attn.dim() == 3:
            attn = attn.squeeze(0)

        attn = attn.unsqueeze(0).unsqueeze(0)
        attn = F.interpolate(
            attn,
            size=(img_bgr.shape[0], img_bgr.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        attn = attn.squeeze().cpu().numpy()

        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn_uint8 = (attn_norm * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

        filename = f"Idx_{i:03d}_Lbl_{labels[i].item()}.jpg"
        cv2.imwrite(os.path.join(dump_dir, filename), overlay)

    print(f"[DEBUG] Saved {count} attention maps to {dump_dir}")
