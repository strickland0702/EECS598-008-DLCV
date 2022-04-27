import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

"""
General utilities to help with implementation
"""


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with
      elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot

    Inputs:
    - X_data: set of [batch, 3, width, height] data
    - y_data: paired label of X_data in [batch] shape
    - samples_per_class: number of samples want to present
    - class_list: list of class names
      e.g.) ['plane', 'car', 'bird', 'cat', 'deer', 'dog',
      'frog', 'horse', 'ship', 'truck']

    Outputs:
    - An grid-image that visualize samples_per_class
      number of samples per class
    """
    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        plt.text(
            -4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha="right"
        )
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)


def detection_visualizer(img, idx_to_class, bbox=None, pred=None, points=None):
    """
    Data visualizer on the original image. Support both GT
    box input and proposal input.

    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates
            (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional),
            a tensor of shape N'x6, where N' is the number
            of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    # Convert image to HWC if it is passed as a Tensor (0-1, CHW).
    if isinstance(img, torch.Tensor):
        img = (img * 255).permute(1, 2, 0)

    img_copy = np.array(img).astype("uint8")
    _, ax = plt.subplots(frameon=False)

    ax.axis("off")
    ax.imshow(img_copy)

    # fmt: off
    if points is not None:
        points_x = [t[0] for t in points]
        points_y = [t[1] for t in points]
        ax.scatter(points_x, points_y, color="yellow", s=24)

    if bbox is not None:
        for single_bbox in bbox:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(1.0, 0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                ax.text(
                    x0, y0, obj_cls, size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )

    if pred is not None:
        for single_bbox in pred:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(0, 1.0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                conf_score = single_bbox[5].item()
                ax.text(
                    x0, y0 + 15, f"{obj_cls}, {conf_score:.2f}",
                    size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )
    # fmt: on
    plt.show()


def attention_visualizer(img, attn_weights, token):
    """
    Visuailze the attended regions on a single frame from a single query word.
    Inputs:
    - img: Image tensor input, of shape (3, H, W)
    - attn_weights: Attention weight tensor, on the final activation map
    - token: The token string you want to display above the image

    Outputs:
    - img_output: Image tensor output, of shape (3, H+25, W)

    """
    C, H, W = img.shape
    assert C == 3, "We only support image with three color channels!"

    # Reshape attention map
    attn_weights = cv2.resize(
        attn_weights.data.numpy().copy(), (H, W), interpolation=cv2.INTER_NEAREST
    )
    attn_weights = np.repeat(np.expand_dims(attn_weights, axis=2), 3, axis=2)

    # Combine image and attention map
    img_copy = img.float().div(255.0).permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    masked_img = cv2.addWeighted(attn_weights, 0.5, img_copy, 0.5, 0)
    img_copy = np.concatenate((np.zeros((25, W, 3)), masked_img), axis=0)

    # Add text
    cv2.putText(
        img_copy,
        "%s" % (token),
        (10, 15),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (255, 255, 255),
        thickness=1,
    )

    return img_copy
