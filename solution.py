from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import gc
import math
import os
import random
import time

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torchvision.transforms import functional as TF


NUM_CLASSES = 20

SEED_R42 = 42
SEED_R1337 = 1337
SEED_INC = 42
SEED_VGG = 42

TRAIN_BATCH = 18

R_IMG = 256
I_IMG = 299
VGG_IMG = 256

R_TEST_BATCH = 48
I_TEST_BATCH = 48
VGG_TEST_BATCH = 80

NUM_WORKERS = 2

EPOCHS = 15
MIX_EPOCHS = 12
CLEAN_EPOCHS = 3

BACKBONE_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 2e-4

HARD_LS = 0.08
MIX_LS = 0.05

MIX_PROB = 0.70
MIXUP_ALPHA = 0.20
CUTMIX_ALPHA = 1.00

EMA_DECAY = 0.999

AUX_LOSS_WEIGHT = 0.40

WARMUP_STEPS = 380
MIN_LR_FACTOR = 0.05

VGG_EPOCHS = 9
VGG_MICRO_BATCH = 6
VGG_ACCUM = 3
VGG_BACKBONE_LR = 1e-4
VGG_HEAD_LR = 5e-4
VGG_REFERENCE_TOTAL_STEPS = 4755

FINAL_WEIGHTS = {
    "R42": 0.153,
    "R1337": 0.357,
    "INC": 0.340,
    "VGG": 0.150,
}

TTA_D4 = [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
]

TTA_D2 = [
    (0, False),
    (2, False),
]

TTA_D1 = [
    (0, False),
]


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(0)

if torch.cuda.device_count() < 2:
    raise RuntimeError("Two CUDA GPUs are required.")

torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def valid_root(p):
    required = [
        "train.csv",
        "val.csv",
        "test.csv",
        "train",
        "val",
        "test",
    ]
    return all((p / x).exists() for x in required)


candidates = [
    Path("."),
    Path("..") / "input",
]

ROOT = None

for base in candidates:
    if valid_root(base):
        ROOT = base
        break

    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and valid_root(p):
                ROOT = p
                break

    if ROOT is not None:
        break


if ROOT is None:
    raise FileNotFoundError(
        "Competition files were not found in the current directory or ../input."
    )


train_df = pd.read_csv(
    ROOT / "train.csv",
    dtype={"id": str},
)

val_df = pd.read_csv(
    ROOT / "val.csv",
    dtype={"id": str},
)

test_df = pd.read_csv(
    ROOT / "test.csv",
    dtype={"id": str},
)


sample_path = None

for p in [
    ROOT / "sample_submission.csv",
    ROOT / "sample_submission_example.csv",
]:
    if p.exists():
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError("Sample submission file was not found.")


sample_submission = pd.read_csv(
    sample_path,
    dtype={"id": str},
)


assert len(train_df) == 4999
assert len(val_df) == 699
assert len(test_df) == 8299


train_df = train_df.copy()
val_df = val_df.copy()

train_df["source"] = "train"
val_df["source"] = "val"

all_labeled = pd.concat(
    [train_df, val_df],
    ignore_index=True,
)

assert len(all_labeled) == 5698

labels = sorted(
    all_labeled["label"]
    .astype(int)
    .unique()
    .tolist()
)

assert labels == list(range(NUM_CLASSES))

assert (
    sample_submission["id"].astype(str).tolist()
    ==
    test_df["id"].astype(str).tolist()
)


STEPS_PER_EPOCH = math.ceil(
    len(all_labeled) / TRAIN_BATCH
)

TOTAL_STEPS = (
    EPOCHS * STEPS_PER_EPOCH
)

assert STEPS_PER_EPOCH == 317
assert TOTAL_STEPS == 4755

print("ROOT:", ROOT)
print("labeled:", len(all_labeled))
print("test:", len(test_df))


# ============================================================
# 2. DATA / AUGMENTATION
# ============================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


class RandomQuarterTurn:

    def __call__(self, img):
        k = random.randrange(4)

        if k == 0:
            return img

        return TF.rotate(
            img,
            90*k,
            interpolation=
                transforms
                .InterpolationMode
                .NEAREST,
        )


def make_train_transform(size):

    return transforms.Compose([
        RandomQuarterTurn(),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.RandomResizedCrop(
            size,
            scale=(0.78, 1.00),
            ratio=(0.90, 1.10),
            interpolation=
                transforms
                .InterpolationMode
                .BILINEAR,
        ),

        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    0.12,
                    0.12,
                    0.15,
                    0.015,
                )
            ],
            p=0.35,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),

        transforms.RandomErasing(
            p=0.10,
            scale=(0.02, 0.08),
            ratio=(0.5, 2.0),
            value="random",
        ),
    ])


def make_eval_transform(size):

    return transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=
                transforms
                .InterpolationMode
                .BILINEAR,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])


class TrainDataset(Dataset):

    def __init__(
        self,
        df,
        transform,
    ):
        self.df = (
            df
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        path = (
            ROOT
            / str(row["source"])
            / str(row["id"])
        )

        with Image.open(path) as im:
            x = self.transform(
                im.convert("RGB")
            )

        return (
            x,
            int(row["label"]),
        )


class TestDataset(Dataset):

    def __init__(
        self,
        df,
        transform,
    ):
        self.df = (
            df
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(
        self,
        idx,
    ):
        image_id = str(
            self.df.iloc[idx]["id"]
        )

        path = (
            ROOT
            / "test"
            / image_id
        )

        with Image.open(path) as im:
            x = self.transform(
                im.convert("RGB")
            )

        return (
            x,
            image_id,
        )


def seed_worker(worker_id):
    s = (
        torch.initial_seed()
        % 2**32
    )

    random.seed(s)
    np.random.seed(s)


def make_train_loader(
    size,
    batch,
    loader_seed,
):

    ds = TrainDataset(
        all_labeled,
        make_train_transform(size),
    )

    g = (
        torch.Generator()
        .manual_seed(loader_seed)
    )

    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        generator=g,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=(
            seed_worker
            if NUM_WORKERS > 0
            else None
        ),
    )


def make_test_loader(
    size,
    batch,
    n=None,
):

    ds = TestDataset(
        test_df,
        make_eval_transform(size),
    )

    if n is not None:
        ds = Subset(
            ds,
            range(
                min(
                    n,
                    len(ds),
                )
            ),
        )

    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )


# ============================================================
# 3. MIX / LOSS / LR
# ============================================================

def smooth_targets(
    y,
    smoothing,
):

    off = (
        smoothing
        / NUM_CLASSES
    )

    on = (
        1.0
        - smoothing
        + off
    )

    t = torch.full(
        (
            len(y),
            NUM_CLASSES,
        ),
        off,
        device=y.device,
        dtype=torch.float32,
    )

    t.scatter_(
        1,
        y[:, None],
        on,
    )

    return t


def soft_ce(
    logits,
    target,
):

    return -(
        target
        * F.log_softmax(
            logits.float(),
            dim=1,
        )
    ).sum(
        dim=1
    ).mean()


def apply_mix(
    x,
    y,
    rng,
):

    target = smooth_targets(
        y,
        MIX_LS,
    )

    if (
        len(y) < 2
        or rng.random() > MIX_PROB
    ):
        return (
            x,
            target,
            "none",
        )

    perm = torch.randperm(
        len(y),
        device=y.device,
    )

    target2 = target[
        perm
    ]

    if rng.random() < 0.5:

        lam = float(
            rng.beta(
                MIXUP_ALPHA,
                MIXUP_ALPHA,
            )
        )

        return (
            lam*x
            + (1-lam)*x[perm],

            lam*target
            + (1-lam)*target2,

            "mixup",
        )

    lam = float(
        rng.beta(
            CUTMIX_ALPHA,
            CUTMIX_ALPHA,
        )
    )

    _, _, h, w = x.shape

    ratio = math.sqrt(
        1-lam
    )

    cw = int(
        w*ratio
    )

    ch = int(
        h*ratio
    )

    cx = int(
        rng.integers(0, w)
    )

    cy = int(
        rng.integers(0, h)
    )

    x1 = max(
        0,
        cx-cw//2,
    )

    x2 = min(
        w,
        cx+cw//2,
    )

    y1 = max(
        0,
        cy-ch//2,
    )

    y2 = min(
        h,
        cy+ch//2,
    )

    z = x.clone()

    z[
        :,
        :,
        y1:y2,
        x1:x2,
    ] = x[
        perm,
        :,
        y1:y2,
        x1:x2,
    ]

    lam_eff = (
        1.0
        -
        (
            (x2-x1)
            * (y2-y1)
        )
        / float(
            h*w
        )
    )

    t = (
        lam_eff*target
        + (1-lam_eff)*target2
    )

    return (
        z,
        t,
        "cutmix",
    )


def lr_factor(
    step,
    total_steps,
    warmup_steps=WARMUP_STEPS,
):

    warmup_steps = min(
        warmup_steps,
        max(
            1,
            total_steps-1,
        ),
    )

    if step < warmup_steps:
        return (
            (step+1)
            / warmup_steps
        )

    denom = max(
        1,
        total_steps
        - warmup_steps
        - 1,
    )

    progress = (
        step-warmup_steps
    ) / denom

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    cosine = (
        0.5
        * (
            1
            + math.cos(
                math.pi*progress
            )
        )
    )

    return (
        MIN_LR_FACTOR
        +
        (
            1-MIN_LR_FACTOR
        )*cosine
    )


def build_r101(seed):
    seed_all(seed)

    m = models.resnet101(
        weights=
            models
            .ResNet101_Weights
            .IMAGENET1K_V2
    )

    n = m.fc.in_features

    m.fc = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(
            n,
            NUM_CLASSES,
        ),
    )

    return m


def build_inception(seed):
    seed_all(seed)

    m = models.inception_v3(
        weights=
            models
            .Inception_V3_Weights
            .IMAGENET1K_V1,
        aux_logits=True,
        transform_input=True,
    )

    n = m.fc.in_features

    m.fc = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(
            n,
            NUM_CLASSES,
        ),
    )

    aux_n = (
        m
        .AuxLogits
        .fc
        .in_features
    )

    m.AuxLogits.fc = nn.Linear(
        aux_n,
        NUM_CLASSES,
    )

    return m


def build_vgg(seed):
    seed_all(seed)

    m = models.vgg16_bn(
        weights=
            models
            .VGG16_BN_Weights
            .IMAGENET1K_V1
    )

    m.avgpool = nn.AdaptiveAvgPool2d(
        (1, 1)
    )

    m.classifier = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(
            512,
            NUM_CLASSES,
        ),
    )

    return m


def main_aux(out):
    if torch.is_tensor(out):
        return out, None

    if hasattr(out, "logits"):
        return (
            out.logits,
            getattr(
                out,
                "aux_logits",
                None,
            ),
        )

    if isinstance(out, (tuple, list)):
        return (
            out[0],
            out[1] if len(out) > 1 else None,
        )

    raise TypeError(type(out))


def build_optimizer(model, kind):
    if kind == "r101":
        head = list(
            model.fc.parameters()
        )
        bb_lr = BACKBONE_LR
        hd_lr = HEAD_LR

    elif kind == "inception":
        head = (
            list(
                model.fc.parameters()
            )
            +
            list(
                model
                .AuxLogits
                .fc
                .parameters()
            )
        )
        bb_lr = BACKBONE_LR
        hd_lr = HEAD_LR

    elif kind == "vgg":
        head = list(
            model
            .classifier
            .parameters()
        )
        bb_lr = VGG_BACKBONE_LR
        hd_lr = VGG_HEAD_LR

    else:
        raise ValueError(kind)

    head_ids = {
        id(p)
        for p in head
    }

    backbone = [
        p
        for p in model.parameters()
        if id(p) not in head_ids
    ]

    opt = torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": bb_lr,
            },
            {
                "params": head,
                "lr": hd_lr,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return opt, bb_lr, hd_lr


def prepare_ema(model, device):
    ema = deepcopy(model).to(device).eval()

    for p in ema.parameters():
        p.requires_grad = False

    es = ema.state_dict()
    ms = model.state_dict()

    ef = []
    mf = []
    ints = []

    for k in es:
        if torch.is_floating_point(es[k]):
            ef.append(es[k])
            mf.append(ms[k])
        else:
            ints.append((es[k], ms[k]))

    return ema, ef, mf, ints


@torch.no_grad()
def ema_update(ef, mf, ints):
    torch._foreach_mul_(
        ef,
        EMA_DECAY,
    )

    torch._foreach_add_(
        ef,
        mf,
        alpha=1-EMA_DECAY,
    )

    for e, m in ints:
        e.copy_(m)


def cpu_state_dict(model):
    return {
        k:
            v.detach()
            .cpu()
            .clone()
        for k, v
        in model
        .state_dict()
        .items()
    }


# ============================================================
# 5. FULL TRAIN ONE MODEL
# ============================================================

def train_full_model(
    *,
    name,
    kind,
    model,
    device_idx,
    size,
    seed,
):

    device = torch.device(
        f"cuda:{device_idx}"
    )

    torch.cuda.set_device(
        device_idx
    )

    # Device-specific RNG seed.
    torch.cuda.default_generators[
        device_idx
    ].manual_seed(
        seed
    )

    model = model.to(
        device
    )

    loader = make_train_loader(
        size,
        TRAIN_BATCH,
        seed,
    )

    (
        optimizer,
        bb_lr,
        hd_lr,
    ) = build_optimizer(
        model,
        kind,
    )

    (
        ema,
        ef,
        mf,
        ints,
    ) = prepare_ema(
        model,
        device,
    )

    scaler = (
        torch.amp
        .GradScaler(
            "cuda",
            enabled=True,
        )
    )

    rng = (
        np.random
        .default_rng(
            seed
        )
    )

    global_step = 0

    t_train = (
        time.perf_counter()
    )

    for epoch in range(
        EPOCHS
    ):

        is_mix = (
            epoch < MIX_EPOCHS
        )

        phase = (
            "MIX"
            if is_mix
            else "CLEAN"
        )

        model.train()

        loss_sum = 0.0
        correct = 0
        seen = 0

        n_mixup = 0
        n_cutmix = 0

        t0 = (
            time.perf_counter()
        )

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            if is_mix:

                (
                    x_in,
                    target,
                    mix_kind,
                ) = apply_mix(
                    x,
                    y,
                    rng,
                )

                n_mixup += int(
                    mix_kind
                    == "mixup"
                )

                n_cutmix += int(
                    mix_kind
                    == "cutmix"
                )

            else:

                x_in = x
                target = None

            fac = lr_factor(
                global_step,
                TOTAL_STEPS,
            )

            optimizer.param_groups[
                0
            ]["lr"] = (
                bb_lr*fac
            )

            optimizer.param_groups[
                1
            ]["lr"] = (
                hd_lr*fac
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=True,
            ):

                out = model(
                    x_in
                )

                (
                    logits,
                    aux,
                ) = main_aux(
                    out
                )

                if is_mix:

                    main_loss = soft_ce(
                        logits,
                        target,
                    )

                else:

                    main_loss = (
                        F.cross_entropy(
                            logits.float(),
                            y,
                            label_smoothing=
                                HARD_LS,
                        )
                    )

                if (
                    kind
                    == "inception"
                    and aux is not None
                ):

                    if is_mix:

                        aux_loss = soft_ce(
                            aux,
                            target,
                        )

                    else:

                        aux_loss = (
                            F.cross_entropy(
                                aux.float(),
                                y,
                                label_smoothing=
                                    HARD_LS,
                            )
                        )

                    loss = (
                        main_loss
                        + AUX_LOSS_WEIGHT
                        * aux_loss
                    )

                else:

                    loss = (
                        main_loss
                    )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            ema_update(
                ef,
                mf,
                ints,
            )

            bs = len(y)

            loss_sum += (
                float(
                    loss.detach()
                )
                * bs
            )

            correct += int(
                (
                    logits
                    .detach()
                    .argmax(1)
                    == y
                ).sum().item()
            )

            seen += bs
            global_step += 1

        sec = (
            time.perf_counter()
            - t0
        )

        print(
            f"[{name}] "
            f"e{epoch+1:02d}/{EPOCHS} "
            f"{phase} | "
            f"loss={loss_sum/seen:.4f} | "
            f"top1={correct/seen:.4f} | "
            f"mixup={n_mixup} | "
            f"cutmix={n_cutmix} | "
            f"{sec:.1f}s"
        )

    assert global_step == TOTAL_STEPS

    torch.cuda.synchronize(
        device
    )

    train_sec = (
        time.perf_counter()
        - t_train
    )

    state = cpu_state_dict(
        ema
    )

    print(
        f"[{name}] TRAIN DONE | "
        f"{train_sec/60:.2f} min"
    )

    del model
    del ema
    del loader
    del optimizer
    del scaler
    del ef
    del mf
    del ints

    gc.collect()
    torch.cuda.empty_cache()

    return (
        state,
        train_sec,
    )


# ============================================================
# 6. TRAIN VGG e9 — microbatch 6 × grad accumulation 3
# ============================================================

def train_vgg_e9(
    *,
    model,
    device_idx,
    seed,
):

    device = torch.device(
        f"cuda:{device_idx}"
    )

    torch.cuda.set_device(
        device_idx
    )

    torch.cuda.default_generators[
        device_idx
    ].manual_seed(
        seed
    )

    model = model.to(
        device
    )

    loader = make_train_loader(
        VGG_IMG,
        VGG_MICRO_BATCH,
        seed,
    )

    (
        optimizer,
        bb_lr,
        hd_lr,
    ) = build_optimizer(
        model,
        "vgg",
    )

    (
        ema,
        ef,
        mf,
        ints,
    ) = prepare_ema(
        model,
        device,
    )

    scaler = (
        torch.amp
        .GradScaler(
            "cuda",
            enabled=True,
        )
    )

    rng = (
        np.random
        .default_rng(
            seed
        )
    )

    # 5698 labels -> effective-batch-18 = 317 optimizer updates/epoch.
    updates_per_epoch = math.ceil(
        len(all_labeled)
        / TRAIN_BATCH
    )

    assert updates_per_epoch == 317

    global_step = 0

    train_t0 = (
        time.perf_counter()
    )

    for epoch in range(
        VGG_EPOCHS
    ):

        model.train()

        iterator = iter(
            loader
        )

        loss_sum = 0.0
        correct = 0
        seen = 0

        n_mixup = 0
        n_cutmix = 0

        epoch_t0 = (
            time.perf_counter()
        )

        for update in range(
            updates_per_epoch
        ):

            # Build one effective batch as up to 3 microbatches.
            group = []

            for _ in range(
                VGG_ACCUM
            ):

                try:
                    batch = next(
                        iterator
                    )
                except StopIteration:
                    break

                group.append(
                    batch
                )

            if not group:
                break

            total_bs = sum(
                len(y)
                for _, y in group
            )

            factor = lr_factor(
                global_step,
                VGG_REFERENCE_TOTAL_STEPS,
            )

            optimizer.param_groups[
                0
            ]["lr"] = (
                bb_lr * factor
            )

            optimizer.param_groups[
                1
            ]["lr"] = (
                hd_lr * factor
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            for x, y in group:

                x = x.to(
                    device,
                    non_blocking=True,
                )

                y = y.to(
                    device,
                    non_blocking=True,
                )

                (
                    x_in,
                    target,
                    mix_kind,
                ) = apply_mix(
                    x,
                    y,
                    rng,
                )

                n_mixup += int(
                    mix_kind == "mixup"
                )

                n_cutmix += int(
                    mix_kind == "cutmix"
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=True,
                ):

                    logits = model(
                        x_in
                    )

                    raw_loss = soft_ce(
                        logits,
                        target,
                    )

                    # Sample-count-correct accumulation,
                    # including the final 10-sample effective batch.
                    loss = (
                        raw_loss
                        * (
                            len(y)
                            / total_bs
                        )
                    )

                scaler.scale(
                    loss
                ).backward()

                bs = len(y)

                loss_sum += (
                    float(
                        raw_loss.detach()
                    )
                    * bs
                )

                correct += int(
                    (
                        logits
                        .detach()
                        .argmax(1)
                        == y
                    )
                    .sum()
                    .item()
                )

                seen += bs

            scaler.step(
                optimizer
            )

            scaler.update()

            ema_update(
                ef,
                mf,
                ints,
            )

            global_step += 1

        sec = (
            time.perf_counter()
            - epoch_t0
        )

        print(
            f"[VGG] "
            f"e{epoch+1:02d}/{VGG_EPOCHS} "
            f"MIX | "
            f"loss={loss_sum/seen:.4f} | "
            f"top1={correct/seen:.4f} | "
            f"mixup={n_mixup} | "
            f"cutmix={n_cutmix} | "
            f"{sec:.1f}s"
        )

    expected_updates = (
        VGG_EPOCHS
        * updates_per_epoch
    )

    assert global_step == expected_updates, (
        global_step,
        expected_updates,
    )

    torch.cuda.synchronize(
        device
    )

    train_sec = (
        time.perf_counter()
        - train_t0
    )

    state = cpu_state_dict(
        ema
    )

    peak_gb = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / 2**30
    )

    print(
        f"[VGG] TRAIN DONE e9 | "
        f"{train_sec/60:.2f} min | "
        f"peak={peak_gb:.2f} GB"
    )

    del model
    del ema
    del loader
    del optimizer
    del scaler
    del ef
    del mf
    del ints

    gc.collect()
    torch.cuda.empty_cache()

    return (
        state,
        train_sec,
        peak_gb,
    )


r42_model = build_r101(SEED_R42)
r1337_model = build_r101(SEED_R1337)
inc_model = build_inception(SEED_INC)
vgg_model = build_vgg(SEED_VGG)


# ============================================================
# 8. PARALLEL TRAINING PIPELINES
#
# GPU0:
#   R42 -> R1337
#
# GPU1:
#   Inception -> VGG e9
# ============================================================

def gpu0_pipeline():

    r42_state, r42_train_sec = (
        train_full_model(
            name="R42",
            kind="r101",
            model=r42_model,
            device_idx=0,
            size=R_IMG,
            seed=SEED_R42,
        )
    )

    r1337_state, r1337_train_sec = (
        train_full_model(
            name="R1337",
            kind="r101",
            model=r1337_model,
            device_idx=0,
            size=R_IMG,
            seed=SEED_R1337,
        )
    )

    return {
        "r42_state":
            r42_state,

        "r1337_state":
            r1337_state,

        "r42_train_sec":
            r42_train_sec,

        "r1337_train_sec":
            r1337_train_sec,
    }


def gpu1_pipeline():

    inc_state, inc_train_sec = (
        train_full_model(
            name="INC",
            kind="inception",
            model=inc_model,
            device_idx=1,
            size=I_IMG,
            seed=SEED_INC,
        )
    )

    (
        vgg_state,
        vgg_train_sec,
        vgg_peak_gb,
    ) = train_vgg_e9(
        model=vgg_model,
        device_idx=1,
        seed=SEED_VGG,
    )

    return {
        "inc_state":
            inc_state,

        "vgg_state":
            vgg_state,

        "inc_train_sec":
            inc_train_sec,

        "vgg_train_sec":
            vgg_train_sec,

        "vgg_peak_gb":
            vgg_peak_gb,
    }


train_wall_t0 = (
    time.perf_counter()
)


with ThreadPoolExecutor(
    max_workers=2
) as ex:

    f0 = ex.submit(
        gpu0_pipeline
    )

    f1 = ex.submit(
        gpu1_pipeline
    )

    gpu0 = f0.result()
    gpu1 = f1.result()


train_wall_sec = (
    time.perf_counter()
    - train_wall_t0
)

del r42_model
del r1337_model
del inc_model
del vgg_model

gc.collect()

for device_idx in range(2):
    with torch.cuda.device(device_idx):
        torch.cuda.empty_cache()


print()
print("=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)

print(
    "wall:",
    f"{train_wall_sec/60:.2f} min",
)

print(
    "R42:",
    f"{gpu0['r42_train_sec']/60:.2f} min",
)

print(
    "R1337:",
    f"{gpu0['r1337_train_sec']/60:.2f} min",
)

print(
    "INC:",
    f"{gpu1['inc_train_sec']/60:.2f} min",
)

print(
    "VGG e9:",
    f"{gpu1['vgg_train_sec']/60:.2f} min",
)

print(
    "VGG peak:",
    f"{gpu1['vgg_peak_gb']:.2f} GB",
)

print("=" * 100)


def load_r101_state(state, device_idx):
    m = models.resnet101(weights=None)

    n = m.fc.in_features

    m.fc = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(n, NUM_CLASSES),
    )

    m.load_state_dict(state)

    return (
        m.to(torch.device(f"cuda:{device_idx}"))
        .half()
        .to(memory_format=torch.channels_last)
        .eval()
    )


def load_inc_state(state, device_idx):
    m = models.inception_v3(
        weights=None,
        aux_logits=True,
        transform_input=True,
        init_weights=False,
    )

    n = m.fc.in_features

    m.fc = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(n, NUM_CLASSES),
    )

    aux_n = m.AuxLogits.fc.in_features

    m.AuxLogits.fc = nn.Linear(
        aux_n,
        NUM_CLASSES,
    )

    m.load_state_dict(state)

    return (
        m.to(torch.device(f"cuda:{device_idx}"))
        .half()
        .to(memory_format=torch.channels_last)
        .eval()
    )


def load_vgg_state(state, device_idx):
    m = models.vgg16_bn(weights=None)

    m.avgpool = nn.AdaptiveAvgPool2d(
        (1, 1)
    )

    m.classifier = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(512, NUM_CLASSES),
    )

    m.load_state_dict(state)

    return (
        m.to(torch.device(f"cuda:{device_idx}"))
        .half()
        .to(memory_format=torch.channels_last)
        .eval()
    )


@torch.inference_mode()
def predict_tta(
    *,
    model,
    device_idx,
    size,
    batch,
    tta,
):
    device = torch.device(
        f"cuda:{device_idx}"
    )

    torch.cuda.set_device(
        device_idx
    )

    loader = make_test_loader(
        size,
        batch,
    )

    probs_all = []
    ids_all = []

    warm_it = iter(loader)
    x0, _ = next(warm_it)

    x0 = (
        x0.to(device, non_blocking=True)
        .half()
        .contiguous(
            memory_format=torch.channels_last
        )
    )

    k0, flip0 = tta[0]

    xv0 = torch.rot90(
        x0,
        k0,
        dims=(2, 3),
    )

    if flip0:
        xv0 = torch.flip(
            xv0,
            dims=(3,),
        )

    xv0 = xv0.contiguous(
        memory_format=torch.channels_last
    )

    out0 = model(xv0)
    logits0, _ = main_aux(out0)

    torch.cuda.synchronize(device)

    del warm_it, x0, xv0, out0, logits0

    t0 = time.perf_counter()

    for x, ids in loader:
        x = (
            x.to(device, non_blocking=True)
            .half()
            .contiguous(
                memory_format=torch.channels_last
            )
        )

        ps = None

        for k, flip in tta:
            xv = torch.rot90(
                x,
                k,
                dims=(2, 3),
            )

            if flip:
                xv = torch.flip(
                    xv,
                    dims=(3,),
                )

            xv = xv.contiguous(
                memory_format=torch.channels_last
            )

            out = model(xv)
            logits, _ = main_aux(out)

            p = torch.softmax(
                logits.float(),
                dim=1,
            )

            ps = (
                p
                if ps is None
                else ps + p
            )

        probs = ps / len(tta)

        probs_all.append(
            probs.cpu().numpy()
        )

        ids_all.extend(
            [str(z) for z in ids]
        )

    torch.cuda.synchronize(device)

    sec = (
        time.perf_counter()
        - t0
    )

    probs = np.concatenate(
        probs_all,
        axis=0,
    ).astype(np.float32)

    assert probs.shape == (
        len(test_df),
        NUM_CLASSES,
    )

    assert (
        ids_all
        ==
        test_df["id"].astype(str).tolist()
    )

    return probs, sec


infer_wall_t0 = time.perf_counter()


def infer_r1337():
    m = load_r101_state(
        gpu0["r1337_state"],
        0,
    )

    probs, sec = predict_tta(
        model=m,
        device_idx=0,
        size=R_IMG,
        batch=R_TEST_BATCH,
        tta=TTA_D4,
    )

    del m
    gc.collect()

    with torch.cuda.device(0):
        torch.cuda.empty_cache()

    return probs, sec


def infer_inc():
    m = load_inc_state(
        gpu1["inc_state"],
        1,
    )

    probs, sec = predict_tta(
        model=m,
        device_idx=1,
        size=I_IMG,
        batch=I_TEST_BATCH,
        tta=TTA_D4,
    )

    del m
    gc.collect()

    with torch.cuda.device(1):
        torch.cuda.empty_cache()

    return probs, sec


with ThreadPoolExecutor(max_workers=2) as ex:
    f_r1337 = ex.submit(infer_r1337)
    f_inc = ex.submit(infer_inc)

    p_r1337, sec_r1337 = f_r1337.result()
    p_inc, sec_inc = f_inc.result()


def infer_vgg():
    m = load_vgg_state(
        gpu1["vgg_state"],
        0,
    )

    probs, sec = predict_tta(
        model=m,
        device_idx=0,
        size=VGG_IMG,
        batch=VGG_TEST_BATCH,
        tta=TTA_D1,
    )

    del m
    gc.collect()

    with torch.cuda.device(0):
        torch.cuda.empty_cache()

    return probs, sec


def infer_r42():
    m = load_r101_state(
        gpu0["r42_state"],
        1,
    )

    probs, sec = predict_tta(
        model=m,
        device_idx=1,
        size=R_IMG,
        batch=R_TEST_BATCH,
        tta=TTA_D2,
    )

    del m
    gc.collect()

    with torch.cuda.device(1):
        torch.cuda.empty_cache()

    return probs, sec


with ThreadPoolExecutor(max_workers=2) as ex:
    f_vgg = ex.submit(infer_vgg)
    f_r42 = ex.submit(infer_r42)

    p_vgg, sec_vgg = f_vgg.result()
    p_r42, sec_r42 = f_r42.result()


infer_wall_sec = (
    time.perf_counter()
    - infer_wall_t0
)


print(
    "inference seconds:",
    round(infer_wall_sec, 2),
)

if infer_wall_sec > 120:
    raise RuntimeError(
        f"Inference exceeded 120 seconds: {infer_wall_sec:.2f}"
    )


assert p_r42.shape == (
    len(test_df),
    NUM_CLASSES,
)

assert p_r1337.shape == (
    len(test_df),
    NUM_CLASSES,
)

assert p_inc.shape == (
    len(test_df),
    NUM_CLASSES,
)

assert p_vgg.shape == (
    len(test_df),
    NUM_CLASSES,
)


s = sum(FINAL_WEIGHTS.values())

w = {
    k: v / s
    for k, v in FINAL_WEIGHTS.items()
}


final_probs = (
    w["R42"] * p_r42
    +
    w["R1337"] * p_r1337
    +
    w["INC"] * p_inc
    +
    w["VGG"] * p_vgg
)

pred = final_probs.argmax(
    axis=1
).astype(int)


submission = sample_submission.copy()

target_cols = [
    c
    for c in submission.columns
    if c != "id"
]

if len(target_cols) != 1:
    raise RuntimeError(
        "Expected exactly one target column."
    )

submission[
    target_cols[0]
] = pred


assert len(submission) == len(test_df)
assert not submission[target_cols[0]].isna().any()
assert submission[target_cols[0]].between(0, 19).all()


submission.to_csv(
    "final_submission.csv",
    index=False,
)


total_wall_sec = (
    train_wall_sec
    +
    infer_wall_sec
)

print(
    "training minutes:",
    round(train_wall_sec / 60, 2),
)

print(
    "inference seconds:",
    round(infer_wall_sec, 2),
)

print(
    "total minutes:",
    round(total_wall_sec / 60, 2),
)

if total_wall_sec > 45 * 60:
    raise RuntimeError(
        f"Total runtime exceeded 45 minutes: {total_wall_sec/60:.2f}"
    )

print("final_submission.csv")
