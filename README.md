# LeafNet-MCMD

**Multi-Crop, Multi-Disease leaf classification with a 0.5M-parameter CNN built from scratch**

Depthwise-separable convolutions · Squeeze-and-excitation attention · Grad-CAM explainability

`Python 3.10+` · `TensorFlow 2.x` · `Keras 3.13` · `MIT License`

---

## Why this project

The standard recipe for PlantVillage is to fine-tune an ImageNet backbone and report the number. It works, but it produces a model of 5M–138M parameters — awkward on exactly the hardware that would run it in the field: a phone, a Raspberry Pi, a drone payload.

**LeafNet-MCMD goes the other way.** The architecture is written from scratch at roughly **502K parameters**, using depthwise-separable convolutions with squeeze-and-excitation attention to keep capacity where it matters and strip it everywhere else.

Small models are also easier to interrogate, which is the second half of the project: every prediction can be traced with Grad-CAM, and every error in the test set is captured, labelled and audited.

---

## Highlights

- **Custom architecture, no pretrained weights.** `FastMicroBlock` — depthwise conv → pointwise conv → SE channel gate — implemented from first principles.
- **~502K parameters.** 7× smaller than MobileNetV2, 51× smaller than ResNet50.
- **38 classes, 14 crop species.** Apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, tomato.
- **Explainability built in.** Grad-CAM heatmaps generated automatically for every class.
- **Error auditing, not just metrics.** Every misclassification is exported as a labelled image plus a CSV with true class, predicted class and confidence.
- **Full multiclass evaluation.** Per-class precision/recall/F1, 38×38 confusion matrix, macro-averaged one-vs-rest ROC.

---

## Parameter efficiency

| Model | Parameters | vs. LeafNet-MCMD |
|---|---:|---:|
| **LeafNet-MCMD** | **~502 K** | **1×** |
| MobileNetV2 | 3.5 M | 7× larger |
| EfficientNetB0 | 5.3 M | 11× larger |
| ResNet50 | 25.6 M | 51× larger |
| VGG16 | 138.4 M | 276× larger |

*Reference architectures at their standard ImageNet configurations. At float32, LeafNet-MCMD is roughly 2 MB on disk.*

---

## Architecture

```
Input (224 x 224 x 3)
    |
    +- Rescaling(1/255)
    +- Conv2D(32, 3x3, stride 2) -> BatchNorm -> ReLU     ->  112 x 112 x  32
    |
    +- FastMicroBlock(64)   -> MaxPool                    ->   56 x  56 x  64
    +- FastMicroBlock(128)  -> MaxPool                    ->   28 x  28 x 128
    +- FastMicroBlock(256)  -> MaxPool                    ->   14 x  14 x 256
    +- FastMicroBlock(512)                                ->   14 x  14 x 512
    |
    +- GlobalAveragePooling2D                             ->        512
    +- Dropout(0.4) -> Dense(256, ReLU) -> Dropout(0.3)
    +- Dense(38, softmax)
```

### Inside a `FastMicroBlock(f)`

```
  x --> DepthwiseConv2D(3x3) -> BatchNorm -> ReLU
    --> Conv2D(f, 1x1)       -> BatchNorm ----+
                                              |
        SE gate:  GlobalAvgPool               |
                  -> Dense(f/4, ReLU)         |
                  -> Dense(f,   sigmoid) -----+
                                              |
                                              v
                                       x * gate -> ReLU
```

Two ideas do the work here:

**Depthwise separability.** A standard 3x3 convolution from *C* input channels to *f* outputs costs `9·C·f` weights. Factoring it into a per-channel spatial filter (`9·C`) followed by a 1x1 channel mixer (`C·f`) cuts that by close to the kernel area — a ~9x reduction at the same receptive field.

**Squeeze-and-excitation.** Global average pooling collapses each feature map to a single value, a bottleneck MLP turns those into per-channel gains, and the block rescales itself accordingly. This gives an otherwise-local network a global view — which matters when the discriminative signal is a small lesion on an otherwise uniform leaf.

### Parameter budget

| Stage | Parameters |
|---|---:|
| Stem (Conv + BN) | 1,024 |
| FastMicroBlock(64) | 4,848 |
| FastMicroBlock(128) | 17,888 |
| FastMicroBlock(256) | 68,544 |
| FastMicroBlock(512) | 268,160 |
| Dense(256) | 131,328 |
| Classifier (38) | 9,766 |
| **Total** | **501,558** |

---

## Training configuration

| Setting | Value |
|---|---|
| Optimiser | Adam, lr = 1e-4 |
| Loss | Sparse categorical crossentropy |
| Input resolution | 224 x 224 x 3 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience 12, restore best weights |
| LR schedule | ReduceLROnPlateau, patience 5, factor 0.5 |
| Checkpointing | Best validation accuracy |
| Split | 70 / 15 / 15 train / val / test |

> **Note:** `Rescaling(1./255)` is a **layer inside the model**, not a preprocessing step. Do not divide inputs by 255 before calling `predict()` — see [Implementation notes](#implementation-notes).

---

## Quickstart

```bash
git clone https://github.com/PSVedant/leafnet_mcmd_gradcam.git
cd leafnet_mcmd_gradcam
pip install -r requirements.txt
```

**Dataset.** Download *New Plant Diseases Dataset (Augmented)* from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and unzip it into the repository root, so that `New Plant Diseases Dataset(Augmented)/` sits alongside `src/`. It is ~87,000 images and is not tracked here.

**Run the pipeline.**

```bash
python src/setup_folders.py     # stage raw data  ->  data/raw/
python src/split_dataset.py     # build splits    ->  data/{train,val,test}/
python src/custom_cnn.py        # train           ->  outputs_novel/
python src/evaluate_custom.py   # metrics, confusion matrix, ROC
python src/gradcam.py           # heatmaps for all 38 classes
python src/analyze_errors.py    # misclassification audit
```

Optional transfer-learning comparator:

```bash
python src/train_model.py       # EfficientNetB0, frozen backbone
```

---

## Pipeline

```
  Kaggle archive
        |
        v
  setup_folders.py ---> data/raw/                staging
        |
        v
  split_dataset.py ---> data/train
                        data/val                 70 / 15 / 15
                        data/test
        |
        v
  custom_cnn.py    ---> best_model.keras         training + checkpointing
        |               training_curves.png
        |
        +---> evaluate_custom.py ---> classification_report.txt
        |                             confusion_matrix.png
        |                             roc_curve.png
        |
        +---> gradcam.py         ---> gradcam/CLASSNAME_gradcam.jpg
        |
        +---> analyze_errors.py  ---> misclassifications_table.csv
                                      error_analysis/images/
```

---

## Evaluation

`evaluate_custom.py` produces the full multiclass picture rather than a single headline figure:

- **Per-class precision, recall and F1** across all 38 classes, so weak categories stay visible instead of being averaged away
- **38 x 38 confusion matrix**, which surfaces *which* diseases get confused — errors cluster into genuinely similar pairs, such as the several tomato blights, rather than scattering at random
- **Macro-averaged one-vs-rest ROC** with per-class AUC, interpolated onto a common FPR grid

`analyze_errors.py` goes further and audits every single mistake, writing each misclassified image to disk annotated with its true label, predicted label and the model's confidence — alongside a CSV of the same. High-confidence errors are the informative ones: they point at real class ambiguity rather than noise.

| Metric | Value |
|---|---|
| Classes | 38 |
| Test images | — |
| Test accuracy | — |
| Macro F1 | — |
| Macro ROC-AUC | — |
| CPU latency / image | — |

---

## Explainability

Accuracy alone cannot distinguish a model that learned pathology from one that learned the photographic backdrop. PlantVillage images are captured against uniform backgrounds, which makes that failure mode a live risk.

`gradcam.py` addresses it directly. It locates the final spatial layer automatically, computes the gradient of the predicted class score with respect to that layer's feature maps, pools those gradients into per-channel importances, and projects the weighted activation back onto the input as a heatmap.

The output is one visualisation per class, written to `outputs_novel/gradcam/`. A heatmap concentrated on a lesion is evidence the model is looking at the disease; a heatmap smeared across the background is evidence it is not.

---

## Repository structure

```
├── src/
│   ├── setup_folders.py      Stage the raw Kaggle download
│   ├── split_dataset.py      Build train / val / test splits
│   ├── load_data.py          Dataset loading sanity check
│   ├── custom_cnn.py         FastMicroBlock architecture + training
│   ├── train_model.py        EfficientNetB0 transfer-learning comparator
│   ├── evaluate_custom.py    Metrics, confusion matrix, ROC
│   ├── gradcam.py            Grad-CAM heatmaps
│   └── analyze_errors.py     Misclassification audit
├── notebooks/
│   └── evaluate.ipynb
├── models/
├── outputs_novel/
└── requirements.txt
```

### Outputs

| Path | Contents |
|---|---|
| `outputs_novel/best_model.keras` | Best checkpoint by validation accuracy |
| `outputs_novel/training_curves.png` | Accuracy and loss curves |
| `outputs_novel/classification_report.txt` | Per-class precision, recall, F1 |
| `outputs_novel/confusion_matrix.png` | 38 x 38 confusion matrix |
| `outputs_novel/roc_curve.png` | Macro-averaged multiclass ROC |
| `outputs_novel/gradcam/` | One heatmap per class |
| `outputs_novel/error_analysis/` | Misclassification CSV and labelled images |

---

## Implementation notes

**Do not rescale inputs manually.** The model contains `Rescaling(1./255)` as its second layer. Dividing images by 255 before `predict()` scales them twice and collapses accuracy to near-chance.

**`FastMicroBlock` must be passed when loading.** It is a custom layer:

```python
model = tf.keras.models.load_model(
    "outputs_novel/best_model.keras",
    custom_objects={"FastMicroBlock": FastMicroBlock},
)
```

**Use `shuffle=False` when evaluating.** `analyze_errors.py` relies on `test_ds.file_paths` staying aligned with prediction order.

---

## Roadmap

- [ ] Benchmarked CPU inference latency and on-disk size
- [ ] Head-to-head sweep against MobileNetV2, EfficientNetB0 and ResNet50
- [ ] TensorFlow Lite export with post-training quantisation
- [ ] Consolidate the `FastMicroBlock` definition into a single shared module
- [ ] Cross-dataset evaluation on field-condition imagery

---

## Team

                                                     Contribution 
|---|---|
| [@PSVedant](https://github.com/PSVedant) |         Co-designed the architecture; led the training pipeline, data preparation and hyperparameter tuning |
| [@ArjunKhimta](https://github.com/ArjunKhimta) |   Co-designed the architecture; led evaluation, Grad-CAM explainability and error analysis |

---

## Dataset

Hughes, D.P. and Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.

Augmented variant redistributed on Kaggle as **New Plant Diseases Dataset (Augmented)**.
