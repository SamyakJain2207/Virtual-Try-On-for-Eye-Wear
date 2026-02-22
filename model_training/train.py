# %% [markdown]
# # Virtual Try-On Model Training Pipeline (Local Environment)
# Ensure your input data is placed in the `data/input/` directory before running.

# %% 
# ==========================================
# CELL 1: Setup Local Directories
# ==========================================
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# Define local paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "UTKFace")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_DIR = os.path.join(BASE_DIR, "training_pipeline")
LABELS_MANUAL = os.path.join(BASE_DIR, "data", "raw", "labels_manual.csv")
LABELS_AUTO = os.path.join(BASE_DIR, "data", "raw", "labels_auto.csv")
LABELS_MERGED = os.path.join(BASE_DIR, "data", "raw", "labels_merged.csv")

# Create the directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PIPELINE_DIR, exist_ok=True)

print("Folders created successfully at:", BASE_DIR)


# %% 
# ==========================================
# CELL 2: Copy Pipeline Files
# ==========================================
import shutil

src_path = os.path.join(BASE_DIR, "data", "input", "all-the-files")
dst_path = PIPELINE_DIR

files_to_copy = [
    "annotation_tool.py", "annotation_tool_buttons.py",
    "annotation_tool_buttons_resume.py", "annotation_tool_collab.py",
    "cnn_trainer.py", "geometric_labeler.py", "run_option2_pipeline.py"
]

for filename in files_to_copy:
    source_file = os.path.join(src_path, filename)
    if os.path.exists(source_file):
        shutil.copy(source_file, os.path.join(dst_path, filename))
    else:
        pass # Silently skip if the user doesn't have these specific pipeline files locally

print("Pipeline files transfer step completed!")


# %% 
# ==========================================
# CELL 3: Copy UTKFace Dataset
# ==========================================
from tqdm import tqdm

src = os.path.join(BASE_DIR, "data", "input", "utkface-manual", "UTKFace")
dst = RAW_DATA_DIR

if os.path.exists(src):
    files = os.listdir(src)
    print(f"Found {len(files)} images. Copying to raw data folder...")
    for f in tqdm(files):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print("✔ Copied all UTKFace images into working directory!")
else:
    print(f"Warning: Dataset not found. Please place UTKFace images in: {src}")


# %%
# ==========================================
# CELL 4: Interactive Annotation Tool
# ==========================================
from IPython.display import display, clear_output
from PIL import Image
import ipywidgets as widgets
import csv, traceback

IMG_DISPLAY_SIZE = (640, 640)
LABELS = ["round","oval","square","heart","oblong"]

os.makedirs(os.path.dirname(LABELS_MANUAL), exist_ok=True)

def list_images(path):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    return sorted([f for f in os.listdir(path) if f.lower().endswith(exts)])

def load_labels(csv_path):
    labels = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                r = csv.reader(f)
                _ = next(r, None)
                for row in r:
                    if len(row) >= 2:
                        labels[row[0]] = row[1]
        except:
            traceback.print_exc()
    return labels

def save_labels(csv_path, data_dict):
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["filename","label"])
        for k,v in data_dict.items():
            w.writerow([k,v])
    os.replace(tmp, csv_path)

images = list_images(RAW_DATA_DIR)
labels_map = load_labels(LABELS_MANUAL)

if len(images) > 0:
    idx = 0
    for i,f in enumerate(images):
        if f not in labels_map:
            idx = i
            break

    out = widgets.Output()
    btns = {
        "round": widgets.Button(description="Round", button_style='primary'),
        "oval": widgets.Button(description="Oval", button_style='info'),
        "square": widgets.Button(description="Square", button_style=''),
        "heart": widgets.Button(description="Heart", button_style='danger'),
        "oblong": widgets.Button(description="Oblong", button_style='warning'),
    }
    btn_prev = widgets.Button(description="◀ Prev")
    btn_next = widgets.Button(description="Next ▶")
    btn_save = widgets.Button(description="Save CSV", button_style='success')

    idx_box = widgets.BoundedIntText(value=idx, min=0, max=len(images)-1, description="Index")
    fname_label = widgets.Label()
    progress_label = widgets.Label()
    current_label = widgets.Label()

    def show_image(i):
        out.clear_output()
        with out:
            fn = images[i]
            path = os.path.join(RAW_DATA_DIR, fn)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail(IMG_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                display(img)
            except Exception as e:
                print("Error loading:", fn, "->", e)
        idx_box.value = i
        fname_label.value = f"{i}/{len(images)-1} — {images[i]}"
        progress_label.value = f"Labeled: {len(labels_map)} / {len(images)}"
        current_label.value = labels_map.get(images[i], "UNLABELED")

    def apply_label(label):
        global idx
        labels_map[images[idx]] = label
        current_label.value = label
        next_i = idx + 1
        while next_i < len(images) and images[next_i] in labels_map:
            next_i += 1
        if next_i >= len(images):
            next_i = 0
        idx = next_i
        show_image(idx)

    for lbl,btn in btns.items():
        btn.on_click(lambda b, l=lbl: apply_label(l))

    def prev_img(b):
        global idx
        idx = max(0, idx-1)
        show_image(idx)

    def next_img(b):
        global idx
        idx = min(len(images)-1, idx+1)
        show_image(idx)

    def save_csv(b):
        save_labels(LABELS_MANUAL, labels_map)
        current_label.value = "SAVED ✔"

    btn_prev.on_click(prev_img)
    btn_next.on_click(next_img)
    btn_save.on_click(save_csv)

    display(widgets.HTML("<h3>Annotation Tool (Local Edition)</h3>"))
    display(fname_label)
    display(progress_label)
    display(current_label)
    display(widgets.HBox(list(btns.values())))
    display(widgets.HBox([btn_prev, btn_next, btn_save, idx_box]))
    display(out)

    show_image(idx)
else:
    print("No images found in UTKFace folder. Cannot start annotation tool.")


# %% 
# ==========================================
# CELL 5: Small Model Training (EfficientNetB0)
# ==========================================
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

SAVE_SMALL_MODEL = os.path.join(MODELS_DIR, "small_effnet_b0.h5")
IMG_SIZE_SMALL = (224, 224)
BATCH = 32
EPOCHS_SMALL = 12

if os.path.exists(LABELS_MANUAL):
    df = pd.read_csv(LABELS_MANUAL)
    print("Loaded manual labels:", len(df))

    df['filepath'] = df['filename'].apply(lambda f: os.path.join(RAW_DATA_DIR, f))
    label_names = ['round','oval','square','heart','oblong']
    label_to_idx = {l:i for i,l in enumerate(label_names)}
    df['label_idx'] = df['label'].map(label_to_idx)
    df = df.dropna().reset_index(drop=True)

    filepaths = df['filepath'].values
    labels = df['label_idx'].values.astype(np.int32)

    def parse_fn(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE_SMALL)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels)).shuffle(len(filepaths), seed=42)
    train_count = int(0.8 * len(filepaths))

    train_ds = ds.take(train_count).map(parse_fn).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    val_ds   = ds.skip(train_count).map(parse_fn).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    base = EfficientNetB0(include_top=False, input_shape=IMG_SIZE_SMALL+(3,), pooling='avg')
    base.trainable = False

    inp = layers.Input(shape=IMG_SIZE_SMALL+(3,))
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(label_names), activation='softmax')(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(SAVE_SMALL_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    print("Starting small model training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_SMALL, callbacks=callbacks)
    print("Small model saved to:", SAVE_SMALL_MODEL)
else:
    print(f"Manual labels not found at {LABELS_MANUAL}. Please annotate data first.")


# %% 
# ==========================================
# CELL 6: Auto-Labeling
# ==========================================
if os.path.exists(SAVE_SMALL_MODEL):
    model = tf.keras.models.load_model(SAVE_SMALL_MODEL)
    files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    label_names = ['round','oval','square','heart','oblong']

    rows = []
    for f in tqdm(files, desc="Auto-labeling"):
        path = os.path.join(RAW_DATA_DIR, f)
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE_SMALL)
            img = tf.expand_dims(img, 0) / 255.0

            preds = model.predict(img, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])

            rows.append((f, label_names[idx], conf))
        except Exception as e:
            rows.append((f, "error", 0.0))

    df_auto = pd.DataFrame(rows, columns=["filename","label","confidence"])
    df_auto.to_csv(LABELS_AUTO, index=False)

    print("🔥 Auto-labeling complete! Saved to:", LABELS_AUTO)
else:
    print("⚠️ Small model not found. Run training step first!")


# %% 
# ==========================================
# CELL 7: Merge Manual + Auto Labels
# ==========================================
CONF_THRESH = 0.20

if os.path.exists(LABELS_MANUAL) and os.path.exists(LABELS_AUTO):
    df_manual = pd.read_csv(LABELS_MANUAL)
    df_auto = pd.read_csv(LABELS_AUTO)

    df_auto_filtered = df_auto[df_auto['confidence'] >= CONF_THRESH].copy()

    manual_map = dict(zip(df_manual['filename'], df_manual['label']))
    auto_map   = dict(zip(df_auto_filtered['filename'], df_auto_filtered['label']))

    merged = []
    all_files = sorted(set(list(auto_map.keys()) + list(manual_map.keys())))

    for f in all_files:
        if f in manual_map:
            merged.append((f, manual_map[f]))   
        else:
            merged.append((f, auto_map[f]))

    df_merged = pd.DataFrame(merged, columns=['filename','label'])
    df_merged.to_csv(LABELS_MERGED, index=False)

    print("🔥 Merged labels saved to:", LABELS_MERGED)
    print("Total merged:", len(df_merged))
else:
    print("⚠️ Need both manual and auto labels to merge.")


# %% 
# ==========================================
# CELL 8: Final EfficientNetB3 Training & Fine-Tuning
# ==========================================
from tensorflow.keras.applications import EfficientNetB3

SAVE_FINAL_MODEL = os.path.join(MODELS_DIR, "effnet_b3_final.h5")
IMG_SIZE_FINAL = (300, 300)
EPOCHS_FINAL = 15

if os.path.exists(LABELS_MERGED):
    df = pd.read_csv(LABELS_MERGED)
    label_names = ['round','oval','square','heart','oblong']
    label_to_idx = {l:i for i,l in enumerate(label_names)}
    df['label_idx'] = df['label'].map(label_to_idx)
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(RAW_DATA_DIR, x))
    df = df.dropna().reset_index(drop=True)

    filepaths = df['filepath'].values
    labels = df['label_idx'].values.astype(np.int32)

    def parse_fn_final(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE_FINAL)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels)).shuffle(len(filepaths), seed=42)
    train_count = int(0.8 * len(filepaths))

    train_ds = ds.take(train_count).map(parse_fn_final).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    val_ds   = ds.skip(train_count).map(parse_fn_final).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    base = EfficientNetB3(include_top=False, input_shape=IMG_SIZE_FINAL+(3,), pooling='avg')
    base.trainable = False

    inp = layers.Input(shape=IMG_SIZE_FINAL+(3,))
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(label_names), activation="softmax")(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(SAVE_FINAL_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    print("Starting Final Model Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINAL, callbacks=callbacks)
    print("Final base model saved to:", SAVE_FINAL_MODEL)

    # === Fine Tuning Phase ===
    print("Starting Fine-Tuning Phase...")
    base.trainable = True

    for layer in base.layers[:-80]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=callbacks)
    print("Fully Fine-tuned model saved to:", SAVE_FINAL_MODEL)
else:
    print("Merged labels not found. Run the merging step first.")