import os
import scipy.io as sio
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

BASE_DIR = "/mnt/e/ML/Stanford Cars/Dataset/"


def load_dataset_dataframe(base_dir, anno_file_name):
    devkit_dir = os.path.join(base_dir, "car_devkit/devkit")

    # loading the metadata
    meta = sio.loadmat(os.path.join(devkit_dir, "cars_meta.mat"))
    class_names = [c[0] for c in meta["class_names"][0]]

    # loading the training annotations
    annos = sio.loadmat(os.path.join(devkit_dir, anno_file_name))
    annotations = annos["annotations"][0]

    data = []
    for anno in annotations:
        fname = anno["fname"][0]
        if "class" in anno.dtype.names:
            class_id = anno["class"][0][0] - 1
            class_name = class_names[class_id]
        else:
            class_name = None

        data.append({"filename": fname, "class_name": class_name})

    return pd.DataFrame(data)


df_train_full = load_dataset_dataframe(
    base_dir=BASE_DIR, anno_file_name="cars_train_annos.mat"
)
df_test = load_dataset_dataframe(
    base_dir=BASE_DIR, anno_file_name="cars_test_annos.mat"
)

##############-----CROPPING AND RESIZING IMAGES TO 300x300-----##############
# def crop_and_save_locally(df, source_folder_name, output_folder_name):
#     # Source path on E: drive
#     src_dir = os.path.join(BASE_DIR, source_folder_name)
#     # Fast native Linux destination path 
#     dest_dir = os.path.join(BASE_DIR, output_folder_name)
#     os.makedirs(dest_dir, exist_ok=True)

#     print(f"Processing {source_folder_name}...")
#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         img_path = os.path.join(src_dir, row['filename'])
#         img = cv2.imread(img_path)
#         if img is None: 
#             continue
            
#         # Extract your newly populated coordinates
#         x1, y1, x2, y2 = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
        
#         # Crop to the car and resize immediately to 300x300
#         cropped = img[y1:y2, x1:x2]
        
#         # Safe fallback check in case bounding box values are clipped or corrupted
#         if cropped.size == 0:
#             resized = cv2.resize(img, (300, 300))
#         else:
#             resized = cv2.resize(cropped, (300, 300))
            
#         # Write directly to native Linux SSD
#         cv2.imwrite(os.path.join(dest_dir, row['filename']), resized)

# # Run the processing sequence for both split folders
# # (Assuming your image subfolders are named 'cars_train' and 'cars_test')
# crop_and_save_locally(df_train_full, "cars_train", "train")
# crop_and_save_locally(df_test, "cars_test", "test")

df_train, df_val = train_test_split(
    df_train_full, test_size=0.2, random_state=42, stratify=df_train_full["class_name"]
)

print(
    f"Train samples: {len(df_train)} | Val samples: {len(df_val)} | Test samples: {len(df_test)}"
)

# chqange this according to the new folder names if you have changed them while cropping and resizing images
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "cars_train", "cars_train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "cars_test", "cars_test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
# validation/Test sets remains clean
val_test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=TRAIN_IMG_DIR,
    x_col="filename",
    y_col="class_name",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_val,
    directory=TRAIN_IMG_DIR,
    x_col="filename",
    y_col="class_name",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=TEST_IMG_DIR,
    x_col="filename",
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)

print("\n Pipeline Ready!")
