from data_loader import train_generator, val_generator
from model import build_efficientnet_model
from model import apply_fine_tuning
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf


# phase 1 - training only the custom classifier head with frozen efficientnet base model for 15 epochs

num_classes = len(train_generator.class_indices)


model = build_efficientnet_model(num_classes)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(
        filepath="best_efficientnet.keras", monitor="val_loss", save_best_only=True
    ),
]

history = model.fit(
    train_generator, validation_data=val_generator, epochs=15, callbacks=callbacks, steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

print("Training phase 1 complete! Starting fine-tuning...")



# phase 2  - unfreezing last 40 layers of efficientnet and training with a lower learning rate and label smoothing to prevent overfitting and improve generalization

best_phase1_model = tf.keras.models.load_model("best_efficientnet.keras")

model_finetuned = apply_fine_tuning(best_phase1_model)

callbacks_finetune = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(
        filepath="best_efficientnet_finetuned.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=4,
        verbose=1,
        min_lr=1e-7,
    )
]
history_finetune = model_finetuned.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks_finetune,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)