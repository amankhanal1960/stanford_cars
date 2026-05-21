import tensorflow as tf
from tensorflow.keras import layers, models



##########-----------FREEZED EFFICIENTNETB3 WITH CUSTOM CLASSIFIER HEAD-----------##########
def build_efficientnet_model(num_classes):
    # loading EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB3(
        # top=False means output layer removed, we will add our own output layer
        include_top=False,
        weights="imagenet",
        input_shape=(300, 300, 3),
    )

    base_model.trainable = False

    phase1_model = models.Sequential(
        [
            base_model,
            layers.BatchNormalization(),  # added batch normalization layer to stabilize training and improve convergence
            layers.GlobalAveragePooling2D(),  # converts the 3d image to flat 1d vector so that it can be fed to the dense layer
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        ]
    )

    phase1_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return phase1_model


#############-----------UNFREEZING LAST 40 LAYERS OF EFFICIENTNETB3 FOR FINE-TUNING-----------##########
def apply_fine_tuning(trained_model):

    base_model = trained_model.layers[0]

    base_model.trainable = True

    # Freeze the BatchNormalization layers
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    fine_tune = len(base_model.layers) - 40
    for layer in base_model.layers[:fine_tune]:
        layer.trainable = False

    trained_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-6, weight_decay=0.01),
        # instead of categorical crossentropy which gave 71% uisng label_smoothing
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
        metrics=["accuracy"],
    )

    return trained_model
