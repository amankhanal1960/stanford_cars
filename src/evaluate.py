from sklearn.metrics import accuracy_score
from data_loader import val_generator
import tensorflow as tf

model = tf.keras.models.load_model("best_efficientnet_finetuned.keras")
predictions = model.predict(val_generator, steps=len(val_generator), verbose=1)
y_pred = predictions.argmax(axis=1)
y_true = val_generator.classes

val_accuracy = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
