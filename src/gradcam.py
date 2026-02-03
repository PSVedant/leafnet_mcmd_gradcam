import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "leafnet_custom_cnn.keras"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
OUTPUT_DIR = "outputs/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")
print(f"Model type: {type(model)}")

# Build the model explicitly
dummy_input = tf.zeros((1, 224, 224, 3))
_ = model(dummy_input, training=False)
print("Model built successfully")

# ---------------- CREATE GRADCAM MODEL (SKIP RESCALING LAYER) ---------------- #
# The Rescaling layer blocks gradients, so we'll create a model that skips it
last_conv_layer_name = "conv2d_3"  # From your diagnostic output
print(f"Using last conv layer: {last_conv_layer_name}")

# Create a new model that takes input AFTER the rescaling layer
# This is the key fix!
def get_gradcam_model(model, layer_name):
    """
    Create a model for GradCAM that skips the Rescaling layer
    """
    # Get the rescaling layer output shape (after rescaling)
    rescaling_layer = model.layers[0]
    
    # Create new input that matches what comes after rescaling
    new_input = tf.keras.Input(shape=(224, 224, 3))
    
    # Pass through all layers except the first rescaling layer
    x = new_input
    layer_found = False
    target_layer_output = None
    
    for i, layer in enumerate(model.layers[1:], 1):  # Skip rescaling layer
        x = layer(x)
        if layer.name == layer_name:
            target_layer_output = x
            layer_found = True
    
    if not layer_found:
        raise ValueError(f"Layer {layer_name} not found")
    
    # Create the GradCAM model
    grad_model = tf.keras.Model(inputs=new_input, outputs=[target_layer_output, x])
    return grad_model

grad_model = get_gradcam_model(model, last_conv_layer_name)
print("GradCAM model created successfully")

# ---------------- GRADCAM FUNCTION ---------------- #
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    """
    Generate GradCAM heatmap
    Note: img_array should already be rescaled (0-1 range)
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # Get the score for the predicted class
        class_channel = predictions[:, pred_index]
    
    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise ValueError("Gradients are None. Cannot compute GradCAM.")
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the pooled gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Normalize between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    return heatmap


def save_and_display_gradcam(img, heatmap, cam_path, alpha=0.4):
    """
    Save GradCAM visualization
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize jet to match image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1-alpha, jet, alpha, 0)
    
    # Save the superimposed image
    cv2.imwrite(cam_path, superimposed_img)
    return superimposed_img


# ---------------- PROCESS IMAGES ---------------- #
print("\nGenerating Grad-CAM visualizations...")

processed_count = 0
error_count = 0

for cls in sorted(os.listdir(TEST_DIR)):
    cls_path = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    
    # Get first image from class
    img_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"⊘ No images found in {cls}")
        continue
        
    img_name = img_files[0]
    img_path = os.path.join(cls_path, img_name)
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⊘ Failed to load {img_path}")
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    # Prepare for model (normalize to 0-1)
    # Since we're skipping the rescaling layer in grad_model, we need to rescale here
    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
    
    # Get prediction from original model
    preds = model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    
    # Generate heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, grad_model, pred_class)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        
        # Save visualization
        save_path = os.path.join(OUTPUT_DIR, f"{cls}_gradcam.png")
        save_and_display_gradcam(img_resized, heatmap_resized, save_path, alpha=0.4)
        
        print(f"✓ {cls}: class {pred_class} ({confidence:.3f})")
        processed_count += 1
    
    except Exception as e:
        print(f"✗ {cls}: {str(e)}")
        error_count += 1
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(f"Grad-CAM Generation Complete!")
print(f"✓ Successfully processed: {processed_count}/{processed_count + error_count}")
print(f"✗ Errors: {error_count}")
print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"{'='*60}")