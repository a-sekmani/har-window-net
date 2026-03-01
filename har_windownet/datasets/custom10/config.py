"""Constants for Custom10 adapter: device/camera IDs, formats, default image size and fps."""

DEVICE_ID = "custom10-offline"
CAMERA_ID = "custom10-cam"

# Format strings in clip metadata
FORMAT_COCO17_NORM = "coco17_norm"
FORMAT_COCO17_PIXEL = "coco17_pixel"

# Default image dimensions for pixel normalization (when not in metadata)
IMG_W = 1920
IMG_H = 1080

# Default fps when not in clip
FPS = 30.0
