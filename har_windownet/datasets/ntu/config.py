"""
Constants for NTU → Window conversion: projection dimensions and defaults.

Use RGB or depth projection when available for stable 2D normalized coordinates.
3D normalization by scene range is unstable across samples (camera distance varies).
"""

# RGB frame size (Kinect V2 / NTU); normalize color_xy by these for [0, 1]
RGB_WIDTH = 1920
RGB_HEIGHT = 1080

# Depth frame size (Kinect V2); normalize depth_xy by these for [0, 1]
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424

# Projection modes for CLI: --projection rgb|depth|3d
PROJECTION_RGB = "rgb"
PROJECTION_DEPTH = "depth"
PROJECTION_3D = "3d"

# Default: rgb when available, else depth, else 3d (document 3d as unstable)
DEFAULT_PROJECTION = PROJECTION_RGB
