import sys
import os
import cv2
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

paths=["./reporter","./posture detector"]
# Add folder path to sys.path
module_path = "./reporter"
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

# Now you can import your module
from telegram import TelegramImageSender

# Now you can import your module
from hand_bbox import HandBoundingBox


