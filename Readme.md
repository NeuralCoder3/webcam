# Installation
Install the packages from `requirements.txt`, `v4l2loopback`, and `tensorflow`.

# Usage
1. Run `add_device` to register a fake webcam device.
2. Run `webcam.py`. This will load the deeplabv3 network for image segmentation and open a cv gui.

# Controls
| Key | Function |
| ---- | ------ |
| Q/Esc | Quit |
| Right Mouse Button | Next background from local `.jpg` |
| O | Open background chooser |


Based on https://github.com/neatpun/real-time-background-replacement
