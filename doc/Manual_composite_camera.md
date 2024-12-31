# Interfacing a composite camera with OpenCV on Linux

10/2021 - By Sébastien Mick

## Background & Goal

As a post-doctoral researcher at ETIS Lab, I was somewhat put in charge of Berenson, a human-sized robot consisting of a mobile wheelbase, a trenchcoat, an actuated head capable of producing facial expressions and a composite camera located in the right eye. My goal was to interface the robot with a facial expression recognition model in order to elaborate a human-robot interaction experimental setup. One of the main tasks was to access the video stream from this camera.

## Platform & Software Tools

Here is a list describing the development platform and tools I employed when working on this task. This is not a list of requirements, considering I mostly used what I had at hand. There are most likely alternative tools that would work as well to achieve the same result.

* Ubuntu 20.04 desktop
* Python 3.8.10
* OpenCV (Python API, version 4.5 3)
* V4L command-line tools (`v4l-utils` package)
* MPV (`mpv` package)

## Hardware

Sadly, I don’t have any product reference for the video hardware, which consists of:

* Composite camera (video-only)
* External 5V DC generator powering the camera
* Capture card digitizing the composite video input (yellow pin) into a USB output

The wiring of the video components was mostly done, and I only had to connect the power supply. If needed, it should not be too difficult to figure out how to connect the various components anyway. The camera’s red and black pins correspond to power supply, whereas the white pin corresponds to data output. The black and white pins can be wired to an RCA connector (preferably yellow, for conformity with the color coding) and plugged into the corresponding input of the capture card.

## Video Source Configuration

Regardless if the camera is properly working, plugging the capture card to a computer’s USB port should make it appear among the available devices. In a terminal, check the available video sources:

`ls /dev/video*`

If you have more than one video source connected to the computer, repeat this command before and after unplugging the capture card to identify its number. Then, V4L allows to detect the various inputs provided by the capture card available at `/dev/video<#>`:

`v4l2-ctl -d /dev/video<#> --list-inputs`

Here is an example of the command’s output:

```
ioctl: VIDIOC_ENUMINPUT
	Input       : 0
	Name        : S-Video
	Type        : 0x00000002 (Camera)
	Audioset    : 0x00000001
	Tuner       : 0x00000000
	Standard    : 0x0000000000FFFFFF
	Status      : 0x00000000 (ok)
	Capabilities: 0x00000004 (SDTV standards)

	Input       : 1
	Name        : Composite
	Type        : 0x00000002 (Camera)
	Audioset    : 0x00000001
	Tuner       : 0x00000000
	Standard    : 0x0000000000FFFFFF
	Status      : 0x00000000 (ok)
	Capabilities: 0x00000004 (SDTV standards)
```

The number and type of detected inputs will depend on the capture. Only one of them should have the name "Composite", and a number ID right above the name. In this example, the Composite input corresponds to the number ID 1. Use this ID to configure the video source with V4L:

`v4l2-ctl -d /dev/video<#> -i <ID>`

You may need to redo this configuration step **every time** the capture card is plugged to the computer or the computer reboots. Executing the command should return an info message indicating if the configuration command was performed successfully. If so, check the video source’s format:

`v4l2-ctl -d /dev/video<#> -V`

In particular, identify the frame’s width and height, as well as pixel format. The latter should be expressed as a four-character code (also known as a "fourcc"). After switching the power on, check if the video source is properly configured by starting Cheese (built-in Ubuntu app) or opening an MPV window:

`mpv av://v4l2:/dev/video<#> tv:///0`

In both cases, the video stream should be visible in the window. Running MPV also outputs some details about the video format in the terminal window (use Ctrl-C to close MPV).

## Reading Frames with OpenCV

The camera’s video stream can be read with OpenCV using the corresponding device name explicitly in the `VideoCapture` constructor:

```py
import cv2
cap = cv2.VideoCapture("/dev/video<#>", cv2.CAP_V4L)
```

The second argument specifies the API preference for video capturing, and may be optional. However, unlike Cheese or MPV, the `VideoCapture` instance may not detect the video format used by the capture card. In this case, the corresponding setting must be specified in the properties of the `VideoCapture` object. This step requires the "fourcc" previously identified,*e.g.* with YUYV:

```py
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
```

Note that the `VideoCapture` object can read frames even if this property is not set to an adequate value, so the code may very well run without raising any error or warning. However, the frames will most likely be subject to warping, color distortion or other forms of noise related to conversion issues. Reading and displaying a frame will unmistakably reveal if it is the case:

```py
ret, frame = cap.read()
cv2.imshow("Frame", frame)
```

If the settings are correct, a frame retrieved with OpenCV should be similar to the video stream displayed by Cheese or MPV.

## Troubleshooting

### Trouble finding the `/dev/video<#>` entry corresponding to the capture card

If no new `/dev/video<#>` entry appears when plugging the capture card to the computer, the card is not detected. Check if it needs to be powered, switched on or connected to a specific type of USB port. Check for malfunction of the USB connection on both sides (male and female ports). Check on a different computer.

### No video stream received

This is typically demonstrated by Cheese or MPV detecting a video source but showing only a black screen (except maybe for a thin green strip at the bottom of the frame). Make sure that the card’s video input was properly set by executing the command again:

`v4l2-ctl -d /dev/video<#> -i <ID>`

If the issue persists, its source is most likely upstream from the capture card. Check if the camera’s power supply is switched on and if the wiring effectively delivers the power to the camera.

### Segmentation fault when reading a frame

If `cap.read()` or `cv2.imshow("Frame", frame)` results in a segmentation fault, try instanciating the `VideoCapture` object with a device number instead of a device name:

```py
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
```
