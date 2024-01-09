# AIMi10-for-Raspi4B
Complete Animal Identification Model Iteration 10 (AIMi10) for use on Raspberry Pi4B hardware, stand alone

01/08/2024 John-M Berk     Berkheimer LLC   john@berkcg.com
The *Animal Identification Model Iteration 10 (AIMi10)* is proprietary software and cannot used without the express written permission of Bushnell Inc.  

This version of this AI Vision model and code is in a form that can be deployed stand-alone on a Raspberry Pi4B single board computer.  It can also
be run on any Linux or Windows PC that supports opencv for an attached USB webcam, and that can build a Python 3.11 virtual environment with the
required support packages.  Once the project is set up within a VSCode IDE, the local .onnx version of the model can be tested and demonstrated
using the *AIMi10predict.py* script (for one-off tests) or the *AIMi10predict10secIntervalFINAL.py* continuously looping script.  
Call *AIMi10predict.py* using this form to pull existing images from the local TCAI Test Images folder and inference them: 
(venv-for-onnx-opencv) user@tcairaspi4b:~/Development/Github/AIMi10-for-Raspi4B $ **python3 AIMi10predict.py TCAI_AIM_Iteration_10_on_Gen_Compact_S1.ONNX/model.onnx TCAI_AIM_Iteration_10_Test_Images/03CougarNight.jpg**<Enter> 
This should return text to the terminal that looks like this:
Label: CougarNight, Probability: 0.97037, box: (0.38457, 0.30277) (1.01042, 0.94044)
Label: CougarNight, Probability: 0.02317, box: (0.39474, 0.06022) (1.01481, 0.55623)
Label: CougarNight, Probability: 0.01076, box: (0.61790, -0.00685) (0.84761, 0.35594)
Label: CougarNight, Probability: 0.01047, box: (0.18100, 0.53567) (0.99813, 0.97460)

Call the looping *AIMi10predict10secIntervalFINAL.py* using this form to continuously pull live images from the attached USB camera and inference them:
(venv-for-onnx-opencv) mike@tcairaspi4b:~/Development/Github/AIMi10-for-Raspi4B $ **python AIMi10predict10secIntervalFINAL.py TCAI_AIM_Iteration_10_on_Gen_Compact_S1.ONNX/model.onnx aimtestimage.jpg**<Enter>
This will also diplay to the terminal, but will produce a image displayed on-screen with bounding boxes surrounding the detected objects.



