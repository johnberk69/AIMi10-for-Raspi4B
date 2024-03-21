# USE CASE
# the goal was to develop a script that could be executed at the command line, accept an input [image] file, and pass
# it to a local Azure Custom Vision model [Object Detector] that had been exported as an ONNX model.  That local model had
# (29) categories of Game Animals, or related objects that might be captured by Trail Cams.  

# IMPORTS
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image

PROB_THRESHOLD = 0.01  # Minimum probably to show results.  ).05 or 0.10 does a better job of filtering out low probability 'noisy' images


class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

# https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
def switch(class_id):
    if class_id == 0:
        return "BearDay"
    elif class_id == 1:
        return "BearNight"
    elif class_id == 2:
        return "CougarDay"
    elif class_id == 3:
        return "CougarNight"
    elif class_id == 4:
        return "CoyoteDay"
    elif class_id == 5:
        return "CoyoteNight"
    elif class_id == 6:
        return "CraneDay"
    elif class_id == 7:
        return "DeerBuckDay"
    elif class_id == 8:
        return "DeerBuckNight"
    elif class_id == 9:
        return "DeerDoeDay"
    elif class_id == 10:
        return "DeerDoeNight"
    elif class_id == 11:
        return "ElkBullDay"
    elif class_id == 12:
        return "ElkBullNight"
    elif class_id == 13:
        return "ElkCowDay"
    elif class_id == 14:
        return "ElkCowNight"
    elif class_id == 15:
        return "HumanDay"
    elif class_id == 16:
        return "HumanNight"
    elif class_id == 17:
        return "MooseBullDay"
    elif class_id == 18:
        return "MooseBullNight"
    elif class_id == 19:
        return "MooseCowDay"
    elif class_id == 20:
        return "MooseCowNight"
    elif class_id == 21:
        return "SwineDay"
    elif class_id == 22:
        return "SwineNight"
    elif class_id == 23:
        return "TurkeyDay"
    elif class_id == 24:
        return "TurkeyNight"
    elif class_id == 25:
        return "VehicleDay"
    elif class_id == 26:
        return "VehicleNight"
    elif class_id == 27:
        return "WolfDay"
    elif class_id == 28:
        return "WolfNight"

def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            clabel = switch(class_id)
#           print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            print(f"Label: {clabel}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.model_filepath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
    
