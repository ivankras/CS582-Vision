import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from IPython.display import Image, display
import numpy as np
tf.enable_eager_execution()

from api.mlcode.losses import compute_loss_and_grads
class StyleTransfer:
    def __init__(self,
            base_image_name, base_image_url,
            style_image_name, style_image_url,
            result_prefix='generated',
            total_variation_weight=1e-6,
            style_weight=1e-6,
            content_weight=2.5e-8,
            img_nrows=400,
            style_layer_names=None,
            content_layer_name=None
            ):
        self._base_image_name = base_image_name
        self._base_image_url = base_image_url
        self._style_image_name = style_image_name
        self._style_image_url = style_image_url
        self._base_image_path = keras.utils.get_file(base_image_name, base_image_url)
        self._style_reference_image_path = keras.utils.get_file(self._style_image_name, self._style_image_url)
        self._result_prefix = result_prefix

        # Weights of the different loss components
        self._total_variation_weight = total_variation_weight
        self._style_weight = style_weight
        self._content_weight = content_weight

        # Dimensions of the generated picture.
        self._width, self._height = keras.preprocessing.image.load_img(self._base_image_path).size
        self._img_nrows = img_nrows
        self._img_ncols = int(self._width * self._img_nrows / self._height)

        # List of layers to use for the style loss.
        self._style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ] if style_layer_names is None else style_layer_names
        # The layer to use for the content loss.
        self._content_layer_name = "block5_conv2" if content_layer_name is None else content_layer_name

        self._hasResults = False

    def preprocess_image(self, image_path, img_nrows, img_ncols):
        # Util function to open, resize and format pictures into appropriate tensors
        img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
        )
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)


    def deprocess_image(self, x, img_nrows, img_ncols):
        # Util function to convert a tensor into a valid image
        x = x.reshape((img_nrows, img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    def setupModel(self, iterations=500):
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self._model = vgg19.VGG19(weights="imagenet", include_top=False)

        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        self._outputs_dict = dict([(layer.name, layer.output) for layer in self._model.layers])

        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        self._feature_extractor = keras.Model(inputs=self._model.inputs, outputs=self._outputs_dict)

        self._optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
            )
        )

        self._base_image = self.preprocess_image(self._base_image_path, self._img_nrows, self._img_ncols)
        self._style_reference_image = self.preprocess_image(self._style_reference_image_path, self._img_nrows, self._img_ncols)
        self._combination_image = tf.Variable(self.preprocess_image(self._base_image_path, self._img_nrows, self._img_ncols))

        self._iterations = iterations



    def train(self):
        for i in range(1, self._iterations + 1):
            loss, grads = compute_loss_and_grads(
                self._combination_image, self._base_image, self._style_reference_image,
                self._feature_extractor, self._style_layer_names, self._content_layer_name,
                self._content_weight, self._style_weight, self._total_variation_weight,
                self._img_nrows, self._img_ncols
            )
            self._optimizer.apply_gradients([(grads, self._combination_image)])
            if i % 100 == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                img = self.deprocess_image(self._combination_image.numpy(), self._img_nrows, self._img_ncols)
                fname = self._result_prefix + "_at_iteration_%d.png" % i
                keras.preprocessing.image.save_img('api/'+fname, img)
                self._hasResults = True


    def printResult(self, resultAt=4000):
        if not self._hasResults:
            print("Error: Network not trained")
            return
        
        if resultAt % 100 != 0:
            print("Error: Only iterations that are multiple of 100 are saved")
            return

        display(Image(self._result_prefix + f"_at_iteration_{resultAt}.png"))


