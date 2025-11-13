
import tensorflow as tf
import numpy as np
import os
import cv2
import click


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('representative_images_dir', nargs=1, type=click.Path())
@click.argument('tflite_int8_model', nargs=1, type=click.Path())
@click.argument('saved_model_dir', nargs=1, type=click.Path())
@click.option('--img-size', 'img_size', type=int, default=512)
def main(**options):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(options['tflite_int8_model']), exist_ok=True)

    # Representative dataset generator (limits to 100 images)
    def representative_dataset():
        image_files = sorted([
            os.path.join(options['representative_images_dir'], f)
            for f in os.listdir(options['representative_images_dir'])
            if f.endswith(".jpg")
        ])[:100]
        for img_path in image_files:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (options['img_size'], options['img_size']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            yield [img]

    # Convert the SavedModel to a TFLite INT8 model with float32 I/O
    print(f"Loading SavedModel from: {options['saved_model_dir']}")
    converter = tf.lite.TFLiteConverter.from_saved_model(options['saved_model_dir'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print("Converting to INT8 with float32 I/O...")
    tflite_quant_model = converter.convert()

    # Save the quantized TFLite model
    with open(options['tflite_int8_model'], "wb") as f:
        f.write(tflite_quant_model)

    print(f"Quantized model saved to: {options['tflite_int8_model']}")


if __name__ == '__main__':
    main()
