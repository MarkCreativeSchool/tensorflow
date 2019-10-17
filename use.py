# coding: utf-8
import cv2
import tensorflow as tf
import numpy as np


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result


if __name__ == '__main__':
    img = cv2.imread("images/fluit/apple/5.apple_logo.png")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    img = img / 255.

    with tf.compat.v1.Session() as sess:
        with tf.io.gfile.GFile("tmp/output_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            _ = tf.import_graph_def(graph_def)
            #tensor_input = sess.graph.get_tensor_by_name("import/Placeholder")
            #tensor_output = sess.graph.get_tensor_by_name("import/final_result")

            t = read_tensor_from_image_file(
                "images/fluit/apple/10.itunes-apple-logo-apple-music-giftcard-social-card.jpg",
                input_height=299,
                input_width=299,
                input_mean=0,
                input_std=255)

            graph = load_graph("tmp/output_graph.pb")
            input_operation = graph.get_operation_by_name("import/Placeholder")
            output_operation = graph.get_operation_by_name("import/final_result")
            with tf.compat.v1.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
                print(results)

            # results = sess.run(output_operation, {input_operation: t})
