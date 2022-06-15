#!/usr/bin/env python

from __future__ import annotations

import argparse
import pathlib

import gradio as gr

from model import Model

DESCRIPTION = '''# CBNetV2

This is an unofficial demo for [https://github.com/VDIGPKU/CBNetV2](https://github.com/VDIGPKU/CBNetV2).'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.cbnetv2" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def main():
    args = parse_args()
    model = Model(args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='numpy')
                with gr.Row():
                    detector_name = gr.Dropdown(list(model.models.keys()),
                                                value=model.model_name,
                                                label='Detector')
                with gr.Row():
                    detect_button = gr.Button(value='Detect')
                    detection_results = gr.Variable()
            with gr.Column():
                with gr.Row():
                    detection_visualization = gr.Image(
                        label='Detection Result', type='numpy')
                with gr.Row():
                    visualization_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.3,
                        label='Visualization Score Threshold')
                with gr.Row():
                    redraw_button = gr.Button(value='Redraw')

        with gr.Row():
            paths = sorted(pathlib.Path('images').rglob('*.jpg'))
            example_images = gr.Dataset(components=[input_image],
                                        samples=[[path.as_posix()]
                                                 for path in paths])

        gr.Markdown(FOOTER)

        detector_name.change(fn=model.set_model_name,
                             inputs=[detector_name],
                             outputs=None)
        detect_button.click(fn=model.detect_and_visualize,
                            inputs=[
                                input_image,
                                visualization_score_threshold,
                            ],
                            outputs=[
                                detection_results,
                                detection_visualization,
                            ])
        redraw_button.click(fn=model.visualize_detection_results,
                            inputs=[
                                input_image,
                                detection_results,
                                visualization_score_threshold,
                            ],
                            outputs=[detection_visualization])
        example_images.click(fn=set_example_image,
                             inputs=[example_images],
                             outputs=[input_image])

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
