#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr

from model import Model

DESCRIPTION = '# [CBNetV2](https://github.com/VDIGPKU/CBNetV2)'

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(label='Input Image', type='numpy')
            with gr.Row():
                detector_name = gr.Dropdown(label='Detector',
                                            choices=list(model.models.keys()),
                                            value=model.model_name)
            with gr.Row():
                detect_button = gr.Button('Detect')
                detection_results = gr.Variable()
        with gr.Column():
            with gr.Row():
                detection_visualization = gr.Image(label='Detection Result',
                                                   type='numpy')
            with gr.Row():
                visualization_score_threshold = gr.Slider(
                    label='Visualization Score Threshold',
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.3)
            with gr.Row():
                redraw_button = gr.Button('Redraw')

    with gr.Row():
        paths = sorted(pathlib.Path('images').rglob('*.jpg'))
        gr.Examples(examples=[[path.as_posix()] for path in paths],
                    inputs=input_image)

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
demo.queue(max_size=10).launch()
