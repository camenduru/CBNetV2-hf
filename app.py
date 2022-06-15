#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

if os.getenv('SYSTEM') == 'spaces':
    import mim

    mim.uninstall('mmcv-full', confirm_yes=True)
    mim.install('mmcv-full==1.5.0', is_yes=True)

    subprocess.run('pip uninstall -y opencv-python'.split())
    subprocess.run('pip uninstall -y opencv-python-headless'.split())
    subprocess.run('pip install opencv-python-headless==4.5.5.64'.split())

    subprocess.run('git apply ../patch'.split(), cwd='CBNetV2')
    subprocess.run('mv palette.py CBNetV2/mmdet/core/visualization/'.split())

import gradio as gr
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'CBNetV2/')

from mmdet.apis import inference_detector, init_detector


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


class Model:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.models = self._load_models()
        self.model_name = 'Improved HTC (DB-Swin-B)'

    def _load_models(self) -> dict[str, nn.Module]:
        model_dict = {
            'Faster R-CNN (DB-ResNet50)': {
                'config':
                'CBNetV2/configs/cbnet/faster_rcnn_cbv2d1_r50_fpn_1x_coco.py',
                'model':
                'https://github.com/CBNetwork/storage/releases/download/v1.0.0/faster_rcnn_cbv2d1_r50_fpn_1x_coco.pth.zip',
            },
            'Mask R-CNN (DB-Swin-T)': {
                'config':
                'CBNetV2/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                'model':
                'https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip',
            },
            #        'Cascade Mask R-CNN (DB-Swin-S)': {
            #            'config':
            #                'CBNetV2/configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py',
            #            'model':
            #                'https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip',
            #        },
            'Improved HTC (DB-Swin-B)': {
                'config':
                'CBNetV2/configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py',
                'model':
                'https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip',
            },
            'Improved HTC (DB-Swin-L)': {
                'config':
                'CBNetV2/configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py',
                'model':
                'https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip',
            },
            'Improved HTC (DB-Swin-L (TTA))': {
                'config':
                'CBNetV2/configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py',
                'model':
                'https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip',
            },
        }

        weight_dir = pathlib.Path('weights')
        weight_dir.mkdir(exist_ok=True)

        def _download(model_name: str, out_dir: pathlib.Path) -> None:
            import zipfile

            model_url = model_dict[model_name]['model']
            zip_name = model_url.split('/')[-1]

            out_path = out_dir / zip_name
            if out_path.exists():
                return
            torch.hub.download_url_to_file(model_url, out_path)

            with zipfile.ZipFile(out_path) as f:
                f.extractall(out_dir)

        def _get_model_path(model_name: str) -> str:
            model_url = model_dict[model_name]['model']
            model_name = model_url.split('/')[-1][:-4]
            return (weight_dir / model_name).as_posix()

        for model_name in model_dict:
            _download(model_name, weight_dir)

        models = {
            key: init_detector(dic['config'],
                               _get_model_path(key),
                               device=self.device)
            for key, dic in model_dict.items()
        }
        return models

    def set_model_name(self, name: str) -> None:
        self.model_name = name

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        model = self.models[self.model_name]
        out = inference_detector(model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        model = self.models[self.model_name]
        vis = model.show_result(image,
                                detection_results,
                                score_thr=score_threshold,
                                bbox_color=None,
                                text_color=(200, 200, 200),
                                mask_color=None)
        return vis[:, :, ::-1]  # BGR -> RGB


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def main():
    args = parse_args()
    model = Model(args.device)

    css = '''
h1#title {
  text-align: center;
}
'''

    with gr.Blocks(theme=args.theme, css=css) as demo:
        gr.Markdown('''<h1 id="title">VDIGPKU/CBNetV2</h1>

This is an unofficial demo for [https://github.com/VDIGPKU/CBNetV2](https://github.com/VDIGPKU/CBNetV2).'''
                    )

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

        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.cbnetv2" alt="visitor badge"/></center>'
        )

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
