# -*- coding: utf-8 -*-
#
# @File:   app.py
# @Author: Haozhe Xie
# @Date:   2024-03-02 16:30:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-03 10:25:43
# @Email:  root@haozhexie.com

import logging
import os
import torch
import gradio as gr
import subprocess
import urllib.request
import ssl
import sys

# Fix: ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), "citydreamer"))
# Import CityDreamer modules
# import citydreamer.model
# import citydreamer.inference


def setup_runtime_env():
    subprocess.call(["pip", "freeze"])
    ext_dir = os.path.join(os.path.dirname(__file__), "citydreamer", "extensions")
    for e in os.listdir(ext_dir):
        if not os.path.isdir(e):
            continue
        subprocess.call(["pip", "install", "."], workdir=os.path.join(ext_dir, e))


def get_models():
    if not os.path.exists("CityDreamer-Fgnd.pth"):
        urllib.request.urlretrieve(
            "https://huggingface.co/hzxie/city-dreamer/resolve/main/CityDreamer-Fgnd.pth",
            "CityDreamer-Fgnd.pth",
        )
    if not os.path.exists("CityDreamer-Bgnd.pth"):
        urllib.request.urlretrieve(
            "https://huggingface.co/hzxie/city-dreamer/resolve/main/CityDreamer-Bgnd.pth",
            "CityDreamer-Bgnd.pth",
        )

    bgm_ckpt = torch.load("CityDreamer-Bgnd.pth")
    fgm_ckpt = torch.load("CityDreamer-Fgnd.pth")
    bgm = citydreamer.model.GanCraftGenerator(bgm_ckpt["cfg"])
    fgm = citydreamer.model.GanCraftGenerator(fgm_ckpt["cfg"])
    if torch.cuda.is_available():
        fgm = torch.nn.DataParallel(fgm).cuda().eval()
        bgm = torch.nn.DataParallel(bgm).cuda().eval()

    return bgm, fgm


def get_generated_city(radius, altitude, azimuth):
    print(radius, altitude, azimuth)


def main(debug):
    title = "CityDreamer Demo 🏙️"
    with open("README.md", "r") as f:
        markdown = f.read()
        desc = markdown[markdown.rfind("---") + 3:]
    with open("ARTICLE.md", "r") as f:
        arti = f.read()
    with open("assets/style.css") as f:
        css = f.read()

    app = gr.Interface(
        get_generated_city,
        [
            gr.Slider(
                128, 512, value=320, step=5, label="Camera Radius (m)"
            ),
            gr.Slider(
                256, 512, value=384, step=5, label="Camera Altitude (m)"
            ),
            gr.Slider(0, 360, value=180, step=5, label="Camera Azimuth (°)"),
        ],
        [gr.Image(type="numpy", label="Generated City")],
        title=title,
        description=desc,
        article=arti,
        allow_flagging="never",
        css=css,
    )
    app.queue(api_open=False)
    app.launch(debug=debug)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO
    )
    logging.info("Compile CUDA extensions...")
    # setup_runtime_env()
    logging.info("Downloading pretrained models...")
    # fgm, bgm = get_models()
    logging.info("Starting the main application...")
    main(os.getenv("DEBUG") == "1")
