# -*- coding: utf-8 -*-
#
# @File:   app.py
# @Author: Haozhe Xie
# @Date:   2024-03-02 16:30:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-03 12:02:23
# @Email:  root@haozhexie.com

import gradio as gr
import logging
import numpy as np
import os
import ssl
import subprocess
import sys
import torch
import urllib.request

from PIL import Image

# Fix: ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
ssl._create_default_https_context = ssl._create_unverified_context
# Import CityDreamer modules
sys.path.append(os.path.join(os.path.dirname(__file__), "citydreamer"))


def setup_runtime_env():
    logging.info("CUDA version is %s" % subprocess.check_output(["nvcc", "--version"]))
    logging.info("GCC version is %s" % subprocess.check_output(["g++", "--version"]))
    # Compile CUDA extensions
    ext_dir = os.path.join(os.path.dirname(__file__), "citydreamer", "extensions")
    for e in os.listdir(ext_dir):
        if not os.path.isdir(os.path.join(ext_dir, e)):
            continue

        subprocess.call(["pip", "install", "."], cwd=os.path.join(ext_dir, e))


def get_models(file_name):
    import citydreamer.model

    if not os.path.exists(file_name):
        urllib.request.urlretrieve(
            "https://huggingface.co/hzxie/city-dreamer/resolve/main/%s" % file_name,
            file_name,
        )

    ckpt = torch.load(file_name)
    model = citydreamer.model.GanCraftGenerator(ckpt["cfg"])
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda().eval()

    return model


def get_city_layout():
    hf = np.array(Image.open("assets/NYC-HghtFld.png"))
    seg = np.array(Image.open("assets/NYC-SegMap.png").convert("P"))
    return hf, seg


def get_generated_city(radius, altitude, azimuth):
    # The import must be done after CUDA extension compilation
    import citydreamer.inference

    return citydreamer.inference.generate_city(
        get_generated_city.fgm,
        get_generated_city.bgm,
        get_generated_city.hf,
        get_generated_city.seg,
        radius,
        altitude,
        azimuth,
    )


def main(debug):
    title = "CityDreamer Demo 🏙️"
    with open("README.md", "r") as f:
        markdown = f.read()
        desc = markdown[markdown.rfind("---") + 3 :]
    with open("ARTICLE.md", "r") as f:
        arti = f.read()

    app = gr.Interface(
        get_generated_city,
        [
            gr.Slider(128, 512, value=320, step=5, label="Camera Radius (m)"),
            gr.Slider(256, 512, value=384, step=5, label="Camera Altitude (m)"),
            gr.Slider(0, 360, value=180, step=5, label="Camera Azimuth (°)"),
        ],
        [gr.Image(type="numpy", label="Generated City")],
        title=title,
        description=desc,
        article=arti,
        allow_flagging="never",
    )
    app.queue(api_open=False)
    app.launch(debug=debug)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO
    )
    logging.info("Compiling CUDA extensions...")
    # setup_runtime_env()

    logging.info("Downloading pretrained models...")
    fgm = get_models("CityDreamer-Fgnd.pth")
    bgm = get_models("CityDreamer-Bgnd.pth")
    get_generated_city.fgm = fgm
    get_generated_city.bgm = bgm

    logging.info("Loading New York city layout to RAM...")
    hf, seg = get_city_layout()
    get_generated_city.hf = hf
    get_generated_city.seg = seg

    logging.info("Starting the main application...")
    main(os.getenv("DEBUG") == "1")
