# -*- coding: utf-8 -*-
#
# @File:   app.py
# @Author: Haozhe Xie
# @Date:   2024-03-02 16:30:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-02 22:26:57
# @Email:  root@haozhexie.com

import streamlit as st

from PIL import Image

LOGGER = st.logger.get_logger(__name__)


def setup_runtime_env():
    pass


def get_models():
    pass


def get_generated_city(radius, altitude, azimuth):
    pass


def main(fgm, bgm):
    st.set_page_config(
        page_title="CityDreamer Demo",
        page_icon="🏙️",
    )
    # Main
    st.write("# CityDreamer Minimal Demo 🏙️")
    with open("README.md", "r") as f:
        markdown = f.read()
        st.markdown(markdown[markdown.rfind("---") :])
    imgbox = st.empty()

    # Sidebar
    st.sidebar.header("CityDreamer Settings")
    radius = st.sidebar.slider("Camera Radius (m)", 128, 512, 320, 5)
    altitude = st.sidebar.slider("Camera Altitude (m)", 256, 512, 384, 5)
    azimuth = st.sidebar.slider("Camera Azimuth (°)", 0, 360, 180, 5)
    if st.sidebar.button("Generate", type="primary"):
        img = get_generated_city(radius, altitude, azimuth)
        imgbox.image(img, caption="CityDreamer Generation")


if __name__ == "__main__":
    LOGGER.info("Setting up runtime environment...")
    setup_runtime_env()
    fgm, bgm = get_models()

    LOGGER.info("Starting the main application...")
    main(fgm, bgm)
