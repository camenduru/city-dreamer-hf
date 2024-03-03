# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-03-26 19:23:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-15 10:47:53
# @Email:  root@haozhexie.com

# Mayavi off screen rendering
# Ref: https://github.com/enthought/mayavi/issues/477#issuecomment-477653210
from xvfbwrapper import Xvfb

vdisplay = Xvfb(width=1920, height=1080)
vdisplay.start()

import logging
import mayavi.mlab
import numpy as np
import os
import sys
import torch
import unittest

from PIL import Image
from torch.autograd import gradcheck

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from extensions.extrude_tensor import ExtrudeTensorFunction

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None


class ExtrudeTensorTestCase(unittest.TestCase):
    @unittest.skip("The CUDA extension is compiled with int types by default.")
    def test_extrude_tensor_grad(self):
        # To run this test, make sure that the int types are replaced by double types in CUDA
        SIZE = 16
        seg_map = (
            torch.randint(low=1, high=7, size=(SIZE, SIZE))
            .double()
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        height_field = (
            torch.randint(low=0, high=255, size=(SIZE, SIZE))
            .double()
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        logging.debug("SegMap Size: %s" % (seg_map.size(),))
        logging.debug("HeightField Size: %s" % (height_field.size(),))
        seg_map.requires_grad = True
        height_field.requires_grad = True
        logging.info(
            "Gradient Check: %s" % "OK"
            if gradcheck(
                ExtrudeTensorFunction.apply, [seg_map.cuda(), height_field.cuda(), 256]
            )
            else "Failed"
        )

    def test_extrude_tensor_gen(self):
        MAX_HEIGHT = 256
        proj_home_dir = os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir
        )
        osm_data_dir = os.path.join(proj_home_dir, "data", "osm")
        osm_name = "US-NewYork"
        seg_map = Image.open(os.path.join(osm_data_dir, osm_name, "seg.png")).convert(
            "P"
        )
        height_field = Image.open(os.path.join(osm_data_dir, osm_name, "hf.png"))
        # Crop the maps
        seg_map = np.array(seg_map)[3840:4096, 3840:4096]
        height_field = np.array(height_field)[3840:4096, 3840:4096]
        # Convert to tensors
        seg_map_tnsr = (
            torch.from_numpy(seg_map).unsqueeze(dim=0).unsqueeze(dim=0).int().cuda()
        )
        height_field_tnsr = (
            torch.from_numpy(height_field)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .int()
            .cuda()
        )
        volume = ExtrudeTensorFunction.apply(
            seg_map_tnsr, height_field_tnsr, MAX_HEIGHT
        )
        # 3D Visualization
        vol = volume.squeeze().cpu().numpy().astype(np.uint8)

        x, y, z = np.where(vol != 0)
        n_pts = len(x)
        colors = np.zeros((n_pts, 4), dtype=np.uint8)
        # fmt: off
        colors[vol[x, y, z] == 1] = [96, 0, 0, 255]         # highway      -> red
        colors[vol[x, y, z] == 2] = [96, 96, 0, 255]        # building     -> yellow
        colors[vol[x, y, z] == 3] = [0, 96, 0, 255]         # green lands  -> green
        colors[vol[x, y, z] == 4] = [0, 96, 96, 255]        # construction -> cyan
        colors[vol[x, y, z] == 5] = [0, 0, 96, 255]         # water        -> blue
        colors[vol[x, y, z] == 6] = [128, 128, 128, 255]    # ground       -> gray
        # fmt: on
        mayavi.mlab.options.offscreen = True
        mayavi.mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))
        pts = mayavi.mlab.points3d(x, y, z, mode="cube", scale_factor=1)
        pts.glyph.scale_mode = "scale_by_vector"
        pts.mlab_source.dataset.point_data.scalars = colors
        mayavi.mlab.savefig(os.path.join(proj_home_dir, "logs", "%s-3d.jpg" % osm_name))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    unittest.main()
