# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2024-03-02 16:30:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-03 15:59:00
# @Email:  root@haozhexie.com

import copy
import cv2
import logging
import math
import numpy as np
import torch
import torchvision

import citydreamer.extensions.extrude_tensor
import citydreamer.extensions.voxlib

# Global constants
HEIGHTS = {
    "ROAD": 4,
    "GREEN_LANDS": 8,
    "CONSTRUCTION": 10,
    "COAST_ZONES": 0,
    "ROOF": 1,
}
CLASSES = {
    "NULL": 0,
    "ROAD": 1,
    "BLD_FACADE": 2,
    "GREEN_LANDS": 3,
    "CONSTRUCTION": 4,
    "COAST_ZONES": 5,
    "OTHERS": 6,
    "BLD_ROOF": 7,
}
# NOTE: ID > 10 are reserved for building instances.
# Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
CONSTANTS = {
    "BLD_INS_LABEL_MIN": 10,
    "LAYOUT_N_CLASSES": 7,
    "LAYOUT_VOL_SIZE": 1536,
    "BUILDING_VOL_SIZE": 672,
    "EXTENDED_VOL_SIZE": 2880,
    "LAYOUT_MAX_HEIGHT": 640,
    "GES_VFOV": 20,
    "GES_IMAGE_HEIGHT": 540,
    "GES_IMAGE_WIDTH": 960,
    "IMAGE_PADDING": 8,
    "N_VOXEL_INTERSECT_SAMPLES": 6,
}


def generate_city(fgm, bgm, hf, seg, cx, cy, radius, altitude, azimuth):
    cam_pos = get_orbit_camera_position(radius, altitude, azimuth)
    seg, building_stats = get_instance_seg_map(seg)
    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z, building_zs = get_latent_codes(
        building_stats,
        bgm.module.cfg.NETWORK.GANCRAFT.STYLE_DIM,
        bgm.output_device,
    )
    # Generate local image patch of the height field and seg map
    part_hf, part_seg = get_part_hf_seg(hf, seg, cx, cy, CONSTANTS["EXTENDED_VOL_SIZE"])
    # Generate local image patch of the height field and seg map
    part_hf, part_seg = get_part_hf_seg(hf, seg, cx, cy, CONSTANTS["EXTENDED_VOL_SIZE"])
    # print(part_hf.shape)    # (2880, 2880)
    # print(part_seg.shape)   # (2880, 2880)
    # Recalculate the building positions based on the current patch
    _building_stats = get_part_building_stats(part_seg, building_stats, cx, cy)
    # Generate the concatenated height field and seg. map tensor
    hf_seg = get_hf_seg_tensor(part_hf, part_seg, bgm.output_device)
    # print(hf_seg.size())    # torch.Size([1, 8, 2880, 2880])
    # Build seg_volume
    logging.info("Generating seg volume ...")
    seg_volume = get_seg_volume(part_hf, part_seg)
    logging.info("Rendering City Image ...")
    img = render(
        (CONSTANTS["GES_IMAGE_HEIGHT"] // 5, CONSTANTS["GES_IMAGE_WIDTH"] // 5),
        seg_volume,
        hf_seg,
        cam_pos,
        bgm,
        fgm,
        _building_stats,
        bg_z,
        building_zs,
    )
    img = ((img.cpu().numpy().squeeze().transpose((1, 2, 0)) / 2 + 0.5) * 255).astype(
        np.uint8
    )
    return img


def get_orbit_camera_position(radius, altitude, azimuth):
    cx = CONSTANTS["LAYOUT_VOL_SIZE"] // 2
    cy = cx
    theta = np.deg2rad(azimuth)
    cam_x = cx + radius * math.cos(theta)
    cam_y = cy + radius * math.sin(theta)
    return {"x": cam_x, "y": cam_y, "z": altitude}


def get_instance_seg_map(seg_map):
    # Mapping constructions to buildings
    seg_map[seg_map == CLASSES["CONSTRUCTION"]] = CLASSES["BLD_FACADE"]
    # Use connected components to get building instances
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (seg_map == CLASSES["BLD_FACADE"]).astype(np.uint8), connectivity=4
    )
    # Remove non-building instance masks
    labels[seg_map != CLASSES["BLD_FACADE"]] = 0
    # Building instance mask
    building_mask = labels != 0
    # Make building instance IDs are even numbers and start from 10
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
    labels = (labels + CONSTANTS["BLD_INS_LABEL_MIN"]) * 2

    seg_map[seg_map == CLASSES["BLD_FACADE"]] = 0
    seg_map = seg_map * (1 - building_mask) + labels * building_mask
    assert np.max(labels) < 2147483648
    return seg_map.astype(np.int32), stats[:, :4]


def get_latent_codes(building_stats, bg_style_dim, output_device):
    bg_z = _get_z(output_device, bg_style_dim)
    building_zs = {
        (i + CONSTANTS["BLD_INS_LABEL_MIN"]) * 2: _get_z(output_device)
        for i in range(len(building_stats))
    }
    return bg_z, building_zs


def _get_z(device, z_dim=256):
    if z_dim is None:
        return None

    return torch.randn(1, z_dim, dtype=torch.float32, device=device)


def get_part_hf_seg(hf, seg, cx, cy, patch_size):
    part_hf = _get_image_patch(hf, cx, cy, patch_size)
    part_seg = _get_image_patch(seg, cx, cy, patch_size)
    assert part_hf.shape == (
        patch_size,
        patch_size,
    ), part_hf.shape
    assert part_hf.shape == part_seg.shape, part_seg.shape
    return part_hf, part_seg


def _get_image_patch(image, cx, cy, patch_size):
    sx = cx - patch_size // 2
    sy = cy - patch_size // 2
    ex = sx + patch_size
    ey = sy + patch_size
    return image[sy:ey, sx:ex]


def get_part_building_stats(part_seg, building_stats, cx, cy):
    _buildings = np.unique(part_seg[part_seg > CONSTANTS["BLD_INS_LABEL_MIN"]])
    _building_stats = {}
    for b in _buildings:
        _b = b // 2 - CONSTANTS["BLD_INS_LABEL_MIN"]
        _building_stats[b] = [
            building_stats[_b, 1] - cy + building_stats[_b, 3] / 2,
            building_stats[_b, 0] - cx + building_stats[_b, 2] / 2,
        ]
    return _building_stats


def get_hf_seg_tensor(part_hf, part_seg, output_device):
    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(output_device)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(output_device)
    part_hf = part_hf / CONSTANTS["LAYOUT_MAX_HEIGHT"]
    part_seg = _masks_to_onehots(part_seg[:, 0, :, :], CONSTANTS["LAYOUT_N_CLASSES"])
    return torch.cat([part_hf, part_seg], dim=1)


def _masks_to_onehots(masks, n_class, ignored_classes=[]):
    b, h, w = masks.shape
    n_class_actual = n_class - len(ignored_classes)
    one_hot_masks = torch.zeros(
        (b, n_class_actual, h, w), dtype=torch.float32, device=masks.device
    )

    n_class_cnt = 0
    for i in range(n_class):
        if i not in ignored_classes:
            one_hot_masks[:, n_class_cnt] = masks == i
            n_class_cnt += 1
    return one_hot_masks


def get_seg_volume(part_hf, part_seg):
    tensor_extruder = citydreamer.extensions.extrude_tensor.TensorExtruder(
        CONSTANTS["LAYOUT_MAX_HEIGHT"]
    )

    if part_hf.shape == (
        CONSTANTS["EXTENDED_VOL_SIZE"],
        CONSTANTS["EXTENDED_VOL_SIZE"],
    ):
        part_hf = part_hf[
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
        ]
        # print(part_hf.shape)  # torch.Size([1, 8, 1536, 1536])
        part_seg = part_seg[
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
        ]
        # print(part_seg.shape)  # torch.Size([1, 8, 1536, 1536])

    assert part_hf.shape == (
        CONSTANTS["LAYOUT_VOL_SIZE"],
        CONSTANTS["LAYOUT_VOL_SIZE"],
    )
    assert part_hf.shape == part_seg.shape, part_seg.shape

    seg_volume = tensor_extruder(
        torch.from_numpy(part_seg[None, None, ...]).cuda(),
        torch.from_numpy(part_hf[None, None, ...]).cuda(),
    ).squeeze()
    logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))
    # Change the top-level voxel of the "Building Facade" to "Building Roof"
    roof_seg_map = part_seg.copy()
    non_roof_msk = part_seg <= CONSTANTS["BLD_INS_LABEL_MIN"]
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
    roof_seg_map = roof_seg_map - 1
    roof_seg_map[non_roof_msk] = 0
    for rh in range(1, HEIGHTS["ROOF"] + 1):
        seg_volume = seg_volume.scatter_(
            dim=2,
            index=torch.from_numpy(part_hf[..., None] + rh).long().cuda(),
            src=torch.from_numpy(roof_seg_map[..., None]).cuda(),
        )
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    return seg_volume


def get_voxel_intersection_perspective(seg_volume, camera_location):
    CAMERA_FOCAL = (
        CONSTANTS["GES_IMAGE_HEIGHT"] / 2 / np.tan(np.deg2rad(CONSTANTS["GES_VFOV"]))
    )
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    camera_target = {
        "x": seg_volume.size(1) // 2 - 1,
        "y": seg_volume.size(0) // 2 - 1,
    }
    cam_origin = torch.tensor(
        [
            camera_location["y"],
            camera_location["x"],
            camera_location["z"],
        ],
        dtype=torch.float32,
        device=seg_volume.device,
    )

    (
        voxel_id,
        depth2,
        raydirs,
    ) = citydreamer.extensions.voxlib.ray_voxel_intersection_perspective(
        seg_volume,
        cam_origin,
        torch.tensor(
            [
                camera_target["y"] - camera_location["y"],
                camera_target["x"] - camera_location["x"],
                -camera_location["z"],
            ],
            dtype=torch.float32,
            device=seg_volume.device,
        ),
        torch.tensor([0, 0, 1], dtype=torch.float32),
        CAMERA_FOCAL * 2.06,
        [
            (CONSTANTS["GES_IMAGE_HEIGHT"] - 1) / 2.0,
            (CONSTANTS["GES_IMAGE_WIDTH"] - 1) / 2.0,
        ],
        [CONSTANTS["GES_IMAGE_HEIGHT"], CONSTANTS["GES_IMAGE_WIDTH"]],
        CONSTANTS["N_VOXEL_INTERSECT_SAMPLES"],
    )
    return (
        voxel_id.unsqueeze(dim=0),
        depth2.permute(1, 2, 0, 3, 4).unsqueeze(dim=0),
        raydirs.unsqueeze(dim=0),
        cam_origin.unsqueeze(dim=0),
    )


def _get_pad_img_bbox(sx, ex, sy, ey):
    psx = sx - CONSTANTS["IMAGE_PADDING"] if sx != 0 else 0
    psy = sy - CONSTANTS["IMAGE_PADDING"] if sy != 0 else 0
    pex = (
        ex + CONSTANTS["IMAGE_PADDING"]
        if ex != CONSTANTS["GES_IMAGE_WIDTH"]
        else CONSTANTS["GES_IMAGE_WIDTH"]
    )
    pey = (
        ey + CONSTANTS["IMAGE_PADDING"]
        if ey != CONSTANTS["GES_IMAGE_HEIGHT"]
        else CONSTANTS["GES_IMAGE_HEIGHT"]
    )
    return psx, pex, psy, pey


def _get_img_without_pad(img, sx, ex, sy, ey, psx, pex, psy, pey):
    if CONSTANTS["IMAGE_PADDING"] == 0:
        return img

    return img[
        :,
        :,
        sy - psy : ey - pey if ey != pey else ey,
        sx - psx : ex - pex if ex != pex else ex,
    ]


def render_bg(
    patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_origin, z
):
    assert hf_seg.size(2) == CONSTANTS["EXTENDED_VOL_SIZE"]
    assert hf_seg.size(3) == CONSTANTS["EXTENDED_VOL_SIZE"]
    hf_seg = hf_seg[
        :,
        :,
        CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
        CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
    ]
    assert hf_seg.size(2) == CONSTANTS["LAYOUT_VOL_SIZE"]
    assert hf_seg.size(3) == CONSTANTS["LAYOUT_VOL_SIZE"]

    blurrer = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2, 2))
    _voxel_id = copy.deepcopy(voxel_id)
    _voxel_id[voxel_id >= CONSTANTS["BLD_INS_LABEL_MIN"]] = CLASSES["BLD_FACADE"]
    assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    bg_img = torch.zeros(
        1,
        3,
        CONSTANTS["GES_IMAGE_HEIGHT"],
        CONSTANTS["GES_IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_bg.output_device,
    )
    # Render background patches by patch to avoid OOM
    for i in range(CONSTANTS["GES_IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["GES_IMAGE_WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = _get_pad_img_bbox(sx, ex, sy, ey)
            output_bg = gancraft_bg(
                hf_seg=hf_seg,
                voxel_id=_voxel_id[:, psy:pey, psx:pex],
                depth2=depth2[:, psy:pey, psx:pex],
                raydirs=raydirs[:, psy:pey, psx:pex],
                cam_origin=cam_origin,
                building_stats=None,
                z=z,
                deterministic=True,
            )
            # Make road blurry
            road_mask = (
                (_voxel_id[:, None, psy:pey, psx:pex, 0, 0] == CLASSES["ROAD"])
                .repeat(1, 3, 1, 1)
                .float()
            )
            output_bg = blurrer(output_bg) * road_mask + output_bg * (1 - road_mask)
            bg_img[:, :, sy:ey, sx:ex] = _get_img_without_pad(
                output_bg, sx, ex, sy, ey, psx, pex, psy, pey
            )

    return bg_img


def render_fg(
    patch_size,
    gancraft_fg,
    building_id,
    hf_seg,
    voxel_id,
    depth2,
    raydirs,
    cam_origin,
    building_stats,
    building_z,
):
    _voxel_id = copy.deepcopy(voxel_id)
    _curr_bld = torch.tensor([building_id, building_id - 1], device=voxel_id.device)
    _voxel_id[~torch.isin(_voxel_id, _curr_bld)] = 0
    _voxel_id[voxel_id == building_id] = CLASSES["BLD_FACADE"]
    _voxel_id[voxel_id == building_id - 1] = CLASSES["BLD_ROOF"]

    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    _hf_seg = copy.deepcopy(hf_seg)
    _hf_seg[hf_seg != building_id] = 0
    _hf_seg[hf_seg == building_id] = CLASSES["BLD_FACADE"]
    _raydirs = copy.deepcopy(raydirs)
    _raydirs[_voxel_id[..., 0, 0] == 0] = 0

    # Crop the "hf_seg" image using the center of the target building as the reference
    cx = CONSTANTS["EXTENDED_VOL_SIZE"] // 2 - int(building_stats[1])
    cy = CONSTANTS["EXTENDED_VOL_SIZE"] // 2 - int(building_stats[0])
    sx = cx - CONSTANTS["BUILDING_VOL_SIZE"] // 2
    ex = cx + CONSTANTS["BUILDING_VOL_SIZE"] // 2
    sy = cy - CONSTANTS["BUILDING_VOL_SIZE"] // 2
    ey = cy + CONSTANTS["BUILDING_VOL_SIZE"] // 2
    _hf_seg = hf_seg[:, :, sy:ey, sx:ex]

    fg_img = torch.zeros(
        1,
        3,
        CONSTANTS["GES_IMAGE_HEIGHT"],
        CONSTANTS["GES_IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    fg_mask = torch.zeros(
        1,
        1,
        CONSTANTS["GES_IMAGE_HEIGHT"],
        CONSTANTS["GES_IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    # Prevent some buildings are out of bound.
    # THIS SHOULD NEVER HAPPEN AGAIN.
    # if (
    #     _hf_seg.size(2) != CONSTANTS["BUILDING_VOL_SIZE"]
    #     or _hf_seg.size(3) != CONSTANTS["BUILDING_VOL_SIZE"]
    # ):
    #     return fg_img, fg_mask

    # Render foreground patches by patch to avoid OOM
    for i in range(CONSTANTS["GES_IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["GES_IMAGE_WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = _get_pad_img_bbox(sx, ex, sy, ey)

            if torch.count_nonzero(_raydirs[:, sy:ey, sx:ex]) > 0:
                output_fg = gancraft_fg(
                    _hf_seg,
                    _voxel_id[:, psy:pey, psx:pex],
                    depth2[:, psy:pey, psx:pex],
                    _raydirs[:, psy:pey, psx:pex],
                    cam_origin,
                    building_stats=torch.from_numpy(np.array(building_stats)).unsqueeze(
                        dim=0
                    ),
                    z=building_z,
                    deterministic=True,
                )
                facade_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id
                ).unsqueeze(dim=1)
                roof_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id - 1
                ).unsqueeze(dim=1)
                facade_img = facade_mask * _get_img_without_pad(
                    output_fg, sx, ex, sy, ey, psx, pex, psy, pey
                )
                # Make roof blurry
                # output_fg = F.interpolate(
                #     F.interpolate(output_fg * 0.8, scale_factor=0.75),
                #     scale_factor=4 / 3,
                # ),
                roof_img = roof_mask * _get_img_without_pad(
                    output_fg,
                    sx,
                    ex,
                    sy,
                    ey,
                    psx,
                    pex,
                    psy,
                    pey,
                )
                fg_mask[:, :, sy:ey, sx:ex] = torch.logical_or(facade_mask, roof_mask)
                fg_img[:, :, sy:ey, sx:ex] = (
                    facade_img * facade_mask + roof_img * roof_mask
                )

    return fg_img, fg_mask


def render(
    patch_size,
    seg_volume,
    hf_seg,
    cam_pos,
    gancraft_bg,
    gancraft_fg,
    building_stats,
    bg_z,
    building_zs,
):
    voxel_id, depth2, raydirs, cam_origin = get_voxel_intersection_perspective(
        seg_volume, cam_pos
    )
    buildings = torch.unique(voxel_id[voxel_id > CONSTANTS["BLD_INS_LABEL_MIN"]])
    # Remove odd numbers from the list because they are reserved by roofs.
    buildings = buildings[buildings % 2 == 0]
    with torch.no_grad():
        bg_img = render_bg(
            patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_origin, bg_z
        )
        for b in buildings:
            assert b % 2 == 0, "Building Instance ID MUST be an even number."
            fg_img, fg_mask = render_fg(
                patch_size,
                gancraft_fg,
                b.item(),
                hf_seg,
                voxel_id,
                depth2,
                raydirs,
                cam_origin,
                building_stats[b.item()],
                building_zs[b.item()],
            )
            bg_img = bg_img * (1 - fg_mask) + fg_img * fg_mask

    return bg_img
