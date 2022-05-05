import cv2
import cv2.aruco as aruco
import numpy as np

# @TODO: use argparse
N_ROW = 7
N_COL = 10
ID_START = 0
MARKER_SIZE = 33
TAG_SPACING_RATE = 0.21
ARUCO_DICTIONARY = aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
OUTPUT_IMAGE_NAME = f"output/april_board_{N_ROW}_{N_COL}_{MARKER_SIZE}mm_A3.png"

""" A4
PAPER_SIZE_HORISONTAL = 297  # [mm]
PAPER_SIZE_VERTICAL = 210  # [mm]
"""

""" A3
"""
PAPER_SIZE_HORISONTAL = 420  # [mm]
PAPER_SIZE_VERTICAL = 297  # [mm]

TAG_SPACING_SIZE = int(MARKER_SIZE * TAG_SPACING_RATE)


def convert_mm_to_pixel(value_mm, dpi=300):
    return int(np.ceil(value_mm / 25.4 * dpi))  # value [mm] / 25.4 [mm/inch] * dpi [pixel/inch]


def paste_aruco_image(back_groud_image, aruco_image, sx, sy):
    h, w = aruco_image.shape[:2]
    back_groud_image[sy : sy + h, sx : sx + w] = aruco_image
    return back_groud_image


def generate_canvas_image():
    """Generate canvas for drawing"""
    paper_size_h = convert_mm_to_pixel(PAPER_SIZE_HORISONTAL)
    paper_size_v = convert_mm_to_pixel(PAPER_SIZE_VERTICAL)
    paper_size_half_h = paper_size_h // 2
    paper_size_half_v = paper_size_v // 2
    paper_size_h = paper_size_half_h * 2
    paper_size_v = paper_size_half_v * 2

    canvas = np.zeros((paper_size_v, paper_size_h), dtype=np.uint8)
    canvas[:] = 255
    return canvas


def draw_aruco_pattern(canvas, aruco_image, start_pos_xy, aruco_size, spacing_size):
    marker_start_pos_x, marker_start_pos_y = start_pos_xy

    rect_size = int((aruco_size + spacing_size * 2) // 2 * 2)
    center_pos_x = marker_start_pos_x + rect_size // 2
    center_pos_y = marker_start_pos_y + rect_size // 2
    aruco_size_h = aruco_size // 2

    """Draw marker corner"""
    canvas[
        center_pos_y - aruco_size_h - spacing_size : center_pos_y - aruco_size_h,
        center_pos_x - aruco_size_h - spacing_size : center_pos_x - aruco_size_h,
    ] = 0
    canvas[
        center_pos_y - aruco_size_h - spacing_size : center_pos_y - aruco_size_h,
        center_pos_x + aruco_size_h : center_pos_x + aruco_size_h + spacing_size,
    ] = 0
    canvas[
        center_pos_y + aruco_size_h : center_pos_y + aruco_size_h + spacing_size,
        center_pos_x - aruco_size_h - spacing_size : center_pos_x - aruco_size_h,
    ] = 0
    canvas[
        center_pos_y + aruco_size_h : center_pos_y + aruco_size_h + spacing_size,
        center_pos_x + aruco_size_h : center_pos_x + aruco_size_h + spacing_size,
    ] = 0

    aruco_image = cv2.resize(aruco_image, (aruco_size_h * 2, aruco_size_h * 2), interpolation=cv2.INTER_NEAREST)

    if (aruco_image.shape[0] != aruco_size_h * 2) or (aruco_image.shape[1] != aruco_size_h * 2):
        raise ValueError(
            f"(aruco_image.shape[0] != aruco_size_h * 2) or (aruco_image.shape[1] != aruco_size_h * 2), aruco_image.shape: {aruco_image.shape}, aruco_size_h: {aruco_size_h}"
        )

    canvas[
        center_pos_y - aruco_size_h : center_pos_y + aruco_size_h,
        center_pos_x - aruco_size_h : center_pos_x + aruco_size_h,
    ] = aruco_image

    return canvas


def generate_tagboard_pattern(canvas, n_markers_per_row=2, n_markers_per_col=4, id_start=0):
    """Setup aruco info"""
    image_height, image_width = canvas.shape

    aruco_size = convert_mm_to_pixel(MARKER_SIZE)
    spacing_size = convert_mm_to_pixel(TAG_SPACING_SIZE)
    start_pos_xy = (spacing_size, spacing_size)
    end_pos_xy = (
        spacing_size * (n_markers_per_col + 1) + aruco_size * n_markers_per_col + start_pos_xy[0],
        spacing_size * (n_markers_per_row + 1) + aruco_size * n_markers_per_row + start_pos_xy[1],
    )

    if (end_pos_xy[0] >= image_width) or (end_pos_xy[1] >= image_height):
        raise ValueError(
            f"(end_pos_xy[0] >= image_width) or (end_pos_xy[1] >= image_height), end_pos_xy:{end_pos_xy}, image_width: {image_width}, image_height: {image_height}"
        )

    aruco_image_list = [
        aruco.drawMarker(ARUCO_DICTIONARY, id, aruco_size) for id in range(id_start, id_start + n_markers_per_row * n_markers_per_col)
    ]

    """ Drawing aruco markers
    """
    for ix in range(n_markers_per_col):
        for iy in range(n_markers_per_row):
            pattern_start_pos_xy = ((aruco_size + spacing_size) * ix + start_pos_xy[0], (aruco_size + spacing_size) * iy + start_pos_xy[1])
            canvas = draw_aruco_pattern(canvas, aruco_image_list[n_markers_per_col * iy + ix], pattern_start_pos_xy, aruco_size, spacing_size)

    return canvas


if __name__ == "__main__":
    """Generate canvas"""
    canvas = generate_canvas_image()
    aruco_image = generate_tagboard_pattern(canvas, n_markers_per_row=N_ROW, n_markers_per_col=N_COL, id_start=ID_START)
    cv2.imwrite(OUTPUT_IMAGE_NAME, aruco_image)
    cv2.waitKey(10)
