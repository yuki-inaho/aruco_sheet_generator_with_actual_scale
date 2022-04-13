import cv2
import cv2.aruco as aruco
from marker_sheet_generator import PATTERN_MARGIN
import numpy as np

# @TODO: use argparse
ID_START = 0
OUTPUT_IMAGE_NAME = "output/april16h5_ar_033.png"
# ARUCO_DICTIONARY = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
ARUCO_DICTIONARY = aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

PAPER_SIZE_HORISONTAL = 297  # [mm]
PAPER_SIZE_VERTICAL = 210  # [mm]

MARKER_SIZE = 80  # [mm]
HEIGHT_PER_WIDTH_RATIO = 0.5
RECT_WIDTH_SIZE = int(1.2 * MARKER_SIZE)
CORNER_RECT_SIZE_RATE = 0.1
PATTERN_MARGIN = 2


def convert_mm_to_pixel(value_mm, dpi=300):
    return int(np.ceil(value_mm / 25.4 * dpi))  # value [mm] / 25.4 [mm/inch] * dpi [pixel/inch]


def paste_aruco_image(back_groud_image, aruco_image, sx, sy):
    h, w = aruco_image.shape[:2]
    back_groud_image[sy : sy + h, sx : sx + w] = aruco_image
    return back_groud_image


def generate_canvas_image():
    """Get aruco marker dictionary"""
    aruco_size = convert_mm_to_pixel(MARKER_SIZE)

    """ Generate canvas for drawing
    """
    paper_size_h = convert_mm_to_pixel(PAPER_SIZE_HORISONTAL)
    paper_size_v = convert_mm_to_pixel(PAPER_SIZE_VERTICAL)
    paper_size_half_h = paper_size_h // 2
    paper_size_half_v = paper_size_v // 2
    paper_size_h = paper_size_half_h * 2
    paper_size_v = paper_size_half_v * 2

    canvas = np.zeros((paper_size_v, paper_size_h), dtype=np.uint8)
    canvas[:] = 255
    return canvas


def draw_aruco_pattern(
    canvas,
    aruco_image_raw,
    marker_start_pos_x,
    marker_start_pos_y,
    aruco_size_width,
    aruco_size_height,
    corner_size,
    rect_width_size,
    rect_height_size,
):
    aruco_image = cv2.resize(aruco_image_raw, None, fx=1.0, fy=HEIGHT_PER_WIDTH_RATIO, interpolation=cv2.INTER_NEAREST)

    center_pos_x = marker_start_pos_x + rect_width_size // 2
    center_pos_y = marker_start_pos_y + rect_height_size // 2
    marker_end_pos_x = marker_start_pos_x + rect_width_size
    marker_end_pos_y = marker_start_pos_y + rect_height_size
    aruco_size_half_width = aruco_size_width // 2
    aruco_size_half_height = aruco_size_height // 2

    """Draw marker corner"""
    canvas[marker_start_pos_y:marker_end_pos_y, marker_start_pos_x:marker_end_pos_x] = 255
    canvas[
        marker_start_pos_y : center_pos_y - aruco_size_half_height,
        marker_start_pos_x : center_pos_x - aruco_size_half_width,
    ] = 0
    canvas[
        center_pos_y + aruco_size_half_height : marker_end_pos_y,
        marker_start_pos_x : center_pos_x - aruco_size_half_width,
    ] = 0
    canvas[
        marker_start_pos_y : center_pos_y - aruco_size_half_height,
        center_pos_x + aruco_size_half_width : marker_end_pos_x,
    ] = 0
    canvas[
        center_pos_y + aruco_size_half_height : marker_end_pos_y,
        center_pos_x + aruco_size_half_width : marker_end_pos_x,
    ] = 0

    aruco_image = cv2.resize(aruco_image_raw, (aruco_size_half_width * 2, aruco_size_half_height * 2), interpolation=cv2.INTER_NEAREST)
    canvas[
        center_pos_y - aruco_size_half_height : center_pos_y + aruco_size_half_height,
        center_pos_x - aruco_size_half_width : center_pos_x + aruco_size_half_width
    ] = aruco_image

    canvas = cv2.rectangle(
        canvas,
        (marker_start_pos_x, marker_start_pos_y),
        (marker_end_pos_x, marker_end_pos_y),
        color=0,
        thickness=1,
        lineType=cv2.LINE_4,
        shift=0,
    )

    return canvas


def draw_aruco(pattern_margin, canvas, id_start=0):
    """Setup aruco info"""
    aruco_size = convert_mm_to_pixel(MARKER_SIZE)
    corner_rect_size = int(aruco_size * CORNER_RECT_SIZE_RATE)
    aruco_image_raw = aruco.drawMarker(ARUCO_DICTIONARY, id_start, aruco_size)
    aruco_size_width = aruco_size
    aruco_size_height = int(aruco_size * HEIGHT_PER_WIDTH_RATIO)
    rect_width_size = convert_mm_to_pixel(RECT_WIDTH_SIZE)
    rect_height_size = int(rect_width_size * HEIGHT_PER_WIDTH_RATIO)

    """ Drawing aruco markers
    """
    pattern_start_pos_x = pattern_margin + (rect_width_size + pattern_margin) * 0
    pattern_start_pos_y = pattern_margin + (rect_height_size + pattern_margin) * 0
    canvas = draw_aruco_pattern(
        canvas,
        aruco_image_raw,
        pattern_start_pos_x,
        pattern_start_pos_y,
        aruco_size_width,
        aruco_size_height,
        corner_rect_size,
        rect_width_size,
        rect_height_size,
    )
    return canvas


if __name__ == "__main__":
    """Generate canvas"""
    canvas = generate_canvas_image()
    aruco_image = draw_aruco(convert_mm_to_pixel(PATTERN_MARGIN), canvas, id_start=ID_START)
    cv2.imwrite(OUTPUT_IMAGE_NAME, aruco_image)
    cv2.waitKey(10)
