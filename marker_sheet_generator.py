import cv2
import cv2.aruco as aruco
import numpy as np

# @TODO: use argparse
N_ROW = 2
N_COL = 3
ID_START = 24
OUTPUT_IMAGE_NAME = "output/april36h11_24_29.png"
# ARUCO_DICTIONARY = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
ARUCO_DICTIONARY = aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

PAPER_SIZE_HORISONTAL = 297  # [mm]
PAPER_SIZE_VERTICAL = 210  # [mm]

MARKER_SIZE = 60  # [mm]
RECT_SIZE = int(1.2 * MARKER_SIZE)
PATTERN_MARGIN = 5
CORNER_RECT_SIZE_RATE = 0.1


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


def draw_aruco_pattern(canvas, aruco_image, marker_start_pos_x, marker_start_pos_y, aruco_size, corner_size, rect_size):
    center_pos_x = marker_start_pos_x + rect_size // 2
    center_pos_y = marker_start_pos_y + rect_size // 2
    aruco_size_h = aruco_size // 2

    """Draw marker corner"""
    canvas[
        center_pos_y - aruco_size_h - corner_size : center_pos_y - aruco_size_h,
        center_pos_x - aruco_size_h - corner_size : center_pos_x - aruco_size_h,
    ] = 0
    canvas[
        center_pos_y - aruco_size_h - corner_size : center_pos_y - aruco_size_h,
        center_pos_x + aruco_size_h : center_pos_x + aruco_size_h + corner_size,
    ] = 0
    canvas[
        center_pos_y + aruco_size_h : center_pos_y + aruco_size_h + corner_size,
        center_pos_x - aruco_size_h - corner_size : center_pos_x - aruco_size_h,
    ] = 0
    canvas[
        center_pos_y + aruco_size_h : center_pos_y + aruco_size_h + corner_size,
        center_pos_x + aruco_size_h : center_pos_x + aruco_size_h + corner_size,
    ] = 0

    aruco_image = cv2.resize(aruco_image, (aruco_size_h * 2, aruco_size_h * 2), interpolation=cv2.INTER_NEAREST)
    canvas[
        center_pos_y - aruco_size_h : center_pos_y + aruco_size_h,
        center_pos_x - aruco_size_h : center_pos_x + aruco_size_h,
    ] = aruco_image

    canvas = cv2.rectangle(
        canvas,
        (marker_start_pos_x, marker_start_pos_y),
        (marker_start_pos_x + rect_size, marker_start_pos_y + rect_size),
        color=0,
        thickness=1,
        lineType=cv2.LINE_4,
        shift=0,
    )

    return canvas


def draw_aruco(pattern_margin, canvas, n_markers_per_row=2, n_markers_per_col=4, id_start=0):
    """Setup aruco info"""

    aruco_size = convert_mm_to_pixel(MARKER_SIZE)
    corner_rect_size = int(aruco_size * CORNER_RECT_SIZE_RATE)
    aruco_image_list = [
        aruco.drawMarker(ARUCO_DICTIONARY, id, aruco_size) for id in range(id_start, id_start + n_markers_per_row * n_markers_per_col)
    ]
    rect_size = convert_mm_to_pixel(RECT_SIZE)

    """ Setup paper info
    """
    paper_size_h = convert_mm_to_pixel(PAPER_SIZE_HORISONTAL)
    paper_size_v = convert_mm_to_pixel(PAPER_SIZE_VERTICAL)
    paper_size_half_h = paper_size_h // 2
    paper_size_half_v = paper_size_v // 2

    """ Drawing aruco markers
    """
    for i in range(n_markers_per_row):
        for j in range(n_markers_per_col):
            pattern_start_pos = (pattern_margin + (rect_size + pattern_margin) * i, pattern_margin + (rect_size + pattern_margin) * j)  # y-x
            canvas = draw_aruco_pattern(
                canvas,
                aruco_image_list[j + n_markers_per_col * i],
                pattern_start_pos[1],
                pattern_start_pos[0],
                aruco_size,
                corner_rect_size,
                rect_size,
            )

    return canvas


if __name__ == "__main__":
    """Generate canvas"""
    canvas = generate_canvas_image()
    aruco_image = draw_aruco(convert_mm_to_pixel(PATTERN_MARGIN), canvas, n_markers_per_row=N_ROW, n_markers_per_col=N_COL, id_start=ID_START)
    cv2.imwrite(OUTPUT_IMAGE_NAME, aruco_image)
    cv2.waitKey(10)
