""" ----------------------------------------
* Creation Time : Tue Jun 13 14:18:20 2023
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import io
import tifffile as tiff
import os


# Function to draw a rectangle given four points
def draw_rectangle(points, ax):
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Compute the width and height
    width = np.linalg.norm(points[1] - points[0])
    height = np.linalg.norm(points[2] - points[1])

    # Compute the angle
    angle = np.arctan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0])

    # Create the rectangle
    rect = patches.Rectangle(
        centroid,
        width,
        height,
        angle=np.degrees(angle),
        linewidth=1,
        edgecolor="gray",
        facecolor="none",
        transform=ax.transData,
    )

    # Add the rectangle
    ax.add_patch(rect)
    ax.set_aspect("equal")


def draw_rectangle_from_coords(ax, ridx, cidx):
    l = np.sqrt(2) / 2  # unit length
    o = 8  # origin (x,y)

    shift = 1 if ridx % 2 == 0 else 0

    x1 = o - 2 * l * cidx + shift * l
    y1 = o - l * ridx
    x2 = o - l - 2 * l * cidx + shift * l
    y2 = o + l - l * ridx
    x3 = o - 2 * l * cidx + shift * l
    y3 = o + 2 * l - l * ridx
    x4 = o + l - 2 * l * cidx + shift * l
    y4 = o + l - l * ridx

    # apply shift based on row and column

    # Parse points
    points = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(-1, 2)

    draw_rectangle(points, ax)

    # annotate ridx and cidx in the center of the rectangle
    ax.annotate(
        f"{ridx},{cidx}",
        xy=(x2, (y1 + y3) / 2),
        horizontalalignment="center",
        verticalalignment="center",
        color="orange",
    )

    # return min max x and y
    return (
        min(x1, x2, x3, x4),
        max(x1, x2, x3, x4),
        min(y1, y2, y3, y4),
        max(y1, y2, y3, y4),
    )


def plot_patch(patch):
    fig, ax = plt.subplots()

    bounds_list = []

    # iterate through patch and render if pixel is 1
    for row in range(patch.shape[0]):
        for col in range(patch.shape[1]):
            if patch[row, col] > 100:
                bounds = draw_rectangle_from_coords(ax, row, col)
                print(bounds)
                bounds_list.append(bounds)

    # Get the min and max x and y values
    xmin = min([b[0] for b in bounds_list])
    xmax = max([b[1] for b in bounds_list])
    ymin = min([b[2] for b in bounds_list])
    ymax = max([b[3] for b in bounds_list])

    # Set the limits of the plot
    plt.xlim(xmin - 2, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)

    return fig, xmin, xmax, ymin, ymax


def DMDPixelTransform(input_img, dmdMapping, xoffset=0, yoffset=0):
    # Initialize an array of zeros with same size as the input image
    transformed_img = np.zeros_like(input_img)

    # Get the dimensions of the input image
    rows, cols = input_img.shape

    # Iterate over the pixels of the input image
    for i in range(rows):
        for j in range(cols):
            # Calculate the new coordinates for the pixel
            ip = i + yoffset
            jp = j + xoffset

            # Apply the dmdMapping transformation if set
            if dmdMapping > 0:
                transformed_i = jp + ip - 2
                transformed_j = (jp - ip + 4) // 2
            else:
                transformed_i = ip
                transformed_j = jp

            # If the new coordinates are within the bounds of the image, copy the pixel value
            if 0 <= transformed_i < rows and 0 <= transformed_j < cols:
                transformed_img[transformed_i, transformed_j] = input_img[i, j]

    # Return the transformed image
    return transformed_img


# Streamlit app
st.title("ONI DMD emulator")

# add author info and context
st.markdown("Charles N. Christensen, ONI — 2023/06/14")


tabs = st.tabs(
    [
        "Basic rendering",
        "Pregenerated pattern with padding",
        "Manual pattern definition",
    ]
)


with tabs[0]:
    st.header("Render DMD layout")
    # Create a new figure
    fig, ax = plt.subplots()

    # streamlit input for number of columns and rows
    cols = st.number_input("Number of columns", min_value=1, max_value=20, value=5)
    rows = st.number_input("Number of rows", min_value=1, max_value=20, value=5)

    bounds_list = []

    for col in range(cols):
        for row in range(rows):
            bounds = draw_rectangle_from_coords(ax, row, col)
            bounds_list.append(bounds)

    # Get the min and max x and y values
    xmin = min([b[0] for b in bounds_list])
    xmax = max([b[1] for b in bounds_list])
    ymin = min([b[2] for b in bounds_list])
    ymax = max([b[3] for b in bounds_list])

    # Set the limits of the plot
    plt.xlim(xmin - 2, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)

    fig.set_size_inches(cols + 0.5, rows + 0.5)

    st.pyplot(fig)


with tabs[1]:
    st.subheader("Upload pregenerated pattern or use default")
    # upload pattern image
    pattern = st.file_uploader("Upload pattern image", type=["tif", "tiff"])

    # fallback
    if pattern is None:
        option = st.selectbox(
            "Select pattern",
            (
                "patterns_spotSize_2_Nspots_5_dmdMapping_1.tif",
                "patterns_spotSize_5_Nspots_20_dmdMapping_1.tif",
                "patterns_spotSize_1_Nspots_5_dmdMapping_1.tif",
                "patterns_pixelsize_ratio_1_k2_200_func_square_wave_one_third_dmdMapping_1.tif",
                "patterns_pixelsize_ratio_1_k2_200_func_square_wave_one_third_dmdMapping_0.tif",
                "patterns_pixelsize_ratio_1_k2_80_func_square_wave_one_third_dmdMapping_1.tif",
                "patterns_pixelsize_ratio_1_k2_80_func_square_wave_one_third_dmdMapping_0.tif",
                "patterns_pixelsize_ratio_1_k2_20_func_square_wave_one_third_dmdMapping_1.tif",
                "patterns_pixelsize_ratio_1.8_k2_200_func_square_wave_one_third.tif",
                "patterns_pixelsize_ratio_1.8_k2_150_func_square_wave_one_third.tif",
                "patterns_pixelsize_ratio_1.8_k2_110_func_square_wave_one_third.tif",
                "patterns_pixelsize_ratio_1.6_k2_80.tif",
            ),
        )
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        def_pattern = f"{cur_dir}/patterns/{option}"
        img = io.imread(def_pattern)
        st.markdown("""**No pattern uploaded**: Loading selected default image.""")
        st.markdown(
            "Note that `pixelsize_ratio > 1` and `_dmdMapping_0` indicate an assumption of ortholinear grid (no DMD layout correction). The `_dmdMapping_1` patterns are corrected for DMD."
        )
    else:
        print("loading image", pattern)
        img = tiff.imread(pattern)

    st.subheader("Parameters")
    cols = st.columns(2)
    # streamlit number input for padding
    with cols[0]:
        x_padding = st.number_input(
            "Vertical padding", min_value=0, max_value=10, value=0
        )
    with cols[1]:
        y_padding = st.number_input(
            "Horizontal padding", min_value=0, max_value=10, value=0
        )

    # streamlit number input for global offset
    cols = st.columns(2)
    with cols[0]:
        x_offset_img = st.number_input(
            "Vertical image offset (change to test near edges of DMD)",
            min_value=0,
            max_value=30,
            value=10,
        )
    with cols[1]:
        y_offset_img = st.number_input(
            "Horizontal image offset (change to test near edges of DMD)",
            min_value=0,
            max_value=30,
            value=20,
        )

    cols = st.columns(2)
    with cols[0]:
        rows_img = st.number_input(
            "Number of rows", min_value=5, max_value=100, value=20
        )
    with cols[1]:
        cols_img = st.number_input(
            "Number of columns", min_value=5, max_value=100, value=20
        )

    cols = st.columns(2)
    with cols[0]:
        frame_idx = st.number_input(
            "Plot frame index start", min_value=0, max_value=400, value=0
        )
    with cols[1]:
        frame_count = st.number_input(
            "Plot frame index range", min_value=1, max_value=5, value=2
        )

    global_bounds = None

    # plot patches in two frames
    for i in range(frame_idx, frame_idx + frame_count):
        patch = img[
            i,
            y_offset_img : y_offset_img + rows_img,
            x_offset_img : x_offset_img + cols_img,
        ]

        # add 1pixel padding in top to patch
        patch = np.pad(
            patch, ((x_padding, 0), (y_padding, 0)), "constant", constant_values=0
        )

        fig, xmin, xmax, ymin, ymax = plot_patch(patch)

        if global_bounds is None:
            global_bounds = xmin, xmax, ymin, ymax

        xmin, xmax, ymin, ymax = global_bounds
        plt.xlim(xmin - 2, xmax + 1)
        plt.ylim(ymin - 2, ymax + 1)
        fig.set_size_inches(xmax - xmin + 4, ymax - ymin + 4)
        st.subheader(f"Frame {i}")
        st.pyplot(fig)


with tabs[2]:
    # 5x5 array of st checkboxes
    st.subheader("Select pixels to turn on in tilted coordinate system")
    grid = np.zeros((5, 5))
    for i in range(5):
        cols = st.columns(5)
        for j in range(5):
            with cols[j]:
                # set value based on i,j = 0,0 1,0 0,1 1,1

                value = False
                if i == 1 or i == 2:
                    if j == 1 or j == 2:
                        value = True

                grid[i, j] = st.checkbox(f"{i},{j}", value=value, key=f"{i},{j}")
    grid = 255 * grid

    # streamlit number input for global offset
    x_offset = st.number_input(
        "Vertical global offset (change to test near edges of DMD)",
        min_value=0,
        max_value=30,
        value=10,
    )
    y_offset = st.number_input(
        "Horizontal global offset (change to test near edges of DMD)",
        min_value=0,
        max_value=30,
        value=20,
    )

    global_bounds = None

    # plot with offsets of 1px in x and y
    for xoffset in range(0, 2):
        for yoffset in range(0, 2):
            patch = np.zeros((40, 40))
            patch[x_offset : x_offset + 5, y_offset : y_offset + 5] = grid
            patch = DMDPixelTransform(patch, 1, xoffset, yoffset)
            fig, xmin, xmax, ymin, ymax = plot_patch(patch)

            if global_bounds is None:
                global_bounds = xmin, xmax, ymin, ymax

            xmin, xmax, ymin, ymax = global_bounds
            plt.xlim(xmin - 2, xmax + 1)
            plt.ylim(ymin - 2, ymax + 1)
            fig.set_size_inches(xmax - xmin + 4, ymax - ymin + 4)
            st.subheader(f"Frame with offset x={xoffset} and y={yoffset}")
            st.pyplot(fig)
