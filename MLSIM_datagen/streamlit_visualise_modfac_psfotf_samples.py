''' ----------------------------------------
* Creation Time : Thu 15 Jun 2023 02:25:19 BST
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------'''

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import re
import os
import glob

files = glob.glob('/home/cc/2work/onirepos/SIM/PhD-ccRestore/runs/bash-script-modfac-psf/**/0002_0.tif')

def get_modfac_psf(s):
    ''' Get modulation factor and psf from filename '''
    modfac_values = re.findall(r'modfac-(\d+\.\d+)', s)
    psf_values = re.findall(r'psf-(\d+\.\d+)', s)

    new_string = f'modfac:{modfac_values[0]}_psf:{psf_values[0]}'
    return new_string, modfac_values[0], psf_values[0]

# generate tabs
# tabs = [get_modfac_psf(file) for file in files]
# lookup = {get_modfac_psf(k)[0]: for k in files}

modfac_values = [get_modfac_psf(file)[1] for file in files]
modfac_values = np.unique(modfac_values)
psf_values = [get_modfac_psf(file)[2] for file in files]
psf_values = np.unique(psf_values)
lookup = {get_modfac_psf(k)[0]: k for k in files}

# radiogroup for each
selected_modfac = st.radio('Select modulation factor', modfac_values, horizontal=True)
selected_psf = st.radio('Select psf', psf_values, horizontal=True)

st.header(f'Modfac: {selected_modfac} - PSF: {selected_psf}')
key = f'modfac:{selected_modfac}_psf:{selected_psf}'
file = lookup[key]
img = io.imread(file)

st.subheader(f'File: {file}')

cols = st.columns(2)

with cols[0]:
    st.image(img[0], caption='frame 0', use_column_width=True)

with cols[1]:
    st.image(img_as_float(img).mean(axis=0), caption='wf', use_column_width=True)

# for fidx, file in enumerate(files):

#     img = io.imread(file)

#     pardir = os.path.basename(os.path.dirname(file))

#     with tabs[fidx]:
#         st.header(f'File: {pardir}')

#         # plot two frames
#         cols = st.columns(2)

#         with cols[0]:
#             st.image(img[0], caption='Original', use_column_width=True)

#         with cols[1]:
#             st.image(img[1], caption='Restored', use_column_width=True)

