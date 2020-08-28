import numpy as np
import skimage.io
import skvideo.io
import matplotlib.pyplot as plt
import io
from PIL import Image


N = 100
fps = 4

# def save_image(data, filename,cmap):
#     sizes = np.shape(data)     
#     fig = plt.figure()
#     fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(data, cmap=cmap)
#     plt.savefig(filename, dpi = sizes[0]) 
#     plt.close()

def plotimage(lr,hr,sr):
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(lr)
    plt.gca().axis('off')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Input')

    plt.subplot(132)
    plt.imshow(sr)
    plt.gca().axis('off')
    plt.xticks([], [])
    plt.yticks([], [])    
    plt.title('Network')

    plt.subplot(133)
    plt.imshow(hr)
    plt.gca().axis('off')
    plt.xticks([], [])
    plt.yticks([], [])    
    plt.title('Ground truth')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    io_buf = io.BytesIO()
    plt.savefig(io_buf, format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    img = Image.open(io_buf).convert("RGB")
    img_arr = np.array(img)
    
    return img_arr

def createVideo(basedir, index):
    lr = Image.open('%s/lr_%d/%d_0000.png' % (basedir, index, index))
    hr = Image.open('%s/hr_%d/%d_0000.png' % (basedir, index, index))

    writer = skvideo.io.FFmpegWriter("%d.avi" % index,inputdict={'-r':str(fps)},outputdict={'-r':str(fps),"-pix_fmt": "yuv420p"})

    for subindex in range(N):
        sr = Image.open('%s/sr_%d/%d_%04d.png' % (basedir, index, index, subindex))

        img_arr = plotimage(lr,hr,sr)
        writer.writeFrame(img_arr)

        print('[%d/%d]' % (subindex+1,N),end='\r')

    writer.close()


createVideo("C:/temp",0)
