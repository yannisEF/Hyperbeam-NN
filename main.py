import pickle
import argparse

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors
from matplotlib.widgets import Slider

from scipy.ndimage import gaussian_filter


def update_image(val, im, slider, slider_blur=None):
    im.set_array(
        gaussian_filter(
            list_layers[int(slider.val)],
            sigma=float(slider_blur.val)
        ) if slider_blur is not None else \
        list_layers[int(slider.val)]
    )

def update_surface(val, ax, X, Y, list_surfaces, slider, slider_blur=None, kwargs={}):
    index = int(slider.val)

    ax.clear()
    ax.plot_surface(
        X, Y,
        Z = gaussian_filter(
            list_surfaces[int(slider.val)],
            sigma=float(slider_blur.val)
        ) if slider_blur is not None else \
            np.array(list_surfaces[int(slider.val)]),
        **kwargs
    )

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--to_show', default='test', type=str)
    args = parser.parse_args()

    with open('Results/{}.pkl'.format(args.to_show), 'rb') as handle:
        list_layers = pickle.load(handle)


    colormap = cm.viridis

    slider_position = [0.15, 0.05, 0.70, 0.04]
    slider_kwargs = {
        "label":"Depth",
        "valmin":0, "valmax":len(list_layers)-1,
        "valinit":0, "valstep":1 , "valfmt":'%d'        
    }

    slider_blur_position = [0.15, 0.01, 0.70, 0.04]
    slider_blur_kwargs = {
        "label":"Gaussian blur",
        "valmin":0, "valmax":1.5,
        "valinit":0, "valstep":.05       
    }

    v_min, v_max = np.min(list_layers), np.max(list_layers)
    norm = colors.Normalize(vmin=v_min, vmax=v_max)

    imshow_kwargs = {
        "interpolation":"None",
        "norm":norm, "cmap":colormap
    }

    surface_kwargs = {
        "norm":norm, "cmap":colormap,
        "linewidth":0, "antialiased":True
    }

    # 2D plot
    fig_2D, ax_2D = plt.subplots(figsize=(8, 8))

    im=ax_2D.imshow(
        list_layers[0],
        **imshow_kwargs
    )

    fig_2D.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colormap),
        orientation="vertical", label="Reward",
        ax=ax_2D, shrink=.5
    )

    slider_2D = Slider(plt.axes(slider_position), **slider_kwargs)
    slider_blur_2D = Slider(plt.axes(slider_blur_position), **slider_blur_kwargs)

    func_update_2D = lambda val: update_image(val, im, slider_2D, slider_blur_2D)
    slider_2D.on_changed(func_update_2D)
    slider_blur_2D.on_changed(func_update_2D)

    # 3D plot
    fig_3D = plt.figure(figsize=(8,8))
    ax_3D = fig_3D.add_subplot(projection='3d')

    fig_3D.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colormap),
        orientation="vertical", label="Reward",
        ax=ax_3D, shrink=.5
    )

    #   Surfaces are plotted from top to bottom...
    list_layers3D = [layer[::-1] for layer in list_layers]
    
    X, Y = np.meshgrid(
        list(range(len(list_layers3D[0][0]))),
        list(range(len(list_layers3D[0])))
    )

    surf = ax_3D.plot_surface(X, Y, np.array(list_layers3D[0]), **surface_kwargs)

    slider_3D = Slider(plt.axes(slider_position), **slider_kwargs)
    slider_blur_3D = Slider(plt.axes(slider_blur_position), **slider_blur_kwargs)

    func_update_3D = lambda val: update_surface(
        val, ax_3D,
        X, Y, list_layers3D,
        slider_3D, slider_blur_3D,
        surface_kwargs
    )

    slider_3D.on_changed(func_update_3D)
    slider_blur_3D.on_changed(func_update_3D)

    plt.show()