import pickle
import argparse

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors
from matplotlib.widgets import Slider

from scipy.ndimage import gaussian_filter


def update_image(val, list_layers, im, slider, slider_blur=None, interpolation=True):
    index = int(slider.val)

    list_layers = np.array(list_layers)

    if (interpolation is True) and index != len(list_layers)-1:
        weight = slider.val - index
        layer = weight * list_layers[index+1] + (1 - weight) * list_layers[index]
    else:
        layer = list_layers[index]

    im.set_array(
        gaussian_filter(
            layer,
            sigma=float(slider_blur.val)
        ) if slider_blur is not None else \
        layer
    )

def update_surface(val, ax, X, Y, list_surfaces, slider, slider_blur=None, kwargs={}, interpolation=True):
    index = int(slider.val)

    list_surfaces = np.array(list_surfaces)

    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])

    if (interpolation is True) and index != len(list_surfaces)-1:
        weight = slider.val - index
        surface = weight * list_surfaces[index+1] + (1 - weight) * list_surfaces[index]
    else:
        surface = list_surfaces[index]

    ax.plot_surface(
        X, Y,
        Z = gaussian_filter(
            surface,
            sigma=float(slider_blur.val)
        ) if slider_blur is not None else \
            np.array(surface),
        **kwargs
    )

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--to_show', default='test', type=str,
        help="Name of the .pkl file to load."
    )
    parser.add_argument(
        '--transposed', default='False', type=str,
        help="Should the beam be transposed? true/false"
    )
    args = parser.parse_args()

    with open(f'Results/{args.to_show}.pkl', 'rb') as handle:
        list_layers = np.array(pickle.load(handle))

    if args.transposed.lower() == 'true':
        list_layers = list_layers.T

    colormap = cm.viridis

    slider_position = [0.15, 0.05, 0.70, 0.04]
    slider_kwargs = {
        "label":"Depth",
        "valmin":0, "valmax":len(list_layers)-1,
        "valinit":0, "valstep":.1 , "valfmt":'%d'        
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
    ax_2D.set_xticks([])
    ax_2D.set_yticks([])

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

    func_update_2D = lambda val: update_image(val, list_layers, im, slider_2D, slider_blur_2D)
    slider_2D.on_changed(func_update_2D)
    slider_blur_2D.on_changed(func_update_2D)

    # 3D plot
    fig_3D = plt.figure(figsize=(8,8))
    ax_3D = fig_3D.add_subplot(projection='3d')
    ax_3D.set_xticks([])
    ax_3D.set_yticks([])

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