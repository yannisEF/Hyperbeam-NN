"""Functions used to find and sample hyperplanes orthognal to a vector.
TODO: Needs to be rethought for better data compression."""

import pickle
import numpy as np

from scipy.stats import truncnorm
from tqdm import tqdm

import utils

# ------------------------------- Orthogonalization -------------------------------

def get_orthogonal(u, u_dot=None):
    """Returns a random vector orthogonal from u."""

    if len(u.shape) > 1:
        raise ValueError("Input not a vector.")

    if u_dot is None:
        u_dot = np.dot(u, u)
    
    v = np.random.uniform(low=-1, high=1, size=u.size)
    result = u_dot * v - np.dot(v, u) * u
    return result

def get_hyperplane(u, verbose=False):
    """Returns an orthonormal basis of the hyperplane orthogonal to u using the Gram-Schmidt procedure."""

    # Hyperplane (dimension N-1)
    plane_dim = u.size - 1
    iterator = tqdm(range(plane_dim-1), desc="Hyperplane basis") if verbose is True \
        else range(plane_dim-1)

    # Gram-Schmidt
    u_dot = np.dot(u, u)
    start_vector = get_orthogonal(u, u_dot)
    basis = [(start_vector, np.dot(start_vector, start_vector))]
    for _ in iterator:
        # Generate N-1 random vector orthogonal to u
        # ...we assume that we end up with a linearly independent set...
        new_vector = get_orthogonal(u, u_dot)

        # Remove the projection on its family
        vect = new_vector - np.sum(
            np.dot(new_vector, b[0] / b[1]) * b[0] 
            for b in basis
        )

        basis.append((vect, np.dot(vect, vect)))
    
    # Normalize
    return np.array([b[0] / np.sqrt(b[1]) for b in basis])

# ------------------------------- Plane sampling - Utils -------------------------------

def center_around(x, basis, combination):
    """Returns an array of points centered around x by combining the basis vectors in the desired way. The center X is always at the first index."""
    
    S = [x]
    for c in combination:
        # No null combination (x already in S)
        if all([i == 0 for i in c]):
            continue

        add = np.zeros(len(x))
        for i in range(len(basis)):
            add += c[i] * basis[i]
        S.append(x + add)

    return np.array(S)

def sample_beam(S, u, N):
    """Returns N layers of a beam, from a surface S shifted by vector u."""
    return [
        np.array([surface_sample + k * u for surface_sample in S])
        for k in np.linspace(0, 1, num=N)
    ]

# ------------------------------- Plane sampling -------------------------------

def combination_random(dim, N):
    """Returns N random combinations."""
    return 2 * (np.random.random((N,dim)) - .5)

def combination_normal(dim, sigma, N):
    """Returns N random normal combinations."""
    return list([np.random.normal(scale=sigma, size=dim) for _ in range(N)])

def combination_sphere(dim, size, N):
    """Returns N uniform random samples of the sphere of radius size."""
    return [
        size * s / np.linalg.norm(s) for s in combination_normal(dim, 1, N)
    ]

def vector_boxmuller(N, dim):
    """Returns N points sampled using Box-Muller sampling rule of dim-dimensional balls."""
    # return [0] + sorted(list(np.random.random(N-1) ** (1/dim)))
    # r = np.random.normal(scale=1, size=N-1)
    r = truncnorm.rvs(0,3, size=N-1) / 3 # Hard coding values for fun
    return [0] + list(np.sort(np.abs(r)))

def sample_landscape(L, N):
    """
    Returns the sampled lines from the center to every point of the layer.
    By ordering the lines in a consistent manner, we get the floor of a landscape.
    Each line will be made of 2*N + 1 pixels.
    """

    landscape = []

    center = L[0]
    dim = len(center)

    for point in L[1:]:
        # Sample this direction, towards and away from the point
        centered_point = point - center
        dir_towards = [-v * centered_point for v in vector_boxmuller(N+1, dim-1)[::-1]]
        dir_away = [v * centered_point for v in vector_boxmuller(N+1, dim-1)[1:]]
        sampled_direction = dir_towards + dir_away

        landscape.append(np.array([
            center + vector_shift 
            for vector_shift in sampled_direction
        ]))

    return landscape

def get_landscape_beam(beam, pixels_per_line, save_path=None):
    """

    NOT OPTIMIZED EXAMPLE
    see model_prepare.py for better implementation

    For each layer of a bream, we sample the resulting landscape.
    For a consistent vizualisation of underlying structures, we organize the landscape's
    lines by closest neighbours.

    This can take a lot of RAM, hence if save_path is not None data is saved layer by layer and RAM freed in the desired path.
    """    

    list_ordered_landscapes = []

    indices = None
    for i, layer in enumerate(beam):

        # We sample the landscape from the layer
        landscape = sample_landscape(layer, pixels_per_line // 2)

        # We memorize the permutations for the coherent organization along the beam
        if indices is None:
            indices = utils.order_neighbours(landscape)[-1]
        
        # We organize the landscape by closest neighbours
        ordered_landscape = utils.insert_with_indices(landscape, indices)

        # We save the data or keep it in memory
        if save_path is not None:
            with open("{}/{}.pkl".format(save_path, i), 'wb') as handle:
                pickle.dump(ordered_landscape, handle)
        else:
            list_ordered_landscapes.append(ordered_landscape)   

    return list_ordered_landscapes

# ------------------------------- Example -------------------------------

if __name__=="__main__":

    dimension = 3

    nb_layers = 3
    nb_lines_per_layer = 25
    pixels_per_line = 6

    # Use 3D for visualization
    #   /!\ The ax XYZ limits can change the perception of angles and distances
    if dimension == 3:
        # Fixing the seed for 3D to 12121
        #   ... we hard coded some visualization variables according to this seed
        np.random.seed(12121)

    # We have a random vector corresponding to the learning between two policies
    u = 3 * np.random.random(dimension)
    # We find the hyperplane orthogonal to that vector
    basis = get_hyperplane(u)

    # We set up a sampling rule for that hyperplane
    origin1 = np.zeros(len(u))
    combination1 = combination_sphere(
        len(basis),
        np.linalg.norm(u) / 3, # Random value
        nb_lines_per_layer
    )
    S1 = center_around(origin1, basis, combination1)

    #   ... for the example, we show other sampling rules
    # origin2 = 3 * np.eye(len(u))[0]
    # combination2 = combination_random(len(basis), nb_lines_per_layer)
    # S2 = center_around(origin2, basis, combination2)

    # Now we sample the beam created by shifting our hyperplane by u
    beam1 = sample_beam(S1, u, nb_layers)
    # beam2 = sample_beam(S2, u, nb_layers)

    # For each of point, we trace a line to the center of the layer
    #   ... this is because the hyperplane is N-th dimensional
    #   ... and we have to use another technique later to visualize
    #   ... these N dimensions in 2D or 3D.
    list_landscape1 = get_landscape_beam(beam1, pixels_per_line)
    # list_landscape2 = get_landscape_beam(beam2, pixels_per_line)

    # We are now supposed to evaluate each of our landscapes on any environment!
    #   ... With can use their fitness as a value for a pixel in an image.
    #   ... With a slider, we can show the image for each landscape along the beam.

    # For the example, let's visualize our samples in the parameter's space...
    if dimension == 3:

        import matplotlib.pyplot as plt

        color_vector = "red"
        color_basis = "blue"

        alpha_surface = .4
        color_surface = "green"

        alpha_landscape = .2
        color_landscape = "orange"

        ax = plt.figure().add_subplot(projection='3d')

        def plot_vector(ax, origin, u, color):
            ax.plot(
                [origin[0], origin[0] + u[0]],
                [origin[1], origin[1] + u[1]],
                [origin[2], origin[2] + u[2]],
                color=color
            )
        
        # REMOVE IF NO SEEDING
        def line_bottom(ax, point, color, text=""):
            alpha = .6

            ax.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2], -1.1020904882813967],
                color=color,
                alpha=alpha,
                linestyle="dotted"
            )

            offset = .1
            ax.plot(
                [point[0]-offset, point[0]+offset],
                [point[1]+offset, point[1]-offset],
                [-1.1020904882813967, -1.1020904882813967],
                color=color,
                alpha=alpha,
                linestyle="dashed"
            )

            ax.plot(
                [point[0]-offset, point[0]+offset],
                [point[1]-offset, point[1]+offset],
                [-1.1020904882813967, -1.1020904882813967],
                color=color,
                alpha=alpha,
                linestyle="dashed"
            )

            ax.text(
                *(np.array([point[0], point[1], -1.1020904882813967]) - 2*offset),
                s = text, fontsize = 18
            )
        # ---

        # for origin in [origin1, origin2]:
        for origin in [origin1]:
            plot_vector(ax, origin, u, color=color_vector)

            # REMOVE IF NO SEEDING
            line_bottom(ax, origin, color="black", text="A")
            line_bottom(ax, origin + u, color="black", text="B")
            # ---

            for b in basis:
                plot_vector(ax, origin, b, color=color_basis)

        # # for beam in [beam1, beam2]:
        for beam in [beam1]:
            for i, surface in enumerate(beam):
                ax.scatter(
                    surface[:,0], surface[:,1], surface[:,2],
                    color=color_surface, alpha=alpha_surface
                )

                # line_bottom(ax, surface[0], color="black")
        
        # # for list_landscape in [list_landscape1, list_landscape2]:
        for list_landscape in [list_landscape1]:
            for i, layer_landscape in enumerate(list_landscape):
                for line in layer_landscape:
                    ax.scatter(
                        line[:,0], line[:,1], line[:,2],
                        color=color_landscape, alpha=alpha_landscape
                    )

        ax.set_box_aspect((1,1,1))
        plt.tight_layout()
        
        # REMOVE IF NO SEEDING
        ax.set_xlim(-1.30679382946117, 3.3597140779077113)
        ax.set_ylim(-1.3633625025125786, 2.967558224305686)
        ax.set_zlim(-1.1020904882813967, 3.9463500196573147)
        ax.view_init(elev=15., azim=-105.)
        # ---

        plt.show()
