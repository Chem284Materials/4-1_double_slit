import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def update_animation(frame):
    # Draw the particle's probability distribution
    wavefunction = wavefunction_gpu.get()
    re_part = wavefunction[..., 0]
    im_part = wavefunction[..., 1]
    prob = re_part * re_part + im_part * im_part
    scaled = np.clip(prob * 255.0 * 50.0, 0, 255).astype(np.uint8)
    image_data = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image_data[..., 0] = scaled

    # Draw the potential energy surface
    for y in range(image_height):
        for x in range(image_width):
            if potential[y][x] > 0.1:
                image_data[y][x][...] = 128

    im.set_data(image_data)
    return [im]

# Set a constant on the GPU
def set_constant(name, value):
    ptr, size = cuda_module.get_global(name)
    pycuda.driver.memcpy_htod(ptr, np.float64(value))

cuda_module = SourceModule(no_extern_c=True, source="""
#include <stdint.h>

__constant__ double dx;
__constant__ double dt;

extern "C" {

__global__ void initialize(double *data, int image_width, int image_height,
                           double sigma, double center_x, double center_y,
                           double momentum_x, double momentum_y) {
  int x = threadIdx.x + ( blockIdx.x * blockDim.x );
  int y = threadIdx.y + ( blockIdx.y * blockDim.y );
  if ( x >= image_width || y >= image_height ) return;

  int index = ( ( y * image_width ) + x ) * 2;

  double px = x * dx;
  double py = y * dx;

  // Create a normalized 2D Gaussian wave packet
  double dr2 = (center_x - px) * (center_x - px)
             + (center_y - py) * (center_y - py);
  double norm = sqrt(2.0 * sigma / M_PI);
  double envelope = norm * exp(-sigma * dr2);

  // Give the packet momentum with a factor of exp(i (kx*x + ky*y))
  double phase = momentum_x * px + momentum_y * py;
  data[index + 0] = envelope * cos(phase);
  data[index + 1] = envelope * sin(phase);
}

}
""")
initialize  = cuda_module.get_function("initialize")

# Parameters defining the animation generation
image_width = 512
image_height = 512
num_frames = 150
steps_per_frame = 30  # To save time and disk space, we won't update the .gif every iteration

# Parameters defining the initial state of the wavefunction
sigma = 0.2
center_x = 15    # center of the gaussian along the x dimension
center_y = 25.6  # center of the gaussian along the y dimension
momentum_x = 8.0 # momentum along the x dimension
momentum_y = 0.0 # momentum along the y dimension

# Parameters defining the space/time integration
dx = 0.1;   # spacing between grid points
dt = 0.001; # time step interval
set_constant("dx", dx) # set dx on the GPU
set_constant("dt", dt) # set dt on the GPU

# Parameters defining the wall (units are in pixels)
wall_thickness = 10
wall_magnitude = 1000.0
slit_width = 10
slit_displacement = 10

# Note that the wavefunction is complex
# There are various libraries that can assist with working with complex values
# In this case, we are just doubling the size of our array
wavefunction = np.zeros((image_height, image_width, 2), dtype=np.double)
wavefunction_gpu = gpuarray.to_gpu(wavefunction)

# Create the potential energy surface
potential = np.zeros((image_height, image_width), dtype=np.double)
half_width = image_width//2
half_height = image_height//2
for ix in range(half_width, half_width + wall_thickness):
    for iy in range(image_height):
        if ( iy > half_height + slit_displacement and iy < half_height + slit_displacement + slit_width ):
            potential[iy][ix] = 0.0
        elif ( iy > half_height - slit_displacement - slit_width and iy < half_height - slit_displacement ):
            potential[iy][ix] = 0.0
        else:
            potential[iy][ix] = wall_magnitude
potential_gpu = gpuarray.to_gpu(potential)

blocksize = 32
ngridx = math.ceil(image_width / blocksize)
ngridy = math.ceil(image_height / blocksize)
initialize(wavefunction_gpu,
           np.int32(image_width),
           np.int32(image_height),
           np.float64(sigma),
           np.float64(center_x),
           np.float64(center_y),
           np.float64(momentum_x),
           np.float64(momentum_y),
           block=(blocksize, blocksize, 1),
           grid=(ngridx, ngridy, 1))

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((image_height, image_width, 3), dtype=np.uint8))
ani = FuncAnimation(fig, update_animation, frames=num_frames)
ani.save("animation.gif", fps=15, dpi=200)
