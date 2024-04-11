import numpy as np
from scipy.fft import fftn, fftshift, ifftshift
from tqdm import tqdm
import scipy.io
import os


def simulate_diffusion(simsize, bmax, bigDelta, smallDelta, gmax, gamma, FA, Dxx_base, fiber_fractions, angles):
    qmax = gamma / (2 * np.pi) * smallDelta * gmax
    qDelta = qmax * 2 / (simsize - 1)
    rFOV = 1 / qDelta
    rmax = rFOV / 2

    # Create grid for R-space
    grid = np.linspace(-rmax, rmax, simsize)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')

    # Coefficients for R-space normalization
    pi_coeff = (4 * np.pi * bigDelta) ** 3

    # Diffusion simulation
    rspace_total = np.zeros_like(X)
    for fraction, angle_degrees, Dxx in zip(fiber_fractions, angles, Dxx_base):
        angle = np.radians(angle_degrees)
        # Calculate rotation
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        Xr = X * cos_angle + Y * sin_angle
        Yr = Y * cos_angle - X * sin_angle

        # Diffusion coefficients, updated to vary per fiber
        Dyy = Dxx * (1 - FA) / np.sqrt(2)
        Dzz = Dxx * (1 - FA) / np.sqrt(2)

        # R-space distribution for the current fiber
        rspace = 1 / np.sqrt(pi_coeff * Dxx * Dyy * Dzz) * np.exp(
            -(Xr ** 2 / (4 * Dxx * bigDelta) + Yr ** 2 / (4 * Dyy * bigDelta) + Z ** 2 / (4 * Dzz * bigDelta)))
        rspace_total += fraction * rspace

    # Fourier transform to Q-space (Signal)
    qspace = fftshift(fftn(ifftshift(rspace_total)))

    return rspace_total, qspace


def generate_samples(data_dir, n_samples, simsize, bmax, bigDelta, smallDelta, gmax, gamma, Dxx_base):
    for i in tqdm(range(n_samples)):
        # Generate random or predetermined parameters for each sample
        FA = np.random.uniform(0.01, 0.99)
        fiber_fractions = np.random.dirichlet(np.ones(3))  # For three fibers
        angles = np.random.uniform(0, 180, size=3)  # Angles between 0 and 180 degrees
        Dxx_individual = np.random.uniform(Dxx_base * 0.8, Dxx_base * 1.2, size=3)  # Vary Dxx slightly for each fiber

        # Simulate diffusion and compute spaces
        rspace, qspace = simulate_diffusion(simsize, bmax, bigDelta, smallDelta, gmax, gamma, FA, Dxx_individual,
                                            fiber_fractions, angles)

        # Save data in individual folders
        sample_dir = f'{data_dir}/sample_{i + 1}'
        os.makedirs(sample_dir, exist_ok=True)
        scipy.io.savemat(os.path.join(sample_dir, 'rspace.mat'), {'rspace': rspace})
        scipy.io.savemat(os.path.join(sample_dir, 'qspace.mat'), {'qspace': qspace})

        with open(os.path.join(sample_dir, 'parameters.txt'), 'w') as f:
            f.write(f'simsize: {simsize}\n')
            f.write(f'FA: {FA}\n')
            f.write(f'Fiber fractions: {fiber_fractions}\n')
            f.write(f'Angles: {angles}\n')
            f.write(f'Dxx_individual: {Dxx_individual}\n')
            f.write(f'bigDelta: {bigDelta}\n')
            f.write(f'smallDelta: {smallDelta}\n')
            f.write(f'gmax: {gmax}\n')
            f.write(f'gamma: {gamma}\n')
            f.write(f'bmax: {bmax}\n')


def data_loader(data_dir):
    data = []
    for sample_dir in os.listdir(data_dir):
        rspace = scipy.io.loadmat(os.path.join(data_dir, sample_dir, 'rspace.mat'))['rspace']
        qspace = scipy.io.loadmat(os.path.join(data_dir, sample_dir, 'qspace.mat'))['qspace']

        data.append((qspace, rspace))
    return data
