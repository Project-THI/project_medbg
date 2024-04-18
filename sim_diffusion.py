import itertools

import numpy as np
from scipy.fft import fftn, fftshift, ifftshift
from tqdm import tqdm
import scipy.io
import os
import mat73


def simulate_diffusion(simsize, bmax, bigDelta, smallDelta, gmax, gamma, FA, Dxx_base, fiber_fractions, angle_xy,
                       angle_xz):
    qmax = gamma / (2 * np.pi) * smallDelta * gmax
    qDelta = qmax * 2 / (simsize - 1)
    rFOV = 1 / qDelta
    rmax = rFOV / 2

    # Create grid for R-space
    grid = np.linspace(-rmax, rmax, simsize)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')

    # Coefficients for R-space normalization
    pi_coeff = (4 * np.pi * bigDelta) ** 3

    rspaces = []

    # Diffusion simulation
    rspace_total = np.zeros_like(X)

    for i in range(len(fiber_fractions)):
        # print(len(fiber_fractions))
        fr = fiber_fractions[i]
        a_xy = angle_xy[i]
        a_xz = angle_xz[i]
        Dxx = Dxx_base

        # print(f'Fraction: {fr}, Angle XY: {a_xy}, Angle YZ: {a_yz}, Dxx: {Dxx}')

        # Calculate fiber rotation around y-z plane:
        # angle = np.radians(a_yz)
        # cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        # Yr = Y * cos_angle + Z * sin_angle
        # Zr = Z * cos_angle - Y * sin_angle

        # Calculate fiber rotation around x-y plane:
        angle = np.radians(a_xy)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        Xr = X * cos_angle + Y * sin_angle
        Yr = Y * cos_angle - X * sin_angle

        # Calculate fiber rotation around x-z plane:
        angle = np.radians(a_xz)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        Xr = Xr * cos_angle + Z * sin_angle
        Zr = Z * cos_angle - Xr * sin_angle

        print(f'Fraction: {fr}, Number: {i}, Angle XY: {a_xy}, Angle XZ: {a_xz}, Dxx: {Dxx}')

        # Diffusion coefficients, updated to vary per fiber
        Dyy = Dxx * (1 - FA) / np.sqrt(2)
        Dzz = Dxx * (1 - FA) / np.sqrt(2)

        # R-space distribution for the current fiber
        rspace = 1 / np.sqrt(pi_coeff * Dxx * Dyy * Dzz) * np.exp(
            -(Xr ** 2 / (4 * Dxx * bigDelta) + Yr ** 2 / (4 * Dyy * bigDelta) + Zr ** 2 / (4 * Dzz * bigDelta)))

        rspace_total += fr * rspace
        rspaces.append(fr * rspace)

    # Fourier transform to Q-space (Signal)
    qspace = fftshift(fftn(ifftshift(rspace_total)))

    # Work with the magnitude of the qspace, disregard the phase information
    qspace = np.abs(qspace)

    return rspace_total, qspace, rspaces


def generate_samples(data_dir, n_samples, simsize, bmax, bigDelta, smallDelta, gmax, gamma, Dxx_base):
    for i in tqdm(range(n_samples)):
        # Generate random or predetermined parameters for each sample
        FA = np.random.uniform(0.01, 0.99)
        num_fibers = np.random.randint(1, 3)
        fiber_fractions = np.random.dirichlet(np.ones(num_fibers))  # For three fibers
        angle_xy = np.random.uniform(0, 179, size=num_fibers)  # Angles between 0 and 180 degrees
        angle_yz = np.random.uniform(0, 179, size=num_fibers)  # Angles between 0 and 180 degrees
        Dxx_individual = np.random.uniform(Dxx_base * 0.8, Dxx_base * 1.2,
                                           size=num_fibers)  # Vary Dxx slightly for each fiber

        # Simulate diffusion and compute spaces
        rspace, qspace, _ = simulate_diffusion(simsize, bmax, bigDelta, smallDelta, gmax, gamma, FA, Dxx_individual,
                                               fiber_fractions, angle_xy, angle_yz)

        print(rspace.shape)
        print(qspace.shape)

        # Save data in individual folders
        sample_dir = f'{data_dir}/sample_{i + 1}'
        os.makedirs(sample_dir, exist_ok=True)
        scipy.io.savemat(os.path.join(sample_dir, 'rspace.mat'), {'rspace': rspace})
        scipy.io.savemat(os.path.join(sample_dir, 'qspace.mat'), {'qspace': qspace})

        with open(os.path.join(sample_dir, 'parameters.txt'), 'w') as f:
            f.write(f'simsize: {simsize}\n')
            f.write(f'FA: {FA}\n')
            f.write(f'Fiber fractions: {fiber_fractions}\n')
            f.write(f'Angle_xy: {angle_xy}\n')
            f.write(f'Angle_yz: {angle_yz}\n')
            f.write(f'Dxx_individual: {Dxx_individual}\n')
            f.write(f'bigDelta: {bigDelta}\n')
            f.write(f'smallDelta: {smallDelta}\n')
            f.write(f'gmax: {gmax}\n')
            f.write(f'gamma: {gamma}\n')
            f.write(f'bmax: {bmax}\n')


def prep_mat_sim(data_dir):
    sims = mat73.loadmat(data_dir)['results']
    loader = []

    for sim in tqdm(range(len(sims)), desc='Loading q- and r-space data'):
        rspace, qspace = sims[sim][0]['rspace'], sims[sim][0]['qspace']
        info = {'fiber_fractions': sims[sim][0]['fiber_fractions'],
                'num_fibers': sims[sim][0]['num_fibers'],
                }
        # TODO Might want to add more info here if it doesn't get too big
        loader.append((qspace, rspace, info))

    return loader
