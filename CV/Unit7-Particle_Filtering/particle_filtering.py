import numpy as np
from tracking_utils import draw_box
from tqdm import tqdm
from scipy.stats import truncnorm
import cv2
from math import ceil


class ParticleFiltering:
    def __init__(self, image_shape, template_window, window_size_x, window_size_y, num_particles=1000,
                 mse_variance=30, dynamics_variance=3, weight_regularization=0.1, template_update_weight=0,
                 debug=False):
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.template = template_window
        self.mse_variance = mse_variance

        self.particles = np.random.randint(low=(0, 0), high=image_shape, size=(num_particles, 2))
        # self.particles[-1] = np.array([79, 124])
        self.particles[-1] = np.array([145, 191])
        self.particles[-2] = np.array([191, 145])
        self.particles_weights = np.ones(num_particles, dtype=float) / num_particles
        self.shape = image_shape
        self.dynamics_var = dynamics_variance
        self.weight_regularization = weight_regularization
        self.debug = debug
        self.template_w = template_update_weight

    def run(self, video):
        output_video = np.zeros_like(video)
        points_scat = []
        templates = []
        for i, frame in tqdm(enumerate(video), total=video.shape[0]):
            self.run_iteration(frame)
            best_particle = self.particles[np.argmax(self.particles_weights)]
            print(f"best scale is {best_particle[-1]}")
            scale = 1
            if len(best_particle) == 3:
                scale = best_particle[-1]
            self.update_template(
                extract_window_with_padding(frame, best_particle.astype(int), self.window_size_x, self.window_size_y))
            output_video[i] = draw_box(frame, *best_particle[:2].astype(int), self.window_size_x * scale,
                                       self.window_size_y * scale)
            points_scat.append((self.particles, self.particles_weights))
            templates.append(self.template)
        if self.debug:
            return output_video, points_scat, templates
        return output_video

    def run_iteration(self, frame):
        # sample
        points_indices = self.sample_particles()
        particles, particles_weights = [], []
        sampled_particles = self.dynamics2(np.repeat(self.particles, points_indices, axis=0))
        # regularize by sampling random points
        random_particles = self.regularize_particles(len(sampled_particles))
        particles = np.concatenate((sampled_particles, random_particles))

        weights_calculated = {}
        for particle in particles:
            hashed_particle = tuple(particle)
            if hashed_particle in weights_calculated:
                particles_weights.append(weights_calculated[hashed_particle])
            else:
                weight = self.importance_weight(frame, particle)
                particles_weights.append(weight)
                weights_calculated[hashed_particle] = weight
        self.particles = np.array(particles)
        weights = np.array(particles_weights, dtype=float)
        self.particles_weights = weights / np.sum(weights)

    def update_template(self, best_window):
        self.template = (1 - self.template_w) * self.template + self.template_w * best_window

    def sample_particles(self):
        return np.random.multinomial(int(len(self.particles_weights) * (1 - self.weight_regularization)),
                                     self.particles_weights)

    def regularize_particles(self, num_points):
        num_points = len(self.particles_weights) - num_points
        return np.random.randint(low=(0, 0), high=self.shape, size=(num_points, 2))

    def dynamics(self, point):
        image_range_x = self.shape[0] - 1
        image_range_y = self.shape[1] - 1
        x_noise = truncnorm(a=-image_range_x / self.dynamics_var, b=+image_range_x / self.dynamics_var,
                            scale=self.dynamics_var).rvs().round().astype(int)
        y_noise = truncnorm(a=-image_range_y / self.dynamics_var, b=+image_range_y / self.dynamics_var,
                            scale=self.dynamics_var).rvs().round().astype(int)
        point += np.array([x_noise, y_noise])
        point[0] = min(max(0, point[0]), image_range_x)
        point[1] = min(max(0, point[1]), image_range_y)
        return point

    def dynamics2(self, points):
        image_range_x = self.shape[0] - 1
        image_range_y = self.shape[1] - 1
        x_noises = truncnorm(a=-image_range_x / self.dynamics_var, b=+image_range_x / self.dynamics_var,
                             scale=self.dynamics_var).rvs(size=len(points)).round().astype(int)
        y_noises = truncnorm(a=-image_range_y / self.dynamics_var, b=+image_range_y / self.dynamics_var,
                             scale=self.dynamics_var).rvs(size=len(points)).round().astype(int)
        points += np.stack((x_noises, y_noises), axis=1)
        return np.clip(points, np.array([0, 0]), np.array([image_range_x, image_range_y]))

    def importance_weight(self, frame, particle):
        window = extract_window_with_padding(frame, particle.astype(int), self.window_size_x, self.window_size_y)
        return calculate_mse(window, self.template, self.mse_variance)


class MeanShiftPF(ParticleFiltering):
    def __init__(self, image_shape, template_window, window_size_x, window_size_y, bins=8, **kwargs):
        super().__init__(image_shape, template_window, window_size_x, window_size_y, **kwargs)
        self.bins = bins

    def importance_weight(self, frame, particle):
        window = extract_window_with_padding(frame, particle.astype(int), self.window_size_x, self.window_size_y)
        hist = window_to_hist(window, self.bins)
        return 1 / (1 + calculate_chi_squared(hist, window_to_hist(self.template, self.bins)))


class PFWindowsSize(MeanShiftPF):
    """Particles now should also include scale"""

    def __init__(self, image_shape, template_window, window_size_x, window_size_y, **kwargs):
        super().__init__(image_shape, template_window, window_size_x, window_size_y, **kwargs)
        # self.particles = np.random.randint(low=(0, 0), high=image_shape, size=(num_particles, 2))
        window_scales = np.ones_like(self.particles_weights)
        self.particles = np.hstack((self.particles, window_scales.reshape(-1, 1)))

    def dynamics2(self, points):
        points_dynamics = super().dynamics2(points[:, :-1])
        windows_sizes_dynamics = points[:, -1] * np.random.uniform(0.97, 1, size=len(points))
        return np.hstack((points_dynamics, windows_sizes_dynamics.reshape(-1, 1)))

    def importance_weight(self, frame, particle):
        x_scaled = ceil(self.window_size_x * particle[-1])
        y_scaled = ceil(self.window_size_y * particle[-1])
        window = extract_window_with_padding(frame, particle[:-1].astype(int), x_scaled, y_scaled)
        hist = window_to_hist(window, self.bins)
        hist_template = window_to_hist(
            cv2.resize(self.template, dsize=(x_scaled, y_scaled), interpolation=cv2.INTER_CUBIC), self.bins)
        return 1 / (1 + calculate_chi_squared(hist, hist_template))

    def regularize_particles(self, num_points):
        num_points = len(self.particles_weights) - num_points
        return np.random.randint(low=(0, 0), high=self.shape, size=(num_points, 3))


def calculate_chi_squared(hist1, hist2):
    return 0.5 * np.sum(np.where((hist1 + hist2) == 0, 0, (hist1 - hist2) ** 2 / (hist1 + hist2)))


def window_to_hist(window: np.ndarray, bins: int) -> np.ndarray:
    histogram_r = np.histogram(window[:, :, 0], bins=bins, range=(0, 255))[0]
    histogram_g = np.histogram(window[:, :, 1], bins=bins, range=(0, 255))[0]
    histogram_b = np.histogram(window[:, :, 2], bins=bins, range=(0, 255))[0]

    histogram = np.concatenate((histogram_r, histogram_g, histogram_b))

    return histogram


def calculate_mse(window, template, variance):
    mse = np.sqrt(np.sum((window - template) ** 2))
    return np.exp(-mse / variance / 2)


def extract_window_with_padding(frame, point, window_size_x, window_size_y):
    pad_x = window_size_x // 2
    pad_y = window_size_y // 2

    padded_frame = np.pad(frame, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode='constant', constant_values=0)

    # Adjust the point coordinates for the padded image
    padded_point_x = point[0] + pad_x
    padded_point_y = point[1] + pad_y

    window = padded_frame[padded_point_x - window_size_x // 2:padded_point_x + (window_size_x + 1) // 2,
             padded_point_y - window_size_y // 2:padded_point_y + (window_size_y + 1) // 2]

    return window
