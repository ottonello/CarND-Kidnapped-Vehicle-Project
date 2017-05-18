/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
// TODO review number of parts
	num_particles = 100;

	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle particle = { i, dist_x(generator), dist_y(generator),
				dist_theta(generator), 1 };
		particles.push_back(particle);

		float w = 1;
		weights.push_back(w);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	double yaw_d = delta_t * yaw_rate;
	for (auto &particle : particles) {
		double x = particle.x
				+ velocity / yaw_rate
						* (sin(particle.theta + yaw_d) - sin(particle.theta));
		double y = particle.y
				+ velocity / yaw_rate
						* (cos(particle.theta) - cos(particle.theta + yaw_d));
		double theta = particle.theta + yaw_d;

		std::normal_distribution<double> dist_x(x, std_x);
		std::normal_distribution<double> dist_y(y, std_y);
		std::normal_distribution<double> dist_theta(theta, std_theta);

		particle.x = dist_x(generator);
		particle.y = dist_y(generator);
		particle.theta = dist_theta(generator);
	}

}

Map::single_landmark_s ParticleFilter::get_closest(const Map& map_landmarks,
		double mapx, double mapy) {
	double min_dist = std::numeric_limits<double>::max();
	Map::single_landmark_s closest_landmark;
	for (auto& landmark : map_landmarks.landmark_list) {
		double dist_x = mapx - landmark.x_f;
		double dist_y = mapy - landmark.y_f;
		double dist = dist_x * dist_x + dist_y * dist_y;
		if (dist < min_dist) {
			min_dist = dist;
			closest_landmark = landmark;
		}
	};
	return closest_landmark;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a multi-variate Gaussian distribution.
	for (auto &particle : particles) {
		double particle_x = particle.x;
		double particle_y = particle.y;
		double pt = particle.theta;

		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];

		// initialize weight
		double weight = 1;

		// Iterate over observations
		for (auto &observation : observations) {
			double x = observation.x;
			double y = observation.y;

			// Transform observation from vehicle to map coordinates.
			//   The following is a good resource for the theory:
			//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
			//   and the following is a good resource for the actual equation to implement:
			//   http://planning.cs.uiuc.edu/node99.html
			double mapx = particle_x + x * cos(pt) - y * sin(pt);
			double mapy = particle_y + x * sin(pt) + y * cos(pt);

			// Find closest landmark to observation
			Map::single_landmark_s closest_landmark = get_closest(map_landmarks,
					mapx, mapy);
			// Calculate weight using multivariate gaussian distribution
			double dx = (closest_landmark.x_f - particle_x)
					- (mapx - particle_x);
			double dy = (closest_landmark.y_f - particle_y)
					- (mapy - particle_y);
			double exponent = -1.0
					* ((dx * dx) / (2.0 * sigma_x * sigma_x)
							+ (dy * dy) / (2.0 * sigma_y * sigma_y));
			double num = exp(exponent);
			double denom = 2 * M_PI * sigma_x * sigma_y;
			weight *= num / denom;
		}
		particle.weight = weight;
		weights[particle.id] = weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight,
	// using the resampling wheel algorithm.
	int n = weights.size();
	double max_weight = *std::max_element(begin(weights), end(weights));
	uniform_real_distribution<> rand_beta(0.0, 2.0 * max_weight);

	// Samples are stored into new vectors
	vector<double> new_weights;
	vector<Particle> new_parts;

	// Initial values for the algorithm
	// Randomly pick an initial point in the wheel
	uniform_int_distribution<int> wheel_initial_distribution(0, n - 1);
	int index = wheel_initial_distribution(generator);
	assert(index < n);
	double beta = 0;

	// Sample new 'n' values
	for (int i = 0; i < n; i++) {
		// Skip ahead by a value between 0 and 2 * max_weight
		beta = beta + rand_beta(generator);

		// Decrease beta by current weight, increase current index until
		// current weight is smaller than beta, then pick that weight/particle
		while (weights[index] < beta) {
			beta = beta - weights[index];
			index = (index + 1) % n;
		}
		Particle new_part = { i, particles[index].x, particles[index].y,
				particles[index].theta, weights[index] };
		new_weights.push_back(weights[index]);
		new_parts.push_back(new_part);
	}
	weights = new_weights;
	particles = new_parts;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " "
				<< particles[i].theta << "\n";
	}
	dataFile.close();
}
