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

	for (int i =0;i<num_particles;i++){
		Particle particle = {
				i,
				dist_x(generator),
				dist_y(generator),
				dist_theta(generator),
				1
		};
        particles.push_back(particle);

        float w = 1;
        weights.push_back(w);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	double yaw_d = delta_t * yaw_rate;
	for(auto &particle: particles) {
		double x = particle.x + velocity / yaw_rate *(sin(particle.theta + yaw_d) - sin(particle.theta));
		double y = particle.y + velocity / yaw_rate *(cos(particle.theta) - cos(particle.theta + yaw_d));
		double theta = particle.theta + yaw_d;

		std::normal_distribution<double> dist_x(x, std_x);
		std::normal_distribution<double> dist_y(y, std_y);
		std::normal_distribution<double> dist_theta(theta, std_theta);

		particle.x =dist_x(generator);
		particle.y =dist_y(generator);
		particle.theta =dist_theta(generator);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    for (auto &particle: particles)
    {
        double px = particle.x;
        double py = particle.y;
        double pt = particle.theta;

        double weight = 1;

        //observations to map frame
        for (int o = 0; o < observations.size(); ++o)
        {
            double x = observations[o].x;
            double y = observations[o].y;

            // Transform observation from vehicle to map coordinates
			double mapx = px + x * cos(pt) - y * sin(pt);
			double mapy = py + x * sin(pt) + y * cos(pt);

			// Find closest landmark
            double min_dist = LONG_MAX;
            int min_dist_id = 0;
            for (int l = 0; l < map_landmarks.landmark_list.size(); ++l)
            {
                double dx = (mapx - map_landmarks.landmark_list[l].x_f);
                double dy = (mapy - map_landmarks.landmark_list[l].y_f);
                double dist = dx * dx + dy * dy;
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_dist_id = l;
                }
            }

            // Calculate weight
            double dmx = map_landmarks.landmark_list[min_dist_id].x_f - px;
            double dmy = map_landmarks.landmark_list[min_dist_id].y_f - py;
            double dox = mapx - px;
            double doy = mapy - py;

            double dist_m = sqrt(dmx * dmx + dmy * dmy);
            double dist_o = sqrt(dox * dox + doy * doy);

            double ang_m = atan2(dmy, dmx);
            double ang_o = atan2(doy, dox);

            // Bivariate Gaussian
            double num_a = (dist_m - dist_o) * (dist_m - dist_o)
                    / (2.0 * std_landmark[0] * std_landmark[0]);
            double num_b = (ang_m - ang_o) * (ang_m - ang_o)
                    / (2.0 * std_landmark[1] * std_landmark[1]);
            double numerator = exp(-1.0 * (num_a + num_b));
            double denominator = 2.0 * M_PI * std_landmark[0] * std_landmark[1];

            weight *= numerator / denominator;
        }
        particle.weight = weight;
        weights[particle.id] = weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    // Normalize weights
    double total_weight = 0.0;
    for (int i = 0; i < particles.size(); i++)
    {
        total_weight += particles[i].weight;
    }
    for (int i = 0; i < particles.size(); i++)
    {
        weights[i] /= total_weight;
        particles[i].weight /= total_weight;
    }

    // Max of these weights
    double max_weight = *std::max_element(weights.begin(), weights.end());

    // Resample
    double beta = 0.0;
    int particles_size = particles.size();

    std::default_random_engine gen1;
    std::uniform_int_distribution<> uniform_int(0, particles_size);
    int index = uniform_int(gen1);

    std::vector<Particle> new_particles;
    std::vector<double> new_weights;

    std::default_random_engine gen2;
    std::uniform_real_distribution<> uniform_double(0.0, 2.0 * max_weight);

    for (int i = 0; i < num_particles; i++)
    {
        beta += uniform_double(gen2);
        while (weights[index] < beta)
        {
            beta -= weights[index];
            index = (index + 1) % particles_size;
        }
        Particle p = {i, particles[index].x, particles[index].y, particles[index].theta, 1.0 };
        new_particles.push_back(p);
        new_weights.push_back(1.0);
    }
    particles = new_particles;
    weights = new_weights;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
