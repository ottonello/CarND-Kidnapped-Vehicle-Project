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

	default_random_engine generator;
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
	default_random_engine generator;
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

	// For each particle
    for (auto &particle: particles)
    {
        double particle_x = particle.x;
        double particle_y = particle.y;
        double pt = particle.theta;

        // initialize weight to 1
        double weight = 1;

        // Iterate over observed points
        for (auto &observation: observations)
        {
            double x = observation.x;
            double y = observation.y;

            // Transform observation from vehicle to map coordinates
			double mapx = particle_x + x * cos(pt) - y * sin(pt);
			double mapy = particle_y + x * sin(pt) + y * cos(pt);

			// Find closest landmark to observation
			double min_dist = std::numeric_limits<double>::max();
			Map::single_landmark_s closest_landmark;
			for(auto &landmark: map_landmarks.landmark_list){
				double dist_x = mapx - landmark.x_f;
				double dist_y = mapy - landmark.y_f;
				double dist = dist_x * dist_x + dist_y * dist_y;
				if(dist < min_dist){
					min_dist = dist;
					closest_landmark = landmark;
				}
			};

            // Calculate weight
			double dx = (closest_landmark.x_f - particle_x) - (mapx - particle_x);
			double dy = (closest_landmark.y_f - particle_y) - (mapy - particle_y);
			double sigma_x = std_landmark[0];
			double sigma_y = std_landmark[1];
			double exponent = -1.0 *((dx*dx)/(2.0 * sigma_x*sigma_x) + (dy*dy)/(2.0 * sigma_y*sigma_y));
			double num = exp(exponent);
			double denom = 2 * M_PI * sigma_x * sigma_y;
			weight *= num/denom;
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

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
