#include "constrained_relation_solver.h"

#undef min
#undef max

#include <ctpl_stl.h>

#include <cppoptlib/solver/lbfgsbsolver.h>

#include <matching/threshold.h>
#include <matching/geodesic_fan.h>
#include <shape_signatures/shape_signature.h>
#include <geometry/patch.h>

// Debug
#include <shape_signatures/shape_diameter_signature.h>

void signature_value_proxy(int id, int index, std::shared_ptr<ShapeSignature> sig, Eigen::VectorXd x, std::shared_ptr<Eigen::MatrixXd> out) {
	if (sig == nullptr) {
		throw std::invalid_argument("Shape signature can't be null!");
	}

	if (out == nullptr) {
		throw std::invalid_argument("Output buffer can't be null!");
	}

	double val = sig->_param_opt->value(x);

	out->row(index) << val;
}

CRSolver::CRSolver() {
}

CRSolver::~CRSolver() {
}

bool CRSolver::add_signature(std::shared_ptr<ShapeSignature> sig) {
	if (sig == nullptr) {
		return false;
	}

	_sigs.push_back(sig);

	return true;
}

std::vector<std::shared_ptr<ShapeSignature>> CRSolver::signatures() const {
	return _sigs;
}

bool CRSolver::add_relation(Relation rel, unsigned int desire_set) {
	auto set = _relations.find(desire_set);

	if (set == _relations.end()) {
		std::vector<Relation> rels = {
			rel
		};

		_relations.insert(std::make_pair(desire_set, rels));
	} else {
		set->second.push_back(rel);
	}

	return true;
}

std::vector<Relation> CRSolver::relations(unsigned int desire_set) const {
	return _relations.find(desire_set)->second;
}

std::vector<std::shared_ptr<Threshold>> CRSolver::solve(std::vector<double>& indices, unsigned int desire_set) const {
	if (_sigs.size() <= 0) {
		// There are no signatures to optimize based on the established relations
		return std::vector<std::shared_ptr<Threshold>>();
	}

	// Solve for new relation
	Eigen::VectorXd x(1);

	auto set = _relations.find(desire_set);

	if (set == _relations.end()) {
		return std::vector<std::shared_ptr<Threshold>>();
	}

	const int thread_pool_size = 10;
	ctpl::thread_pool pool(10);

	std::vector<std::shared_ptr<Eigen::MatrixXd>> buffers;

	const int granularity = 100;
	for (int i = 0; i < _sigs.size(); ++i) {
		auto sig = _sigs[i];
		auto steps = sig->_param_opt->param_steps(granularity);
		std::shared_ptr<Eigen::MatrixXd> out = std::make_shared<Eigen::MatrixXd>(granularity, 1);

		sig->_param_opt->set_value_desire_set(set->second);

		// Run _sigs[i]->_value(), find minimum, then step to best local minimum it with LbfgsbSolver
		for (int j = 0; j < steps.size(); ++j) {
			pool.push(signature_value_proxy, j, sig, steps.row(j), out);
		}

		buffers.push_back(out);
	}
	
	pool.stop(true);

	// TODO: Currently only support the first signature in the set, and first parameter of that signature
	indices.resize(_sigs.size());
	for (int i = 0; i < buffers.size(); ++i) {
		Eigen::DenseIndex minIndex;
		double min_value = buffers[i]->col(0).minCoeff(&minIndex);
		auto steps = _sigs[i]->_param_opt->param_steps(granularity);

		indices[i] = steps(minIndex);

		std::cout << "min_value == " << min_value << std::endl;
		std::cout << "minIndex == " << minIndex << std::endl;
		std::cout << "indices[i] == " << indices[i] << std::endl;

		std::dynamic_pointer_cast<ShapeDiameterSignature>(_sigs[i])->resample_at_t(indices[i]);
	}

	std::cout << "--- CRSolver ---" << std::endl;
	std::cout << "desire_set:		" << desire_set << std::endl;
	std::cout << "relations:		" << set->second.size() << std::endl;
	std::cout << "signatures:		" << _sigs.size() << std::endl << std::endl;

	for (int i = 0; i < buffers.size(); ++i) {
		auto steps = _sigs[i]->_param_opt->param_steps(granularity);
		std::cout << "buffers[" << i << "]:" << std::endl;
		for (int j = 0; j < buffers[i]->rows(); ++j) {
			std::cout << "[ index, param, obj ]: [ " << j << ", " << steps(j) << ", " << buffers[i]->row(j) << " ]" << std::endl;
		}
		std::cout << "selected param: " << indices[i] << std::endl << std::endl;
	}

	std::vector<std::shared_ptr<Threshold>> thresholds {
		 optimal_relation_threshold(indices[0], desire_set)
	};

	for (int i = 0; i < set->second.size(); ++i) {
		Eigen::DenseIndex vid = set->second[i]._patch->vid_to_origin_mesh(0);

		if (((set->second[i]._designation == Relation::Designation::Include) && !thresholds[0]->contains(_sigs[0]->get_signature_values(indices[0]).row(vid))) ||
			((set->second[i]._designation == Relation::Designation::Exclude) && thresholds[0]->contains(_sigs[0]->get_signature_values(indices[0]).row(vid)))) {
			std::clog << "VID [ " << vid << " ] is " << ((set->second[i]._designation == Relation::Designation::Include) ? "Included" : "Excluded") << " and is" << (!thresholds[0]->contains(_sigs[0]->get_signature_values(indices[0]).row(vid)) ? " NOT" : "") << " in threshold [ " << _sigs[0]->get_signature_values(indices[0]).row(vid) << " ]" << std::endl;
		}
	} 

	std::cout << std::endl;

	return thresholds;
}

std::shared_ptr<Threshold> CRSolver::optimal_relation_threshold(double optimal_t, unsigned int set) const {
	std::shared_ptr<Threshold> threshold = nullptr;

	//double optimal_t = _param_opt->solved_value(set);

	if (optimal_t < 0.0 || relations(set).size() == 0) {
		return nullptr;
	}

	if (_sigs.size() <= 0) {
		// There are no signatures to threshold!
		return threshold;
	}

	// TODO: Only a single signature is supported right now
	auto sig = _sigs[0];

	// Find the "good" include interval
	// The metric is the l2-norm of the patch's geodesic fan across its discrete exponential map
	// There are a four scenarios:
	// 1. There are no include patches, and any number of exclude patches (> 0), so partition space so that all excludes are outside the threshold interval ([a,max()] or [min(), a]
	// 2. There is only one include patch, so the similarity interval can be arbitrarily defined as a choosable sigma of a normal distribution across the differences from the include patch
	// 3. Only one include patch and any number of exclude patches, the find the closest exclude patches, and take the interval boundaries as the midpoint between those and the include
	// 4. There are many include patches, so the interval is defined by the greatest and least include patch norms

	// Count the number of include/exclude
	unsigned int included = 0;
	unsigned int excluded = 0;
	for (const Relation& r : relations(set)) {
		if (r._designation == Relation::Designation::Include) {
			included++;
		}
		else if (r._designation == Relation::Designation::Exclude) {
			excluded++;
		}
	}

	// Determine threshold based on number of include/exclude patches
	double i_min = 0.0; // _sig.minCoeff();
	double i_max = 1.0; // _sig.maxCoeff();

	Eigen::VectorXd disc_center;

	if (included == 0) {
		// 1. There are no include patches, and any number of exclude patches (> 0), so partition space by the excluded patch norms and pick the largest partition
		// For each relation, pop the interval containing its norm, split it, and insert the two resultant into storage
		// When done, find the largest interval in a single pass search
		std::vector<Threshold> intervals;
		intervals.push_back(Threshold(i_min, i_max));

		for (const Relation& r : relations(set)) {
			std::shared_ptr<Patch> patch = r._patch;

			auto fan = sig->_param_opt->geodesic_fan_from_relation(r);

			double norm = fan->scaled_l2_norm();

			for (auto interval = intervals.begin(); interval != intervals.end(); ++interval) {
				if (interval->contains((Eigen::VectorXd(1) << norm).finished())) {
					// remove from list, divide interval, insert two resultant intervals
					Threshold bisected = *interval;

					intervals.erase(interval);

					std::array<Threshold, 2> split = bisected.split(norm);

					intervals.push_back(split[0]);
					intervals.push_back(split[1]);

					break;
				}
			}
		}

		// Determine largest interval
		double interval_width = -1.0;
		std::size_t largest = -1;

		for (std::size_t i = 0; i < intervals.size(); ++i) {
			double width = intervals[i].max() - intervals[i].min();

			if (width > interval_width) {
				interval_width = width;
				largest = i;
			}
		}

		threshold = std::make_shared<Threshold>(intervals[largest]);

	}
	else if (included == 1 && excluded == 0) {
		// 2. There is only one include patch, so the similarity interval can be arbitrarily defined as a choosable sigma of a normal distribution across the differences from the include patch
		auto fan = sig->_param_opt->geodesic_fan_from_relation(relations(set)[0]);

		double norm = fan->scaled_l2_norm();

		// TODO: need signature standard deviation to make a meaningful interval
		Threshold norm_dist = Threshold::from_clipped_normal_dist(norm, 1.0, 0.5);

		i_min = std::max(i_min, norm_dist.min());
		i_max = std::min(i_max, norm_dist.max());

		if (i_min > i_max) {
			// Don't believe this can happen given the internals of Threshold, but doesn't hurt to check
			std::swap(i_min, i_max);
		}

		threshold = std::make_shared<Threshold>(i_min, i_max);

	}
	else if (included == 1 && excluded > 0) {
		// 3. Only one include patch and any number of exclude patches, the find the closest exclude patches, and take the interval boundaries as the midpoint between those and the include
		// In this case, we have a single interval that needs to be clipped for each excluded patch norm. The remaining interval is then returned.
		double i_norm = 0.0;

		for (const Relation& r : relations(set)) {
			if (r._designation == Relation::Designation::Include) {
				std::shared_ptr<Patch> patch = r._patch;

				auto fan = sig->_param_opt->geodesic_fan_from_relation(r);

				double norm = fan->scaled_l2_norm();

				break;
			}
		}

		for (const Relation& r : relations(set)) {
			std::shared_ptr<Patch> patch = r._patch;

			auto fan = sig->_param_opt->geodesic_fan_from_relation(r);

			double norm = fan->scaled_l2_norm();

			if (i_min < norm && norm < i_norm) {
				i_min = norm;
			}
			else if (i_norm < norm && norm < i_max) {
				i_max = norm;
			}
		}

		threshold = std::make_shared<Threshold>(i_min, i_max);

	}
	else if (included > 1) {
		// 4. There are many include patches, so the interval is defined by the greatest and least include patch norms
		i_min = std::numeric_limits<double>::max();
		i_max = std::numeric_limits<double>::min();

		// Find good interval
		for (const Relation& r : relations(set)) {
			if (r._designation == Relation::Designation::Exclude) {
				continue;
			}

			std::shared_ptr<Patch> patch = r._patch;

			assert(patch->vids().size() == 1);

			double norm = sig->get_signature_values().row(*patch->vids().begin()).norm();

			if (norm < i_min) {
				i_min = norm;
			}

			if (norm > i_max) {
				i_max = norm;
			}
		}

		// Make sure that no exclude patches are within the exception interval range
		exception_interval(Threshold(i_min, i_max), optimal_t, set);

		threshold = std::make_shared<Threshold>(i_min, i_max);
	}

	return std::make_shared<Threshold>(threshold->min(), threshold->max());
}

void CRSolver::exception_interval(Threshold interval, double mean, unsigned int set) const {
	if (_sigs.size() <= 0) {
		return;
	}

	// For every excluded patch, add an impulse to the exception map that removes it from the exception interval with the least possible energy
	auto sig = _sigs[0];
	sig->clear_quadratic_bumps();

	auto rels = relations(set);

	// Determine largest patch size
	unsigned int inc = 0;
	unsigned int exc = 0;
	unsigned int size = 0;
	for (const Relation& r : rels) {
		size = std::max(r._patch->vids().size(), static_cast<size_t>(size));
	}

	if (size == 1) {
		// All of the selections are single vertices -- this simplifies some of our assumptions
		std::vector<double> inc_norms;
		for (const Relation& r : rels) {
			if (r._designation == Relation::Designation::Include && r._patch != nullptr && r._patch->vids().size() > 0) {
				Eigen::DenseIndex inc_vid = *r._patch->vids().begin();

				inc_norms.push_back(sig->get_signature_values().row(inc_vid)(0));
			}
		}

		if (inc_norms.size() <= 0) {
			throw std::exception("There are no included norms?!");
		}

		// Ascending order sort
		std::sort(inc_norms.begin(), inc_norms.end());

		// Create new thresholds 
		std::vector<Threshold> inc_intervals;

		//inc_intervals.push_back(Threshold(interval.min(), *inc_norms.begin()));
		for (unsigned int i = 1; i < inc_norms.size(); ++i) {
			inc_intervals.push_back(Threshold(inc_norms[i - 1], inc_norms[i]));
		}
		//inc_intervals.push_back(Threshold(*(inc_norms.end() - 1), interval.max()));

		// For each excluded patch, check if it exists within any of the sub intervals
		std::vector<std::pair<Eigen::DenseIndex, double>> shifted_vids;

		for (const Relation& r : rels) {
			if (r._designation == Relation::Designation::Include) {
				continue;
			}

			Eigen::DenseIndex exc_vid = *r._patch->vids().begin();

			for (auto sub : inc_intervals) {
				if (sub.contains(sig->get_signature_values().row(exc_vid))) {
					// Find fair subinterval about excluded value
					std::shared_ptr<Threshold> exc_interval = sub.padded_subinterval_about(sig->get_signature_values().row(exc_vid));

					// Shift all signature values within the exc_interval outside of the interval by the width of the interval
					for (Eigen::DenseIndex i = 0; i < sig->get_signature_values().rows(); ++i) {
						if (exc_interval->contains(sig->get_signature_values().row(i))) {
							double shift = 0.0;

							if (std::fabs(exc_interval->midpoint() - interval.min()) < std::fabs(interval.max() - exc_interval->midpoint())) {
								shift = -1.0 * (std::fabs(exc_interval->midpoint() - interval.min()) + exc_interval->width());
							}
							else {
								shift = std::fabs(interval.max() - exc_interval->midpoint()) + exc_interval->width();
							}

							std::cout << "Added unlabeled vertex to shift: " << i << std::endl;

							shifted_vids.push_back(std::pair<Eigen::DenseIndex, double>(i, shift));
						}
					}
				}
			}
		}

		// Apply shifting energy 
		for (auto shifting : shifted_vids) {
			Eigen::DenseIndex vid = shifting.first;
			double energy = shifting.second;

			// Eigen::VectorXd energy_curve = /*energy * */ Eigen::VectorXd::Zero(sig->_exception_map.cols());

			double width = sig->step_width(mean);
			//double mean = solved_value(set);

			// TODO: Need to check that the _t_steps are linear, and not log for this to be valid
			/*for (Eigen::DenseIndex i = 0; i < sig->sig_steps().size(); ++i) {
				double dist = std::fabs(mean - sig->sig_steps()(i));

				if (dist < 1e-7) {
					continue;
				}

				if (dist < width) {
					width = dist;
				}
			}*/

			QuadraticBump<double> bump(vid, mean, width, energy);

			/*for (Eigen::DenseIndex i = 0; i < energy_curve.cols(); ++i) {
				energy_curve(i) = bump.energy_shift_by_parameter(sig->sig_steps()(i));
			}*/

			// For each vid, add the energy to its row in the exception map
			//sig->_exception_map.row(vid) = sig->_exception_map.row(vid) + energy_curve.transpose();
			sig->apply_quadratic_bump(bump);

			if (interval.contains(sig->get_signature_values().row(vid))) {
				char metrics[4096];
				sprintf(metrics, "interval [ %f, %f ], sig %f, shift %f", interval.min(), interval.max(), sig->get_signature_values().row(vid), energy);
				std::cout << metrics << std::endl;
				//std::cout << energy_curve.transpose() << std::endl;
				//throw std::exception("Energy did not shift remove exclusion patch?!");
				std::cout << "Energy did not shift remove exclusion patch?!" << std::endl;
			}
		}

	} /*else {
		// Else, use old algorithm for shifting just the patch
		double margin = 0.0001; // TODO: Change to a value that scales with the underlying signature values
		for (const Relation& r : rels) {
			if (r._designation == Relation::Designation::Include) {
				continue;
			}

			std::shared_ptr<Patch> patch = r._patch;

			auto fan = geodesic_fan_from_relation(r);

			double norm = fan->scaled_l2_norm();

			if (interval.contains((Eigen::VectorXd(1) << norm).finished())) {
				double energy;

				if (std::fabs(norm - interval.min()) < std::fabs(interval.max() - norm)) {
					// TODO: tossing epsilon in there might not actually change the number, so do something else to separate the excluded patch from the exception interval boundary
					energy = interval.min() - norm - margin;
				} else {
					energy = interval.max() - norm + margin;
				}

				Eigen::VectorXd energy_curve = energy * Eigen::VectorXd::Zero(sig->_exception_map.cols());

				double width = std::fabs(sig->t_upper_bound() - sig->t_lower_bound());

				for (Eigen::DenseIndex i = 0; i < energy_curve.cols(); ++i) {
					double curve_t = sig->_t_steps(i) - solved_value();
					double falloff = sig->falloff_quadratic(curve_t, width);

					// If curve_t is zero, then the falloff should be 1.0
					assert((std::fabs(curve_t) < std::numeric_limits<double>::epsilon() && std::fabs(falloff - 1.0) < std::numeric_limits<double>::epsilon()) ||
						std::fabs(curve_t) > std::numeric_limits<double>::epsilon());

					char metrics[4096];
					sprintf(metrics, "sig->_t_steps(i) %f, solved_value() %f, curve_t %f, falloff %f\n", sig->_t_steps(i), solved_value(), curve_t, falloff);
					std::cout << metrics << std::endl;

					energy_curve(i) = energy * falloff;
				}

				for (auto vid : patch->vids()) {
					// For each vid, add the energy to its row in the exception map
					// TODO: Should the addition of the energy be restricted to some column associated with the optimized t value?? Would this mean GeodesicFan needs to handle its l2-norm differently??
					sig->_exception_map.row(vid) = sig->_exception_map.row(vid) + energy_curve.transpose();
				}

				auto fan_check = geodesic_fan_from_relation(r);

				if (interval.contains((Eigen::VectorXd(1) << fan_check->scaled_l2_norm()).finished())) {
					char metrics[4096];
					sprintf(metrics, "inc %i, exc %i, interval [ %f, %f ], norm %f, energy %f", inc, exc, interval.min(), interval.max(), norm, energy);
					std::cout << metrics << std::endl;
					std::cout << energy_curve << std::endl;
					throw std::exception("Energy did not shift remove exclusion patch?!");
				}
			}
		}
	}*/
}