#include "self_similarity_map.h"

#include "matching/geodesic_fan.h"
#include "matching/threshold.h"

#include <shape_signatures/shape_signature.h>

SelfSimilarityMap::SelfSimilarityMap(const CRSolver& crsolver, std::shared_ptr<ShapeSignature> metric, unsigned int active_set) {
	// Get threshold from signature relations
	std::vector<double> indices;
	auto thresholds = crsolver.solve(indices, active_set);

	std::shared_ptr<Threshold> threshold = nullptr;
	if (thresholds.size() > 0) {
		threshold = thresholds[0];
	}

	if (threshold == nullptr) {
		throw std::logic_error("SelfSimilarityMap::SelfSimilarityMap(): There is no threshold!!");
	}

	std::cout << "Similarity Threshold: ( " << threshold->min() << ", " << threshold->max() << " )" << std::endl;

	// Create the geodesic fan for all the nodes based off of the mean patch extent
	std::shared_ptr<Mesh> mesh = metric->origin_mesh();

	_similarity_ratings = Eigen::VectorXd::Zero(mesh->vertices().rows());

	for (unsigned int i = 0; i < mesh->vertices().rows(); ++i) {
		/*if (threshold->contains(metric->get_signature_values(indices[0]).row(i))) {
			_similarity_ratings(i) = 0.0;
		} else {*/
			_similarity_ratings(i) = threshold->distance(metric->get_signature_values(indices[0]).row(i));
		//}
	}

	// Normalize ratings and invert range
	double maxCoeff = _similarity_ratings.maxCoeff();
	if (maxCoeff > 1.0) {
		_similarity_ratings /= maxCoeff;
	}
	_similarity_ratings = 1.0 - _similarity_ratings.array();

}

const Eigen::VectorXd& SelfSimilarityMap::similarity_ratings() {
	return _similarity_ratings;
}