#ifndef SELF_SIMILARITY_MAP_H_
#define SELF_SIMILARITY_MAP_H_

#include <memory>

#include <Eigen/Dense>

#include <geometry/mesh.h>
#include <geometry/patch.h>
#include <shape_signatures/shape_signature.h>
#include <matching/constrained_relation_solver.h>

class SelfSimilarityMap {
	public:
		SelfSimilarityMap(const CRSolver& crsolver, std::shared_ptr<ShapeSignature> metric, unsigned int active_set = 0);

		const Eigen::VectorXd& similarity_ratings();
		std::shared_ptr<ShapeSignature> metric() { return _metric; }

	private:
		SelfSimilarityMap();

		std::shared_ptr<ShapeSignature> _metric;
		Eigen::VectorXd _similarity_ratings;
};

#endif
