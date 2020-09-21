#ifndef SURFACE_BY_NUMBERS_H_
#define SURFACE_By_NUMBERS_H_

#include <memory>

#include <geometry/component.h>
#include <matching/constrained_relation_solver.h>

class SurfaceByNumbers {
	public:
		SurfaceByNumbers(const CRSolver& crsolver, double eta, double gamma);
		~SurfaceByNumbers();

		const Eigen::VectorXi& segment_by_face() const;
		std::vector<std::shared_ptr<Component>> components();

	private:
		double SurfaceByNumbers::dihedral_angle(std::shared_ptr<Mesh> mesh, Eigen::DenseIndex face1, Eigen::DenseIndex face2) const;

		double _eta;
		double _gamma;

		Eigen::VectorXi _segment_by_face;
		std::shared_ptr<Mesh> _mesh;
};

#endif