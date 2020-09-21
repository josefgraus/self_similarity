#include "shape_signature.h"

#include <exception>
#include <algorithm>

#include <cppoptlib/solver/lbfgsbsolver.h>

#include <geometry/patch.h>
#include <geometry/mesh.h>
#include <matching/geodesic_fan.h>
#include <matching/threshold.h>
#include <matching/constrained_relation_solver.h>
#include <matching/quadratic_bump.h>

ShapeSignature::ShapeSignature(std::shared_ptr<Mesh> mesh, std::shared_ptr<ParameterOptimization> param_opt) :
	_param_opt(param_opt),
	_mesh(mesh),
	_sig(Eigen::MatrixXd::Zero(0, 0)),
	_exception_map(Eigen::MatrixXd::Zero(0, 0)),
	_exception_filtered(Eigen::MatrixXd::Zero(0, 0)) {

	if (param_opt == nullptr) {
		throw std::logic_error("ParameterOptimzation for signature is not properly implemented!");
	}
}

ShapeSignature::~ShapeSignature() {
}

std::shared_ptr<Mesh> ShapeSignature::origin_mesh() {
	return _mesh;
}

const Eigen::MatrixXd& ShapeSignature::get_signature_values() {
	_exception_filtered = _sig + _exception_map;

	return _exception_filtered;
}

double ShapeSignature::lower_bound() {
	auto sig_vals = get_signature_values();

	if (sig_vals.size() <= 0) {
		return 0.0;
	}

	return sig_vals.minCoeff();
}

double ShapeSignature::upper_bound() {
	auto sig_vals = get_signature_values();

	if (sig_vals.size() <= 0) {
		return 0.0;
	}

	return sig_vals.maxCoeff();
}

void ShapeSignature::apply_quadratic_bump(QuadraticBump<double> bump) {
	_bumps.push_back(bump);

	// Update _exception_map with new bump
	Eigen::VectorXd energy_curve = Eigen::VectorXd::Zero(_exception_map.cols());

	auto steps = sig_steps();
	for (Eigen::DenseIndex i = 0; i < energy_curve.cols(); ++i) {
		energy_curve(i) = bump.energy_shift_by_parameter(steps(i));
	}

	// add the energy to its row in the exception map
	Eigen::DenseIndex vid = bump.feature_index();
	_exception_map.row(vid) = _exception_map.row(vid) + energy_curve.transpose();
}

void ShapeSignature::clear_quadratic_bumps() {
	_bumps.clear();

	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

ShapeSignature::ParameterOptimization::ParameterOptimization() {
}

ShapeSignature::ParameterOptimization::~ParameterOptimization() {

}

void ShapeSignature::ParameterOptimization::bind_signature(std::shared_ptr<ShapeSignature> optimizing_sig) {
	if (optimizing_sig == nullptr) {
		throw std::invalid_argument("The signature associated with an optimization instance can't be null!");
	}

	_optimizing_sig = optimizing_sig;
}

void ShapeSignature::ParameterOptimization::set_value_desire_set(std::vector<Relation> desire_set) {
	_value_desire_set = desire_set;
}