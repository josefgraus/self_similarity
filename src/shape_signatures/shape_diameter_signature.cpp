#include "shape_diameter_signature.h"

#ifdef _WIN32
#include <io.h> 
#define access    _access_s
#else
#include <unistd.h>
#endif

#include <unsupported/Eigen/MatrixFunctions>

#include <cppoptlib/solver/lbfgsbsolver.h>

#include <igl/shape_diameter_function.h>
#include <igl/hessian_energy.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>

#include <geometry/mesh.h>
#include <geometry/patch.h>
#include <matching/threshold.h>
#include <matching/geodesic_fan.h>
#include <utilities/units.h>
#include <utilities/eigen_read_write_binary.h>
#include <matching/constrained_relation_solver.h>

struct SDFInstancer : public ShapeDiameterSignature {
	SDFInstancer(std::shared_ptr<Mesh> mesh, double t) : ShapeDiameterSignature(mesh, t) {}
};

ShapeDiameterSignature::ShapeDiameterSignature(const std::shared_ptr<Mesh> mesh, double t):
	ShapeSignature(mesh, std::make_shared<SDFParameterOptimization>()), 
	_t(t) {

	if (!mesh->loaded()) {
		return;
	}

	calc_sdf_data(mesh->resource_dir(), mesh->name(), 1000);
	resample_at_t(t);
}

ShapeDiameterSignature::~ShapeDiameterSignature() {

}

void ShapeDiameterSignature::calc_sdf_data(std::string resource_dir, std::string model_name, int k) {
	// Attempt to load precalculated file
	std::string hessian_path = resource_dir + "//cache//" + model_name + ".hessian";
	std::string massmatrix_path = resource_dir + "//cache//" + model_name + ".massmat";
	std::string sdf_path = resource_dir + "//cache//" + model_name + ".sdf";

	bool calculated = false;
	if (!(access(hessian_path.c_str(), 0) == 0) || !(access(massmatrix_path.c_str(), 0) == 0) || !(access(sdf_path.c_str(), 0) == 0)) {
		// Files don't exist, or are inaccessible

		Eigen::MatrixXd V = _mesh->vertices().leftCols<3>();
		Eigen::MatrixXi F = _mesh->faces().leftCols<3>();
		Eigen::MatrixXd N = _mesh->vertex_normals().leftCols<3>();

		igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, _M2);

		igl::shape_diameter_function(V, F, V, N, k, _sdf_zero);
		igl::hessian_energy(V, F, _QH);

		write_binary(hessian_path.c_str(), _QH);
		write_binary(massmatrix_path.c_str(), _M2);
		write_binary(sdf_path.c_str(), _sdf_zero);
	}
	else {
		// Else, use precalculated matrices
		//std::cout << "Loading saved shape diameter function values..." << std::endl;
		Eigen::MatrixXd dQH;
		Eigen::MatrixXd dM2;

		read_binary(hessian_path.c_str(), _QH);
		read_binary(massmatrix_path.c_str(), _M2);
		read_binary(sdf_path.c_str(), _sdf_zero);
	}
}

std::shared_ptr<ShapeDiameterSignature> ShapeDiameterSignature::instantiate(std::shared_ptr<Mesh> mesh, double t) {
	std::shared_ptr<ShapeDiameterSignature> sdf = std::make_shared<SDFInstancer>(mesh, t);

	sdf->_param_opt->bind_signature(sdf);

	return sdf;
}

const Eigen::VectorXd ShapeDiameterSignature::lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid) {
	return _sig.row(vid);
}

Eigen::VectorXd ShapeDiameterSignature::lerpable_to_signature_value(const Eigen::VectorXd& lerped) {
	return lerped;
}

unsigned long ShapeDiameterSignature::feature_count() {
	return _sig.rows();
}

unsigned long ShapeDiameterSignature::feature_dimension() {
	return _sig.cols();
}

const Eigen::MatrixXd& ShapeDiameterSignature::get_signature_values() {
	return ShapeSignature::get_signature_values();
}

Eigen::VectorXd ShapeDiameterSignature::get_signature_values(double index) {
	Eigen::DenseIndex n = -1;
	double dist = std::numeric_limits<double>::max();

	for (Eigen::DenseIndex i = 0; i < _t_steps.rows(); ++i) {
		if (std::fabs(_t_steps(i) - index) < dist) {
			n = i;
		}
	}

	if (n > -1) {
		return get_signature_values().col(n);
	}

	return Eigen::VectorXd();
}

double ShapeDiameterSignature::t_lower_bound() {
	return 0.0;
}

double ShapeDiameterSignature::t_upper_bound() {
	return 1.0 - 1e-7;
}

double ShapeDiameterSignature::param_lower_bound() {
	return t_lower_bound();
}
double ShapeDiameterSignature::param_upper_bound() {
	return t_upper_bound();
}

Eigen::MatrixXd ShapeDiameterSignature::calculate_sdf(const Eigen::VectorXd& t_steps) const {
	if (_mesh == nullptr) {
		return Eigen::MatrixXd();
	}

	if (!(t_steps.size() > 0)) {
		return Eigen::MatrixXd();
	}

	Eigen::MatrixXd smoothed_sdf(_mesh->vertices().rows(), t_steps.size());

	for (int i = 0; i < t_steps.size(); ++i) {
		const double& alpha = t_steps(i);
		
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> hessSolver(alpha * _QH + (1. - alpha) * _M2);
		smoothed_sdf.col(i) << hessSolver.solve((1.0 - alpha) * _M2 * _sdf_zero);
	}

	return std::move(smoothed_sdf);
}

Eigen::MatrixXd ShapeDiameterSignature::sig_steps() {
	return _t_steps;
}

const double ShapeDiameterSignature::step_width(double param) {
	double lower = lower_bound();
	double upper = upper_bound();

	int index = -1;
	double dist = std::numeric_limits<double>::max();

	for (int i = 0; i < _t_steps.size(); ++i) {
		double tdist = std::fabs(param - _t_steps(i));

		if (tdist < dist) {
			index = i;
			dist = tdist;
		}
	}

	if (index < 0 || std::fabs(_t_steps(index) - param) > std::numeric_limits<double>::epsilon()) {
		throw std::logic_error("This parameter does not currently exist as a sampled column in the signature!");
	}

	if (index > 0) {
		lower = _t_steps(index - 1);
	}

	if (index < _t_steps.size() - 1) {
		upper = _t_steps.size() - 1;
	}

	return std::min(std::fabs(param - lower), std::fabs(param - upper));
}

void ShapeDiameterSignature::resample_at_t(double t) {
	// TODO: Fix this race condition
	_t_steps = sdf_steps(t, t, 1);
	_sig = calculate_sdf(_t_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
	_exception_filtered = _sig;
	_bumps.clear();
}

void ShapeDiameterSignature::resample_at_param(double param) {
	return resample_at_t(param);
}

Eigen::VectorXd ShapeDiameterSignature::sdf_steps(double tmin, double tmax, int steps) const {
	if (tmin > tmax || steps <= 0) {
		tmin = tmax;
		steps = 1;
	}

	Eigen::VectorXd t_steps;
	double stepsize;

	if (steps == 1) {
		stepsize = 0.0;
	}
	else {
		stepsize = (tmax - tmin) / (steps - 1);
	}

	t_steps.setLinSpaced(steps, 0.0, steps - 1);
	t_steps = (t_steps * stepsize).array() + tmin;

	return t_steps;
}

ShapeDiameterSignature::SDFParameterOptimization::SDFParameterOptimization() {

}

ShapeDiameterSignature::SDFParameterOptimization::~SDFParameterOptimization() {

}

// An objective function of one value
// Input: Eigen::VectorXd size of 1, whose value is the t parameter of the HKS with which to compare patch geodesic fans
// Output: A double indicating the optimility of the input t parameter is maximizing distance of excluded patches and minimizing distance of included patches
double ShapeDiameterSignature::SDFParameterOptimization::value(const TVector &x) {
	std::vector<Relation>& rels = _value_desire_set;

	if (rels.size() < 1) {
		// No minimization needed for a problem with one or zero patches
		return 0.0;
	}

	std::shared_ptr<ShapeDiameterSignature> sig = std::dynamic_pointer_cast<ShapeDiameterSignature>(_optimizing_sig.lock());

	double t = x(0);
	double inc_obj = 0.0;
	double ex_obj = 0.0;
	double penalty = 0.0;

	// clamp
	t = std::max(sig->t_lower_bound(), std::min(t, sig->t_upper_bound()));

	// Update signature with full signature tmin, tmax if necessary
	//std::shared_ptr<ShapeDiameterSignature> obj_interval = ShapeDiameterSignature::instantiate(sig->_mesh, t);
	Eigen::MatrixXd sig_values = sig->calculate_sdf(sig->sdf_steps(t, t, 1));

	// g(x) = f_i(x) + 1 / (1e-10 + f_e(x));
	double objective = 0.0;

	std::vector<std::pair<Eigen::DenseIndex, Relation::Designation>> fans;

	for (Relation& r : rels) {
		// Just store away the value at the single point associated with the relation
		std::shared_ptr<Patch> patch = r._patch;

		double sig_value = 0.0;
		
		if (patch != nullptr) {
			assert(patch->vids().size() == 1);

			sig_value = sig_values(*patch->vids().begin(), 0);
		} else {
			sig_value = r._bc.to_sig_value(sig->origin_mesh(), sig_values)(0);
		}

		fans.push_back(std::make_pair(sig_value, r._designation));
	}

	for (unsigned int i = 0; i < fans.size(); ++i) {
		for (unsigned int j = i + 1; j < fans.size(); ++j) {
			// Subsequent patch for comparison
			double comp = std::fabs(fans[i].first - fans[j].first);

			if (fans[i].second == fans[j].second) {
				// Include
				inc_obj += comp;
			} else {
				// Exclude
				ex_obj += comp;
			}
		}
	}

	objective = ((1 + inc_obj) / (1 + ex_obj)) + penalty;

	return objective;
}

cppoptlib::Problem<double>::TVector ShapeDiameterSignature::SDFParameterOptimization::upperBound() const {
	std::shared_ptr<ShapeDiameterSignature> sig = std::dynamic_pointer_cast<ShapeDiameterSignature>(_optimizing_sig.lock());

	TVector upper_bound(1);
	upper_bound << sig->t_upper_bound();

	return upper_bound;
}

cppoptlib::Problem<double>::TVector ShapeDiameterSignature::SDFParameterOptimization::lowerBound() const {
	std::shared_ptr<ShapeDiameterSignature> sig = std::dynamic_pointer_cast<ShapeDiameterSignature>(_optimizing_sig.lock());

	TVector lower_bound(1);	// tmin, tmax
	lower_bound << sig->t_lower_bound();

	return lower_bound;
}

Eigen::MatrixXd ShapeDiameterSignature::SDFParameterOptimization::param_steps(unsigned int steps) {
	auto sig = std::dynamic_pointer_cast<ShapeDiameterSignature>(_optimizing_sig.lock());

	assert(sig != nullptr && "Bound signature is invalid!");

	return sig->sdf_steps(sig->t_lower_bound(), sig->t_upper_bound(), steps);
}

bool ShapeDiameterSignature::SDFParameterOptimization::callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x) {
	// Capture state of solver per step taken
	std::cout << "step: x = " << x.transpose() << std::endl;

	return true;
}

std::shared_ptr<GeodesicFan> ShapeDiameterSignature::SDFParameterOptimization::geodesic_fan_from_relation(const Relation& r) {
	// There seems to be no saved metrics, so generate them here
	std::shared_ptr<ShapeDiameterSignature> sig = std::dynamic_pointer_cast<ShapeDiameterSignature>(_optimizing_sig.lock());

	if (sig == nullptr) {
		return nullptr;
	}

	auto patch = r._patch;

	if (patch == nullptr) {
		return nullptr;
	}

	auto it = _metrics_map.find(patch);

	Metrics patch_metrics;

	if (it == _metrics_map.end()) {
		// There seems to be no saved metrics, so generate them here
		patch_metrics._centroid_vid = patch->get_centroid_vid_on_origin_mesh();
		patch_metrics._geodesic_radius = patch->get_geodesic_extent(patch_metrics._centroid_vid);

		patch_metrics._dem = patch->discrete_exponential_map(patch_metrics._centroid_vid);

		_metrics_map.insert(std::pair<std::shared_ptr<Patch>, Metrics>(patch, patch_metrics));
	}
	else {
		patch_metrics = it->second;
	}

	// Fans are based off of value inputs, so can't be pre-computed
	auto fan = std::make_shared<GeodesicFan>(patch_metrics._dem, sig);

	return fan;
}