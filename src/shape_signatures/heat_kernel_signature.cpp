#include "heat_kernel_signature.h"

#include <memory>
#include <exception>

#include <unsupported/Eigen/MatrixFunctions>

#include <cppoptlib/solver/lbfgsbsolver.h>

#include <geometry/patch.h>
#include <matching/threshold.h>
#include <matching/quadratic_bump.h>
#include <utilities/units.h>
#include <matching/constrained_relation_solver.h>
#include <matching/geodesic_fan.h>

struct HKSInstancer : public HeatKernelSignature {
	HKSInstancer(std::shared_ptr<Mesh> mesh, int k) : HeatKernelSignature(mesh, k) {}
	HKSInstancer(std::shared_ptr<Mesh> mesh, int steps, int k) : HeatKernelSignature(mesh, steps, k) {}
	HKSInstancer(std::shared_ptr<Mesh> mesh, double tmin, double tmax, int k) : HeatKernelSignature(mesh, tmin, tmax, k) {}
	HKSInstancer(std::shared_ptr<Mesh> mesh, double tmin, double tmax, int steps, int k) : HeatKernelSignature(mesh, tmin, tmax, steps, k) {}
};

HeatKernelSignature::HeatKernelSignature(std::shared_ptr<Mesh> mesh, int k):
	SpectralSignature(std::make_shared<HKSParameterOptimization>(), mesh, k),
	_t_steps(Eigen::VectorXd::Zero(0)) {
	
	if (!mesh->loaded()) {
		return;
	}

	double tmin = t_lower_bound();
	double tmax = t_upper_bound();
	int	steps = eigenvalues().rows();

	_t_steps = hks_steps(tmin, tmax, steps);
	_sig = calculate_hks(_t_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

HeatKernelSignature::HeatKernelSignature(std::shared_ptr<Mesh> mesh, int steps, int k) :
	SpectralSignature(std::make_shared<HKSParameterOptimization>(), mesh, k),
	_t_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	double tmin = t_lower_bound();
	double tmax = t_upper_bound();

	_t_steps = hks_steps(tmin, tmax, steps);
	_sig = calculate_hks(_t_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

HeatKernelSignature::HeatKernelSignature(std::shared_ptr<Mesh> mesh, double tmin, double tmax, int k):
	SpectralSignature(std::make_shared<HKSParameterOptimization>(), mesh, k),
	_t_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	int	steps = eigenvalues().rows();

	_t_steps = hks_steps(tmin, tmax, steps);
	_sig = calculate_hks(_t_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

HeatKernelSignature::HeatKernelSignature(std::shared_ptr<Mesh> mesh, double tmin, double tmax, int steps, int k):
	SpectralSignature(std::make_shared<HKSParameterOptimization>(), mesh, k),
	_t_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	_t_steps = hks_steps(tmin, tmax, steps);
	_sig = calculate_hks(_t_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

HeatKernelSignature::~HeatKernelSignature() {

}

std::shared_ptr<HeatKernelSignature> HeatKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, int k) {
	std::shared_ptr<HeatKernelSignature> hks = std::make_shared<HKSInstancer>(mesh, k);

	hks->_param_opt->bind_signature(hks);

	return hks;
}

std::shared_ptr<HeatKernelSignature> HeatKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, int steps, int k) {
	std::shared_ptr<HeatKernelSignature> hks = std::make_shared<HKSInstancer>(mesh, steps, k);

	hks->_param_opt->bind_signature(hks);

	return hks;
}

std::shared_ptr<HeatKernelSignature> HeatKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, double tmin, double tmax, int k) {
	std::shared_ptr<HeatKernelSignature> hks = std::make_shared<HKSInstancer>(mesh, tmin, tmax, k);

	hks->_param_opt->bind_signature(hks);

	return hks;
}

std::shared_ptr<HeatKernelSignature> HeatKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, double tmin, double tmax, int steps, int k) {
	std::shared_ptr<HeatKernelSignature> hks = std::make_shared<HKSInstancer>(mesh, tmin, tmax, steps, k);

	hks->_param_opt->bind_signature(hks);

	return hks;
}

void HeatKernelSignature::resample_at_t(double t, int k) {
	//int steps = eigenvalues().rows();
	resample_k(k);

	_t_steps = hks_steps(t, t, 1);
	_sig = calculate_hks(_t_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

void HeatKernelSignature::resample_at_param(double param) {
	return resample_at_t(param, -1);
}

std::shared_ptr<HeatKernelSignature> operator-(std::shared_ptr<HeatKernelSignature> lhs, std::shared_ptr<HeatKernelSignature> rhs) {
	std::shared_ptr<HeatKernelSignature> hks = nullptr;

	const Eigen::MatrixXd& A = lhs->get_signature_values();
	const Eigen::MatrixXd& B = rhs->get_signature_values();

	if (A.rows() != B.rows() || A.cols() != B.cols()) {
		// operation not defined on differently sized matrices!
		return hks;
	}

	Eigen::MatrixXd residual = lhs->get_signature_values() - rhs->get_signature_values();

	throw std::exception("Not implemented!");
}

const Eigen::VectorXd HeatKernelSignature::lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid) {
	return _sig.row(vid);
}

Eigen::VectorXd HeatKernelSignature::lerpable_to_signature_value(const Eigen::VectorXd& lerped) {
	return lerped;
}

unsigned long HeatKernelSignature::feature_count() {
	return _sig.rows();
}

unsigned long HeatKernelSignature::feature_dimension() {
	return _sig.cols();
}

const Eigen::MatrixXd& HeatKernelSignature::get_signature_values() {
	return ShapeSignature::get_signature_values();
}

Eigen::VectorXd HeatKernelSignature::get_signature_values(double index) {
	Eigen::DenseIndex n = -1;
	double dist = std::numeric_limits<double>::max();

	for (Eigen::DenseIndex i = 0; i < _t_steps.rows(); ++i) {
		if (std::fabs(_t_steps(i) - index) < dist) {
			n = i;
		}
	}

	if (n > -1) {
		return std::move(get_signature_values().col(n));
	}

	return Eigen::VectorXd();
}

int HeatKernelSignature::get_k_pairs_used() {
	return get_current_k();
}

unsigned int HeatKernelSignature::get_steps() {
	return _t_steps.size();
}

double HeatKernelSignature::get_tmin() {
	return _t_steps(0);
}

double HeatKernelSignature::get_tmax() {
	return _t_steps(_t_steps.size() - 1);
}

Eigen::VectorXd HeatKernelSignature::hks_steps(double tmin, double tmax, int steps) const {
	if (tmin > tmax || steps <= 0) {
		tmin = tmax;
		steps = 1;
	}

	Eigen::VectorXd t_steps;
	const Eigen::VectorXd& evals = this->eigenvalues();

	if (evals.rows() < evals.cols()) {
		return t_steps;
	}

	double stepsize;

	if (steps == 1) {
		stepsize = 0.0;
	} else {
		stepsize = (std::log(tmax) - std::log(tmin)) / (steps - 1);
	}

	t_steps.setLinSpaced(steps, 0.0, steps - 1);
	t_steps = ((t_steps * stepsize).array() + std::log(tmin)).exp();

	return t_steps;
}

Eigen::MatrixXd HeatKernelSignature::calculate_hks(const Eigen::VectorXd& t_steps) const {
	if (_mesh == nullptr) {
		return Eigen::MatrixXd();
	}

	if (!(t_steps.size() > 0)) {
		return Eigen::MatrixXd();
	}

	// TODO: The next two statements are the most expensive part of the entire operation -- can anything be done to avoid/reduce/optimize them?
	Eigen::MatrixXd evals_t = ((-1.0) * eigenvalues().cwiseAbs() * t_steps.transpose()).array().exp();

	Eigen::MatrixXd hks = eigenvectors().cwiseProduct(eigenvectors()) * evals_t;

	// HKS scaling (TODO: implement Scale-invariant HKS instead of just dividing by the heat trace)
	Eigen::VectorXd heat_trace = evals_t.array().colwise().sum();
	for (unsigned int i = 0; i < heat_trace.size(); ++i) {
		hks.col(i) /= heat_trace(i);
	}

	return std::move(hks);
}

Eigen::MatrixXd HeatKernelSignature::sig_steps() {
	return _t_steps;
}

const double HeatKernelSignature::step_width(double param) {
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

double HeatKernelSignature::t_lower_bound() const {
	// Heat Kernel Signature
	// [tmin, tmax] as suggested in SOG09: http://dl.acm.org/citation.cfm?id=1735603.1735621
	const Eigen::MatrixXd& evals = eigenvalues();

	return 4.0 * std::log(10) / std::fabs(evals(evals.rows() - 1));
}

double HeatKernelSignature::t_upper_bound() const {
	// Heat Kernel Signature
	// [tmin, tmax] as suggested in SOG09: http://dl.acm.org/citation.cfm?id=1735603.1735621
	const Eigen::MatrixXd& evals = eigenvalues();

	return 4.0 * std::log(10) / std::fabs(evals(1));
}

double HeatKernelSignature::param_lower_bound() {
	return t_lower_bound();
}
double HeatKernelSignature::param_upper_bound() {
	return t_upper_bound();
}

HeatKernelSignature::HKSParameterOptimization::HKSParameterOptimization() {

}

HeatKernelSignature::HKSParameterOptimization::~HKSParameterOptimization() {

}

// An objective function of one value
// Input: Eigen::VectorXd size of 1, whose value is the t parameter of the HKS with which to compare patch geodesic fans
// Output: A double indicating the optimility of the input t parameter is maximizing distance of excluded patches and minimizing distance of included patches
double HeatKernelSignature::HKSParameterOptimization::value(const TVector &x) {
	std::vector<Relation>& rels = _value_desire_set;

	if (rels.size() < 2) {
		// No minimization needed for a problem with one or zero patches
		return 0.0;
	}
	
	std::shared_ptr<HeatKernelSignature> sig = std::dynamic_pointer_cast<HeatKernelSignature>(_optimizing_sig.lock());

	double t = x(0);
	double inc_obj = 0.0;
	double ex_obj = 0.0;
	double penalty = 0.0;

	// Generate penalty for clamping
	penalty += std::pow(std::max(0.0, sig->t_lower_bound() - t), 2);
	penalty += std::pow(std::max(0.0, t - sig->t_upper_bound()), 2);

	// clamp
	t = std::max(sig->t_lower_bound(), std::min(t, sig->t_upper_bound()));

	// Update signature with full signature tmin, tmax if necessary
	std::shared_ptr<HeatKernelSignature> obj_interval = HeatKernelSignature::instantiate(sig->_mesh, t, t, 1);

	// g(x) = f_i(x) + 1 / (1e-10 + f_e(x));
	double objective = 0.0;

	std::vector<std::pair<std::shared_ptr<GeodesicFan>, Relation::Designation>> fans;

	for (Relation& r : rels) {
		std::shared_ptr<Patch> patch = r._patch;

		auto it = _metrics_map.find(patch);

		Metrics patch_metrics;

		if (it == _metrics_map.end()) {
			// There seems to be no saved metrics, so generate them here
			patch_metrics._centroid_vid = patch->get_centroid_vid_on_origin_mesh();
			patch_metrics._geodesic_radius = patch->get_geodesic_extent(patch_metrics._centroid_vid);

			patch_metrics._dem = patch->discrete_exponential_map(patch_metrics._centroid_vid);

			_metrics_map.insert(std::pair<std::shared_ptr<Patch>, Metrics>(patch, patch_metrics));
		} else {
			patch_metrics = it->second;
		}

		// Fans are based off of value inputs, so can't be pre-computed
		auto fan = std::make_shared<GeodesicFan>(patch_metrics._dem, obj_interval);

		fans.push_back(std::pair<std::shared_ptr<GeodesicFan>, Relation::Designation>(fan, r._designation));
	}

	for (unsigned int i = 0; i < fans.size(); ++i) {
		for (unsigned int j = i + 1; j < fans.size(); ++j) {
			// Subsequent patch for comparison
			double orientation = 0.0;
			double comp = fans[i].first->compare(*fans[j].first, orientation);

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

cppoptlib::Problem<double>::TVector HeatKernelSignature::HKSParameterOptimization::upperBound() const {
	std::shared_ptr<HeatKernelSignature> sig = std::dynamic_pointer_cast<HeatKernelSignature>(_optimizing_sig.lock());

	TVector upper_bound(1);
	upper_bound << sig->t_upper_bound();

	return upper_bound;
}

cppoptlib::Problem<double>::TVector HeatKernelSignature::HKSParameterOptimization::lowerBound() const {
	std::shared_ptr<HeatKernelSignature> sig = std::dynamic_pointer_cast<HeatKernelSignature>(_optimizing_sig.lock());

	TVector lower_bound(1);	// tmin, tmax
	lower_bound << sig->t_lower_bound();

	return lower_bound;
}

Eigen::MatrixXd HeatKernelSignature::HKSParameterOptimization::param_steps(unsigned int steps) {
	auto sig = std::dynamic_pointer_cast<HeatKernelSignature>(_optimizing_sig.lock());

	assert(sig != nullptr && "Bound signature is invalid!");

	return sig->hks_steps(sig->t_lower_bound(), sig->t_upper_bound(), steps);
}

bool HeatKernelSignature::HKSParameterOptimization::callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x) {
	// Capture state of solver per step taken
	std::cout << "step: x = " << x.transpose() << std::endl;

	return true;
}

std::shared_ptr<GeodesicFan> HeatKernelSignature::HKSParameterOptimization::geodesic_fan_from_relation(const Relation& r) {
	// There seems to be no saved metrics, so generate them here
	std::shared_ptr<HeatKernelSignature> sig = std::dynamic_pointer_cast<HeatKernelSignature>(_optimizing_sig.lock());

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