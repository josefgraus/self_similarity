#include "wave_kernel_signature.h"

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

struct WKSInstancer : public WaveKernelSignature {
	WKSInstancer(std::shared_ptr<Mesh> mesh, int k) : WaveKernelSignature(mesh, k) {}
	WKSInstancer(std::shared_ptr<Mesh> mesh, int steps, int k) : WaveKernelSignature(mesh, steps, k) {}
	WKSInstancer(std::shared_ptr<Mesh> mesh, double emin, double emax, int k) : WaveKernelSignature(mesh, emin, emax, k) {}
	WKSInstancer(std::shared_ptr<Mesh> mesh, double emin, double emax, int steps, int k) : WaveKernelSignature(mesh, emin, emax, steps, k) {}
};

WaveKernelSignature::WaveKernelSignature(std::shared_ptr<Mesh> mesh, int k) :
	SpectralSignature(std::make_shared<WKSParameterOptimization>(), mesh, k),
	_e_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	double emin = e_lower_bound();
	double emax = e_upper_bound();
	int	steps = eigenvalues().rows();

	_e_steps = wks_steps(emin, emax, steps);
	_sig = calculate_wks(_e_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

WaveKernelSignature::WaveKernelSignature(std::shared_ptr<Mesh> mesh, int steps, int k) :
	SpectralSignature(std::make_shared<WKSParameterOptimization>(), mesh, k),
	_e_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	double emin = e_lower_bound();
	double emax = e_upper_bound();

	_e_steps = wks_steps(emin, emax, steps);
	_sig = calculate_wks(_e_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

WaveKernelSignature::WaveKernelSignature(std::shared_ptr<Mesh> mesh, double emin, double emax, int k) :
	SpectralSignature(std::make_shared<WKSParameterOptimization>(), mesh, k),
	_e_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	int	steps = eigenvalues().rows();

	_e_steps = wks_steps(emin, emax, steps);
	_sig = calculate_wks(_e_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

WaveKernelSignature::WaveKernelSignature(std::shared_ptr<Mesh> mesh, double emin, double emax, int steps, int k) :
	SpectralSignature(std::make_shared<WKSParameterOptimization>(), mesh, k),
	_e_steps(Eigen::VectorXd::Zero(0)) {

	if (!mesh->loaded()) {
		return;
	}

	_e_steps = wks_steps(emin, emax, steps);
	_sig = calculate_wks(_e_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

WaveKernelSignature::~WaveKernelSignature() {

}

std::shared_ptr<WaveKernelSignature> WaveKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, int k) {
	std::shared_ptr<WaveKernelSignature> wks = std::make_shared<WKSInstancer>(mesh, k);

	wks->_param_opt->bind_signature(wks);

	return wks;
}

std::shared_ptr<WaveKernelSignature> WaveKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, int steps, int k) {
	std::shared_ptr<WaveKernelSignature> wks = std::make_shared<WKSInstancer>(mesh, steps, k);

	wks->_param_opt->bind_signature(wks);

	return wks;
}

std::shared_ptr<WaveKernelSignature> WaveKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, double emin, double emax, int k) {
	std::shared_ptr<WaveKernelSignature> wks = std::make_shared<WKSInstancer>(mesh, emin, emax, k);

	wks->_param_opt->bind_signature(wks);

	return wks;
}

std::shared_ptr<WaveKernelSignature> WaveKernelSignature::instantiate(std::shared_ptr<Mesh> mesh, double emin, double emax, int steps, int k) {
	std::shared_ptr<WaveKernelSignature> wks = std::make_shared<WKSInstancer>(mesh, emin, emax, steps, k);

	wks->_param_opt->bind_signature(wks);

	return wks;
}

void WaveKernelSignature::resample_at_e(double t, int k) {
	//int	steps = eigenvalues().rows();
	resample_k(k);

	_e_steps = wks_steps(t, t, 1);
	_sig = calculate_wks(_e_steps);
	_exception_map = Eigen::MatrixXd::Zero(_sig.rows(), _sig.cols());
}

std::shared_ptr<WaveKernelSignature> operator-(std::shared_ptr<WaveKernelSignature> lhs, std::shared_ptr<WaveKernelSignature> rhs) {
	std::shared_ptr<WaveKernelSignature> wks = nullptr;

	const Eigen::MatrixXd& A = lhs->get_signature_values();
	const Eigen::MatrixXd& B = rhs->get_signature_values();

	if (A.rows() != B.rows() || A.cols() != B.cols()) {
		// operation not defined on differently sized matrices!
		return wks;
	}

	Eigen::MatrixXd residual = lhs->get_signature_values() - rhs->get_signature_values();


}

const Eigen::VectorXd WaveKernelSignature::lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid) {
	return _sig.row(vid);
}

Eigen::VectorXd WaveKernelSignature::lerpable_to_signature_value(const Eigen::VectorXd& lerped) {
	return lerped;
}

unsigned long WaveKernelSignature::feature_count() {
	return _sig.rows();
}

unsigned long WaveKernelSignature::feature_dimension() {
	return _sig.cols();
}

const Eigen::MatrixXd& WaveKernelSignature::get_signature_values() {
	return ShapeSignature::get_signature_values();
}

Eigen::VectorXd WaveKernelSignature::get_signature_values(double index) {
	Eigen::DenseIndex n = -1;
	double dist = std::numeric_limits<double>::max();

	for (Eigen::DenseIndex i = 0; i < _e_steps.rows(); ++i) {
		if (std::fabs(_e_steps(i) - index) < dist) {
			n = i;
		}
	}

	if (n > -1) {
		return std::move(get_signature_values().col(n));
	}

	return Eigen::VectorXd();
}

int WaveKernelSignature::get_k_pairs_used() {
	return get_current_k();
}

unsigned int WaveKernelSignature::get_steps() {
	return _e_steps.size();
}

double WaveKernelSignature::get_emin() {
	return _e_steps(0);
}

double WaveKernelSignature::get_emax() {
	return _e_steps(_e_steps.size() - 1);
}

Eigen::VectorXd WaveKernelSignature::wks_steps(double emin, double emax, int steps) const {
	if (emin > emax || steps <= 0) {
		emin = emax;
		steps = 1;
	}

	Eigen::VectorXd e_steps;
	const Eigen::VectorXd& evals = this->eigenvalues();

	if (evals.rows() < evals.cols()) {
		return e_steps;
	}

	double stepsize;

	if (steps == 1) {
		stepsize = 0.0;
	} else {
		stepsize = (emax - emin) / (steps - 1);
	}

	e_steps.setLinSpaced(steps, 0.0, steps - 1);
	e_steps = ((e_steps * stepsize).array() + emin);

	return e_steps;
}

void WaveKernelSignature::resample_at_param(double param) {
	return resample_at_e(param, -1);
}

Eigen::MatrixXd WaveKernelSignature::calculate_wks(const Eigen::VectorXd& e_steps) const {
	if (_mesh == nullptr) {
		return Eigen::MatrixXd();
	}

	if (!(e_steps.size() > 0)) {
		return Eigen::MatrixXd();
	}

	double sigma = 1.0;

	if (e_steps.size() > 1) {
		sigma = e_steps(1) - e_steps(0);
	}

	// TODO: The next two statements are the most expensive part of the entire operation -- can anything be done to avoid/reduce/optimize them?
	//Eigen::MatrixXd evals_t = ((-1.0) * eigenvalues().cwiseAbs() * e_steps.transpose()).array().exp();
	
	Eigen::MatrixXd evals_t(eigenvalues().rows(), e_steps.rows());
	Eigen::VectorXd e_diff = eigenvalues().cwiseAbs().array().log();
	for (Eigen::DenseIndex i = 0; i < e_steps.rows(); ++i) {
		evals_t.col(i) = (-1.0) * (e_steps(i) - e_diff.array()).square() / (2.0 * std::pow(sigma, 2.0));
	}

	Eigen::MatrixXd wks = eigenvectors().cwiseProduct(eigenvectors()) * evals_t;

	// WKS scaling
	Eigen::VectorXd energy_trace = evals_t.array().colwise().sum();
	for (unsigned int i = 0; i < energy_trace.size(); ++i) {
		wks.col(i) /= energy_trace(i);
	}

	return std::move(wks);
}

Eigen::MatrixXd WaveKernelSignature::sig_steps() {
	return _e_steps;
}

const double WaveKernelSignature::step_width(double param) {
	double lower = lower_bound();
	double upper = upper_bound();

	int index = -1;
	double dist = std::numeric_limits<double>::max();

	for (int i = 0; i < _e_steps.size(); ++i) {
		double tdist = std::fabs(param - _e_steps(i));

		if (tdist < dist) {
			index = i;
			dist = tdist;
		}
	}

	if (index < 0 || std::fabs(_e_steps(index) - param) > std::numeric_limits<double>::epsilon()) {
		throw std::logic_error("This parameter does not currently exist as a sampled column in the signature!");
	}

	if (index > 0) {
		lower = _e_steps(index - 1);
	}

	if (index < _e_steps.size() - 1) {
		upper = _e_steps.size() - 1;
	}

	return std::min(std::fabs(param - lower), std::fabs(param - upper));
}

double WaveKernelSignature::param_lower_bound() {
	return e_lower_bound();
}
double WaveKernelSignature::param_upper_bound() {
	return e_upper_bound();
}

double WaveKernelSignature::e_lower_bound() const {
	// Heat Kernel Signature
	// [emin, emax] as suggested in ASC11: https://vision.cs.tum.edu/_media/spezial/bib/aubry-et-al-4dmod11.pdf
	const Eigen::MatrixXd& evals = eigenvalues();

	return std::log(std::fabs(evals(0)));
}

double WaveKernelSignature::e_upper_bound() const {
	// Heat Kernel Signature
	// [emin, emax] as suggested in ASC11: https://vision.cs.tum.edu/_media/spezial/bib/aubry-et-al-4dmod11.pdf
	const Eigen::MatrixXd& evals = eigenvalues();

	return std::log(std::fabs(evals(evals.rows() - 1)));
}

WaveKernelSignature::WKSParameterOptimization::WKSParameterOptimization() {

}

WaveKernelSignature::WKSParameterOptimization::~WKSParameterOptimization() {

}

// An objective function of one value
// Input: Eigen::VectorXd size of 1, whose value is the t parameter of the WKS with which to compare patch geodesic fans
// Output: A double indicating the optimility of the input t parameter is maximizing distance of excluded patches and minimizing distance of included patches
double WaveKernelSignature::WKSParameterOptimization::value(const TVector &x) {
	std::vector<Relation>& rels = _value_desire_set;

	if (rels.size() < 2) {
		// No minimization needed for a problem with one or zero patches
		return 0.0;
	}

	std::shared_ptr<WaveKernelSignature> sig = std::dynamic_pointer_cast<WaveKernelSignature>(_optimizing_sig.lock());

	double e = x(0);
	double inc_obj = 0.0;
	double ex_obj = 0.0;
	double penalty = 0.0;

	// Generate penalty for clamping
	penalty += std::pow(std::max(0.0, sig->e_lower_bound() - e), 2);
	penalty += std::pow(std::max(0.0, e - sig->e_upper_bound()), 2);

	// clamp
	double emin = sig->e_lower_bound();
	double emax = sig->e_upper_bound();

	e = std::max(sig->e_lower_bound(), std::min(e, sig->e_upper_bound()));

	// Update signature with full signature emin, emax if necessary
	//std::shared_ptr<WaveKernelSignature> obj_interval = WaveKernelSignature::instantiate(sig->_mesh, t, t, 1);
	sig->resample_at_e(e, sig->get_current_k());

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
		auto fan = std::make_shared<GeodesicFan>(patch_metrics._dem, sig);

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

cppoptlib::Problem<double>::TVector WaveKernelSignature::WKSParameterOptimization::upperBound() const {
	std::shared_ptr<WaveKernelSignature> sig = std::dynamic_pointer_cast<WaveKernelSignature>(_optimizing_sig.lock());

	TVector upper_bound(1);
	upper_bound << sig->e_upper_bound();

	return upper_bound;
}

cppoptlib::Problem<double>::TVector WaveKernelSignature::WKSParameterOptimization::lowerBound() const {
	std::shared_ptr<WaveKernelSignature> sig = std::dynamic_pointer_cast<WaveKernelSignature>(_optimizing_sig.lock());

	TVector lower_bound(1);	// tmin, tmax
	lower_bound << sig->e_lower_bound();

	return lower_bound;
}

Eigen::MatrixXd WaveKernelSignature::WKSParameterOptimization::param_steps(unsigned int steps) {
	auto sig = std::dynamic_pointer_cast<WaveKernelSignature>(_optimizing_sig.lock());

	assert(sig != nullptr && "Bound signature is invalid!");

	return sig->wks_steps(sig->e_lower_bound(), sig->e_upper_bound(), steps);
}

bool WaveKernelSignature::WKSParameterOptimization::callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x) {
	// Capture state of solver per step taken
	std::cout << "step: x = " << x.transpose() << std::endl;

	return true;
}

std::shared_ptr<GeodesicFan> WaveKernelSignature::WKSParameterOptimization::geodesic_fan_from_relation(const Relation& r) {
	// There seems to be no saved metrics, so generate them here
	std::shared_ptr<WaveKernelSignature> sig = std::dynamic_pointer_cast<WaveKernelSignature>(_optimizing_sig.lock());

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