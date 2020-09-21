#include "stroke_transfer.h"

#include <queue>
#include <random>
#include <thread>
#include <queue>
#include <chrono>

#include <cppoptlib/solver/lbfgsbsolver.h>
#include <ctpl_stl.h>

#include <matching/threshold.h>
#include <matching/geodesic_fan.h>
#include <matching/parameterization/curve_unrolling.h>

using namespace cppoptlib;

void curve_solve_proxy(int id, int index, std::shared_ptr<CurveUnrolling> cu_src, std::shared_ptr<CurveUnrolling> cu_cpy, std::shared_ptr<ShapeSignature> sig, std::shared_ptr<DisplayLock> display_lock) {
	if (cu_src == nullptr) {
		throw std::invalid_argument("Source curve can't be null!");
	}

	if (cu_cpy == nullptr) {
		throw std::invalid_argument("Copied curve can't be null!");
	}

	// Use fancy solver (CppOptLib)
	try {
		StrokeDiffReduce replicate(cu_src, cu_cpy, sig, display_lock, &index);
		StrokeDiffReduce::TVector x(3);
		StrokeDiffReduce::TVector g(3);
		x << 0.0, 0.0, 0.0;
		std::shared_ptr<LbfgsbSolver<StrokeDiffReduce>> solver = nullptr;
		
		solver = std::make_shared<LbfgsbSolver<StrokeDiffReduce>>();
		//solver->setDebug(cppoptlib::DebugLevel::High);			
		//std::cout << "StrokeDiffReduce::checkGradient(): " << (replicate.checkGradient(x) ? "True" : "False") << std::endl;
		solver->minimize(replicate, x);
		replicate.gradient(x, g);
		std::clog << "[ " << id << " ]: gradient = " << g.transpose() << std::endl;

		cu_cpy->transform(x(0), x(1), x(2));
	}
	catch (std::exception e) {
		std::clog << e.what() << std::endl;
		// pass
	}
}

StrokeDiffReduce::StrokeDiffReduce(std::shared_ptr<CurveUnrolling> source, std::shared_ptr<CurveUnrolling> copy, std::shared_ptr<ShapeSignature> sig, std::shared_ptr<DisplayLock> display_lock, int* id): _source(source), _copy(copy), _sig(sig), _display_lock(display_lock), _id(id) {
	if (_source == nullptr || _copy == nullptr) {
		throw std::invalid_argument("Source and/or copy strokes cannot be null!");
	}

	assert(_copy->unrolled_stroke()->blade_points().size() == _source->unrolled_stroke()->blade_points().size());

	if (sig == nullptr) {
		throw std::invalid_argument("Signature cannot be nullptr!");
	}
}

StrokeDiffReduce::~StrokeDiffReduce() {
}

double StrokeDiffReduce::value(const TVector &x) {
	if (x.cwiseAbs()(0) > upperBound()(0) ||
		x.cwiseAbs()(1) > upperBound()(1) ||
		x.cwiseAbs()(2) > upperBound()(2)) {
		return 1e10;
	}

	std::shared_ptr<CurveUnrolling> tcopy = _copy->clone();

	tcopy->transform(x(0), x(1), x(2));

	double val = _source->unrolled_on_origin_mesh()->compare(*tcopy->unrolled_on_origin_mesh(), _sig);

	//std::clog << "value at x [ " << x.transpose() << " ]: " << val << std::endl;

	return val;
}

// TODO: Compare with finite differencing (small change, 1e-5)
void StrokeDiffReduce::gradient(const TVector &args, TVector& grad) {
	grad << 0.0, 0.0, 0.0;
	
	/*if (args.cwiseAbs()(0) > upperBound()(0) ||
		args.cwiseAbs()(1) > upperBound()(1) ||
		args.cwiseAbs()(2) > upperBound()(2)) {
		return;
	}*/

	return cppoptlib::Problem<double>::gradient(args, grad);

	// x = ( x_s, y_s, theta )
	// Re: analytical derivative
	assert(_sig->sig_steps().size() == 1);
	assert(_copy->unrolled_stroke()->blade_points().size() == _source->unrolled_stroke()->blade_points().size());
	assert(_copy->unrolled_on_origin_mesh()->blade_points().size() == _source->unrolled_on_origin_mesh()->blade_points().size());

	

	std::shared_ptr<CurveUnrolling> tcopy = _copy->clone();

	assert(tcopy->unrolled_stroke()->blade_points().size() == _copy->unrolled_stroke()->blade_points().size());

	// These are offsets from origin, not from current position -- assure this transformation interprets it this way!
	tcopy->transform(args(0), args(1), args(2));

	assert(tcopy->unrolled_stroke()->blade_points().size() == _copy->unrolled_stroke()->blade_points().size());

	Eigen::VectorXd sig_values = _sig->get_signature_values(_sig->sig_steps()(0));

	// Get copied stroke transformed points in terms of the 2D parameterization
	Eigen::MatrixXd copy_2d(2, tcopy->unrolled_stroke()->blade_points().size());
	for (std::size_t i = 0; i < tcopy->unrolled_stroke()->blade_points().size(); ++i) {
		copy_2d.col(i) << tcopy->unrolled_stroke()->blade_points()[i].to_world(tcopy->parameterized_mesh()).topRows<2>();
	}

	// dE/dx = -2 * sum((o -c) * dc/dx)
	// These strokes must exist on a common mesh with the signature for comparison-- so use remapped strokes to origin
	Eigen::VectorXd dE_residual = -2.0 * _source->unrolled_on_origin_mesh()->per_point_diff(*tcopy->unrolled_on_origin_mesh(), _sig);

	std::vector<BarycentricCoord> copy_bcs = tcopy->unrolled_stroke()->blade_points();
	double N = static_cast<double>(copy_bcs.size());

	// In order to calculate the signature gradient of a face, we need the vertices making up that face
	// Origin mesh faces can have multiple mappings to the 2D unrolled parameterization, making us have multiple signature space normals for a single unique fid
	// For each fid, find n = (A, B, C) [signature space normal] of form z = Ax + By + C
	for (std::size_t i = 0; i < copy_bcs.size(); ++i) {
		const BarycentricCoord& bc = copy_bcs[i];

		Eigen::DenseIndex origin_fid = tcopy->fid_map().at(bc._fid);
		
		if (tcopy->faces().rows() <= bc._fid) {
			std::clog << "Malformed curve! Out of bounds bc._fid!" << std::endl;
			continue;
		}

		Eigen::VectorXi face = tcopy->faces().row(bc._fid).transpose();

		if (face(0) >= tcopy->vertices().rows() ||
			face(1) >= tcopy->vertices().rows() ||
			face(2) >= tcopy->vertices().rows()) { 
			std::clog << "Malformed curve! Out of bounds faces(i)!" << std::endl;
			continue;
		}

		// Find [A, B, C]^T
		Eigen::Matrix3d XYZ; XYZ << tcopy->vertices().row(face(0)).leftCols<2>(), 1.0,
									tcopy->vertices().row(face(1)).leftCols<2>(), 1.0,
									tcopy->vertices().row(face(2)).leftCols<2>(), 1.0;

		Eigen::Vector3d H; H << sig_values(face(0)), sig_values(face(1)), sig_values(face(2));
		Eigen::Vector3d n = XYZ.inverse() * H;

		// Curve point
		//Eigen::Vector2d cp = copy_2d(0, i);
		double x = copy_2d(0, i);
		double y = copy_2d(1, i);
		double theta = args(2);
		double A = n(0);
		double B = n(1);

		double cdx = A;
		double cdy = B;
		double cdtheta = A * (-1.0 * x * std::sin(theta) - y * std::cos(theta))
					   + B * (x * std::cos(theta) - y * std::sin(theta));

		grad(0) += dE_residual(i) * cdx;
		grad(1) += dE_residual(i) * cdy;
		grad(2) += dE_residual(i) * cdtheta;
	}

	grad /= N;
}

bool StrokeDiffReduce::callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x) {
	// Capture state of solver per step taken
	std::clog << "[ " << *_id << " ]: step: x = " << x.transpose() << " : value = " << value(x) << std::endl;

	// display if viewer isn't nullptr
	if (_display_lock != nullptr && _id != nullptr && _display_lock->_display.size() > *_id) {
		_display_lock->_mtx.lock();
		_display_lock->_display[*_id].push_back(_copy->unrolled_on_origin_mesh());
		_display_lock->_updated = true;
		_display_lock->_mtx.unlock();
	}

	return true;
}

StrokeDiffReduce::TVector StrokeDiffReduce::upperBound() const {
	return (TVector(3) << 5.0, 5.0, M_PI).finished();
}

StrokeDiffReduce::TVector StrokeDiffReduce::lowerBound() const {
	return (TVector(3) << -5.0, -5.0, -M_PI).finished();
}


StrokeTransfer::StrokeTransfer(std::shared_ptr<Mesh> mesh, std::shared_ptr<ShapeSignature> sig, igl::opengl::glfw::Viewer* viewer):
	_mesh(mesh), 
	_source(SurfaceStroke::instantiate(mesh)), 
	_sig(sig),
	_crsolver(std::make_shared<CRSolver>()),
	_viewer(viewer) {
	_crsolver->add_signature(_sig);
}

void StrokeTransfer::add_to_source(Eigen::DenseIndex fid, Eigen::Vector3d bc) {
	_source->add_curve_point(fid, bc);
}

std::vector<std::shared_ptr<SurfaceStroke>> StrokeTransfer::suggested_transfers(int num_to_suggest, int cull_sample, std::shared_ptr<SelfSimilarityMap>* self_sim_map) {
	// Find "Centers of Interest"	
	std::shared_ptr<DiscreteExponentialMap> dem;
	std::shared_ptr<Patch> curve_cover;

	_source->parameterized_space_points_2d(&curve_cover, &dem);

	std::clog << "curve_cover vertices: " << curve_cover->vertices().rows() << std::endl;
	std::clog << "curve_cover faces:    " << curve_cover->faces().rows() << std::endl;

	// EXPERIMENT: Submit each barycentric coordinate individually
	Eigen::VectorXd sig_values = _sig->get_signature_values(0.0);
	double include_sig_value = 0.0;
	//std::vector<Relation> rels;
	//std::vector<BarycentricCoord> source_curve_points;

	//for (BarycentricCoord& bc : source_curve_points) {
	//	include_sig_value += bc.to_sig_value(_mesh, sig_values).norm() / static_cast<double>(source_curve_points.size());
	//	_crsolver->add_relation(Relation(bc, Relation::Designation::Include));
	//}

	// Find opposing anchor exclude (All signatures so far flatten out to the same value at parameter extremes)
	//Eigen::DenseIndex max_diff_index;
	//(sig_values.array() - include_sig_value).abs().maxCoeff(&max_diff_index);

	//_crsolver->add_relation(Relation(Patch::instantiate(_mesh, max_diff_index), Relation::Designation::Exclude));

	// Find connected components of matching points
	//std::shared_ptr<SelfSimilarityMap> ssm = std::make_shared<SelfSimilarityMap>(_sig);

	//std::clog << "Number of similar vertices on mesh: " << ssm->similarity_ratings().sum() << std::endl;

	std::vector<std::shared_ptr<CurveUnrolling>> seeded_curves;
	//std::set<Eigen::DenseIndex> visited_vids;

	std::shared_ptr<CurveUnrolling> cu = std::make_shared<CurveUnrolling>(_source);
	
	// Determine number of curves to seed
	const int num_seeded_curves = 500;

	auto cmp = [](const std::pair<double, std::shared_ptr<SurfaceStroke>>& lhs, const std::pair<double, std::shared_ptr<SurfaceStroke>>& rhs) -> bool {
		return lhs.first > rhs.first;
	};
	auto ccmp = [](const std::pair<double, std::shared_ptr<CurveUnrolling>>& lhs, const std::pair<double, std::shared_ptr<CurveUnrolling>>& rhs) -> bool {
		return lhs.first > rhs.first;
	};
	cppoptlib::Problem<double>::TVector x(3);
	x << 0.0, 0.0, 0.0;

	// Select seed points at random (std::begin(curve) at face center)
	std::priority_queue<std::pair<double, std::shared_ptr<CurveUnrolling>>, std::vector<std::pair<double, std::shared_ptr<CurveUnrolling>>>, decltype(ccmp)> Q_curves(ccmp);
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> dis(0, _mesh->faces().rows() - 1);
		for (int i = 0; i < num_seeded_curves; ++i) {
			// Select starting face for curve start
			Eigen::DenseIndex fid = static_cast<Eigen::DenseIndex>(dis(gen));

			// Determine each curve's search parameter (based on the signature being used)
			try {
				std::shared_ptr<CurveUnrolling> seeded_curve = std::make_shared<CurveUnrolling>(_mesh, *cu, fid);
				StrokeDiffReduce replicate(cu, seeded_curve, _sig);

				double val = replicate.value(x);

				Q_curves.push(std::make_pair(val, seeded_curve));
			} catch (std::exception) {
				// pass
			}
		}
	}

	const int CULL_REMAIN_MAX = cull_sample;

	for (int i = 0; i < CULL_REMAIN_MAX && i < Q_curves.size(); ++i) {
		seeded_curves.push_back(Q_curves.top().second);
		Q_curves.pop();
	}

	// Solve curves in parallel 
	const unsigned int thread_pool_size = 10; // std::thread::hardware_concurrency();
	ctpl::thread_pool pool(thread_pool_size);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	std::shared_ptr<DisplayLock> display_lock = nullptr; //std::make_shared<DisplayLock>(num_seeded_curves);
	for (int i = 0; i < seeded_curves.size(); ++i) {
		pool.push(curve_solve_proxy, i, cu, seeded_curves[i], _sig, display_lock);
	}

	if (display_lock != nullptr) {
		while (pool.n_idle() < pool.size()) {
			bool clear = true;
			for (int i = 0; i < display_lock->_display.size(); ++i) {
				if (display_lock->_updated) {
					if (display_lock->_display[i].size() > 0) {
						for (int j = 0; j < display_lock->_display[i].size(); ++j) {
							display_lock->_mtx.lock();
							double alpha = static_cast<double>(j + 1) / static_cast<double>(display_lock->_display[i].size());
							display_lock->_display[i][j]->display(*_viewer, true, _mesh, clear, Eigen::Vector3d::Zero(), alpha * Eigen::Vector3d::UnitY() + (1.0 - alpha) * Eigen::Vector3d::UnitX());
							display_lock->_mtx.unlock();
							clear = false;
							std::this_thread::yield();
						}
					}
				}

				auto start = std::chrono::high_resolution_clock::now();
				auto end = start + std::chrono::milliseconds(250);
				do {
					std::this_thread::yield();
				} while (std::chrono::high_resolution_clock::now() < end);
			}

			display_lock->_updated = false;
		}
	}

	pool.stop(true);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time to Solve = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;

	// Merge overlapping curves, and cull curves that don't have a low enough final objective value
	const double curve_threshold = 1e-1;

	std::vector<std::pair<double, std::shared_ptr<SurfaceStroke>>> replicated;
	
	for (int i = 0; i < seeded_curves.size(); ++i) {
		StrokeDiffReduce replicate(cu, seeded_curves[i], _sig);
		
		double val = replicate.value(x);

		/*if (val > curve_threshold) {
			continue;
		}*/

		bool append = true;
		std::shared_ptr<SurfaceStroke> cpy = seeded_curves[i]->unrolled_on_origin_mesh();
		for (int j = 0; j < replicated.size(); ++j) {
			if (!cpy->is_disjoint_from(replicated[j].second)) {
				double prev_val = replicated[j].first;

				if (val < prev_val) {
					replicated[j].second = cpy;
				}

				append = false;
			}
		}

		if (append) {
			replicated.push_back(std::make_pair(val, cpy));
		}
	}

	std::priority_queue<std::pair<double, std::shared_ptr<SurfaceStroke>>, std::vector<std::pair<double, std::shared_ptr<SurfaceStroke>>>, decltype(cmp)> Q(cmp);
	for (auto kv : replicated) {
		Q.push(std::make_pair(kv.first, kv.second));
	}

	//const int top_n = 5;
	std::vector<std::shared_ptr<SurfaceStroke>> top;
	int i = 0;
	while (!Q.empty()) {
		if (i >= num_to_suggest) {
			break;
		}

		std::clog << "top [ " << i++ << " ]: " << Q.top().first << std::endl;
		top.push_back(Q.top().second);
		Q.pop();
	}

	std::clog << "Replicated strokes suggested [ " << top.size() << " ]" << std::endl;

	/*if (self_sim_map != nullptr) {
		*self_sim_map = ssm;
	}*/

	return top;
}