#ifndef STROKE_TRANSFER__
#define STROKE_TRANSFER__

#include <vector>
#include <memory>
#include <mutex>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <cppoptlib/problem.h>
#include <cereal/cereal.hpp>

#include <geometry/mesh.h>
#include <matching/constrained_relation_solver.h>
#include <matching/surface_stroke.h>
#include <matching/parameterization/curve_unrolling.h>

struct DisplayLock {
	DisplayLock(int size): _updated(false) {
		_display.resize(size);
	}

	bool _updated;
	std::vector<std::vector<std::shared_ptr<SurfaceStroke>>> _display;
	std::mutex _mtx;
};

class StrokeDiffReduce : public cppoptlib::Problem<double> {
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;

		StrokeDiffReduce(std::shared_ptr<CurveUnrolling> source, std::shared_ptr<CurveUnrolling> copy, std::shared_ptr<ShapeSignature> sig, std::shared_ptr<DisplayLock> display_lock = nullptr, int* id = nullptr);
		~StrokeDiffReduce();

		virtual double value(const TVector &x);
		virtual void gradient(const TVector &x, TVector& grad);	// Re: analytical derivative
		virtual TVector upperBound() const;
		virtual TVector lowerBound() const;

		virtual bool callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x);

	protected:
		std::shared_ptr<CurveUnrolling> _source;
		std::shared_ptr<CurveUnrolling> _copy;
		std::shared_ptr<ShapeSignature> _sig;

		std::shared_ptr<DisplayLock> _display_lock;
		int *_id;
};

// Defines the higher level interface for establishing a surface stroke and matching it across a mesh
class StrokeTransfer {
	public:
		enum class OptType : int {
			Naive,
			CppOptLib
		};

		StrokeTransfer(std::shared_ptr<Mesh> mesh, std::shared_ptr<ShapeSignature> sig, igl::opengl::glfw::Viewer* viewer = nullptr);

		void add_to_source(Eigen::DenseIndex fid, Eigen::Vector3d barycentric_coord);

		std::shared_ptr<SurfaceStroke> source() { return _source; }
		std::shared_ptr<ShapeSignature> signature() { return _sig; }

		std::vector<std::shared_ptr<SurfaceStroke>> suggested_transfers(int num_to_suggest, int cull_sample, std::shared_ptr<SelfSimilarityMap>* self_sim_map = nullptr);

	private:
		std::shared_ptr<Mesh> _mesh;
		std::shared_ptr<ShapeSignature> _sig;
		std::shared_ptr<SurfaceStroke> _source;
		std::shared_ptr<CRSolver> _crsolver;

		igl::opengl::glfw::Viewer* _viewer;
};

#endif