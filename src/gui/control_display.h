#ifndef CONTROL_DISPLAY_H_
#define CONTROL_DISPLAY_H_

#include <string>
#include <memory>
#include <thread>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <geometry/mesh.h>
#include <geometry/component.h>
#include <utilities/timer.h>
#include <matching/constrained_relation_solver.h>
#include <shape_signatures/heat_kernel_signature.h>
#include <shape_signatures/wave_kernel_signature.h>
#include <shape_signatures/shape_diameter_signature.h>
#include <matching/stroke_transfer.h>

class Patch;

class ControlDisplay {
	public:
		ControlDisplay();
		~ControlDisplay();

		void run();

	private:
		enum class ClickMode {
			View,
			Select,
			Draw
		};

		enum class MatchMode {
			Vertex,
			Patch,
			Component
		};

		bool load_mesh();
		void unload_mesh();
		bool load_texture();

		bool step_hks_t(double step);

		// Display
		igl::opengl::glfw::Viewer _viewer;
		igl::opengl::glfw::imgui::ImGuiMenu _menu;

		// Mouse picking controls
		ClickMode _click_mode;
		Eigen::Vector2f _prev_mouse_pos;
		bool _mouse_down;

		// Matching within a model
		MatchMode _match_mode;

		// Loaded mesh
		std::shared_ptr<Mesh> _active_mesh;
		std::string _active_mesh_path;
		std::string _active_mesh_texture_path;

		std::shared_ptr<Patch> _current_vertex_patch;

		// Matching
		std::shared_ptr<CRSolver> _crsolver;

		// Surface curve
		std::shared_ptr<StrokeTransfer> _stroke_transfer;

		// Shape Signatures
		struct HKS {
			HKS::HKS(): _k(300), _t(0.001f), _tstep(0.3), _sig(nullptr) { }
			HKS::HKS(double t, double tstep): _k(300), _t(t), _tstep(tstep), _sig(nullptr) { }
			HKS::HKS(int k, double t, double tstep): _k(k), _t(t), _tstep(tstep), _sig(nullptr) {}

			int _k;
			double _t;
			double _tstep;
			Eigen::MatrixXd _H;
			Eigen::MatrixXd _H_t;
			std::shared_ptr<HeatKernelSignature> _sig;
		} _HKSParams;

		struct WKS {
			WKS::WKS() : _k(300), _e(0.001f), _estep(0.3), _sig(nullptr) { }
			WKS::WKS(double e, double estep) : _k(300), _e(e), _estep(estep), _sig(nullptr) { }
			WKS::WKS(int k, double e, double estep) : _k(k), _e(e), _estep(estep), _sig(nullptr) {}

			int _k;
			double _e;
			double _estep;
			Eigen::MatrixXd _W;
			Eigen::MatrixXd _W_e;
			std::shared_ptr<WaveKernelSignature> _sig;
		} _WKSParams;

		struct SDF {
			SDF::SDF() : _t(0.001), _tstep(0.01), _sig(nullptr), _minVal(0.0), _maxVal(1.0) { }
			SDF::SDF(double t, double tstep) : _t(t), _tstep(tstep), _sig(nullptr), _minVal(0.0), _maxVal(1.0) { }

			double _t;
			double _tstep;
			double _minVal;
			double _maxVal;
			Eigen::MatrixXd _W;
			Eigen::MatrixXd _W_t;
			std::shared_ptr<ShapeSignature> _sig;
		} _SDFParams;

		// Desire set
		unsigned long _active_set;

		// Mesh color
		igl::ColorMapType _cm;

		// Segment by Numbers mesh segmentation
		double _gamma;
		double _eta;

		std::vector<std::shared_ptr<Component>> _components;
		std::shared_ptr<Component> _selected_component;

		std::shared_ptr<Component> _mm_source;
		std::shared_ptr<Component> _mm_target;

		std::thread _suggest_stroke_thread;
		std::vector<std::shared_ptr<SurfaceStroke>> _suggested_strokes;
		int _num_suggested;
		int _cull_sample;
};

#endif