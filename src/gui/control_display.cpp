#include "control_display.h"

#include <random>

#include <matching/self_similarity/self_similarity_map.h>
#include <geometry/patch.h>
#include <geometry/segmentation/surface_by_numbers.h>
#include <attributes/transfer/meshmatch.h>
#include <matching/constrained_relation_solver.h>

#include <experiments/stroke_transfer_exp.h>
#include <cereal/archives/json.hpp>
#include <iostream>

ControlDisplay::ControlDisplay():
	_viewer(),
	_active_mesh(nullptr),
	_active_mesh_path(""),
	_active_mesh_texture_path(""),
	_crsolver(std::make_shared<CRSolver>()),
	_stroke_transfer(nullptr),
	_current_vertex_patch(nullptr),
	_click_mode(ClickMode::View),
	_prev_mouse_pos(0.0f, 0.0f),
	_mouse_down(false),
	_match_mode(MatchMode::Component),
	_SDFParams(),
	_active_set(1),
	_selected_component(nullptr),
	_mm_source(nullptr),
	_mm_target(nullptr),
	_cm(igl::COLOR_MAP_TYPE_VIRIDIS),
	_gamma(1.0),
	_eta(1.0),
	_num_suggested(5),
	_cull_sample(12),
	_menu() {

	// Attach a _menu plugin
	_viewer.plugins.push_back(&_menu);

	_viewer.core.background_color = Eigen::Vector4f::Ones();

	_menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent _menu content
		_menu.draw_viewer_menu();
	};

	// Draw additional windows
	_menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * _menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiSetCond_FirstUseEver);
		ImGui::Begin(
			"Constrained Thresholds", nullptr,
			ImGuiWindowFlags_NoSavedSettings
		);

		// Add new group
		if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Load OBJ
			if (ImGui::Button("Load OBJ", ImVec2(-1, 0))) {
				// Adapted from http://www.cplusplus.com/forum/windows/169960/

				char filename[MAX_PATH];

				OPENFILENAME ofn;
				ZeroMemory(&filename, sizeof(filename));
				ZeroMemory(&ofn, sizeof(ofn));
				ofn.lStructSize = sizeof(ofn);
				ofn.hwndOwner = NULL;
				ofn.lpstrFilter = "OBJ Files\0*.obj\0Any File\0*.*\0";
				ofn.lpstrFile = filename;
				ofn.nMaxFile = MAX_PATH;
				ofn.lpstrTitle = "Select an OBJ file to load.";
				ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

				if (GetOpenFileNameA(&ofn)) {
					_active_mesh_path = filename;

					load_mesh();
				}
			}

			// Load Texture
			if (ImGui::Button("Load Texture", ImVec2(-1, 0))) {
				char filename[MAX_PATH];

				OPENFILENAME ofn;
				ZeroMemory(&filename, sizeof(filename));
				ZeroMemory(&ofn, sizeof(ofn));
				ofn.lStructSize = sizeof(ofn);
				ofn.hwndOwner = NULL;
				ofn.lpstrFilter = "PNG Files\0*.png\0Any File\0*.*\0";
				ofn.lpstrFile = filename;
				ofn.nMaxFile = MAX_PATH;
				ofn.lpstrTitle = "Select a texture file to load.";
				ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

				if (GetOpenFileNameA(&ofn)) {
					_active_mesh_texture_path = filename;

					load_texture();
				}
			}

			if (ImGui::InputDouble("t", &_SDFParams._t, 0, 0, "%.6f")) {
				if (_SDFParams._sig != nullptr) {
					_SDFParams._t = std::max(_SDFParams._sig->param_lower_bound(), std::min(_SDFParams._sig->param_upper_bound(), _SDFParams._t));

					step_hks_t(0.0);
				}
			}

			ImGui::InputDouble("t step", &_SDFParams._tstep, 0, 0, "%.6f");

			if (ImGui::Button("Show Signature at t", ImVec2(-1, 0))) {
				if (_active_mesh != nullptr) {
					_SDFParams._sig = ShapeDiameterSignature::instantiate(_active_mesh, _SDFParams._t);

					_SDFParams._W_t = _SDFParams._sig->get_signature_values();

					_active_mesh->set_scalar_vertex_color(_viewer, _SDFParams._W_t, _SDFParams._W_t.minCoeff(), _SDFParams._W_t.maxCoeff(), _cm);  
				}
			}

			ImGui::Combo("Mouse Mode", (int *)(&_match_mode), "Vertex\0Patch\0Component\0\0");

			if (ImGui::Button("Clear Selection", ImVec2(-1, 0))) {
				if (_active_mesh != nullptr) {
					_active_mesh->deselect_all(_viewer);
					//_crsolver = std::make_shared<CRSolver>();
					//_crsolver->add_signature(_SDFParams._sig);
					_stroke_transfer = std::make_shared<StrokeTransfer>(_active_mesh, _SDFParams._sig, &_viewer);
					_current_vertex_patch = nullptr;
					_components.clear();
					_selected_component = nullptr;
					_mm_source = nullptr;
					_mm_target = nullptr;
					_viewer.data().clear();
					
					try {
						std::shared_ptr<SelfSimilarityMap> sim_map = std::make_shared<SelfSimilarityMap>(*_crsolver, _SDFParams._sig, _active_set);
						_active_mesh->set_scalar_vertex_color(_viewer, sim_map->similarity_ratings(), igl::COLOR_MAP_TYPE_VIRIDIS);
					} catch (std::exception e) {
						// Nothing to construct similarity map on
					}

					_active_mesh->display(_viewer);
				}
			}

			ImGui::Combo("Select Mode", (int *)(&_click_mode), "View\0Select\0Draw\0\0");

			if (ImGui::Button("Show Curve Unrolling", ImVec2(-1, 0))) {
				if (_active_mesh != nullptr) {
					// _active_mesh->show_patch_map(viewer, _SDFParams._sig);
					_active_mesh->show_patch_map(_viewer, true, _stroke_transfer->source(), _SDFParams._sig);
				}
			}

			if (ImGui::Button("Show Full Model DEM", ImVec2(-1, 0))) {
				if (_active_mesh != nullptr) {
					// _active_mesh->show_patch_map(viewer, _SDFParams._sig);
					_active_mesh->show_patch_map(_viewer, _SDFParams._sig);
				}
			}

			if (ImGui::Button("Include Patch", ImVec2(-1, 0))) {
				if (_active_mesh != nullptr) {
					if (_match_mode == MatchMode::Patch || _match_mode == MatchMode::Component) {
						std::shared_ptr<Patch> patch = _active_mesh->emit_selected_as_patch();

						if (patch == nullptr) {
							return true;
						}

						//_active_mesh->select_patch(viewer, patch, { 0.0, 0.0, 1.0, 1.0 });

						// EXPERIMENT: Submit each vertex of the patch individually
						std::vector<std::shared_ptr<Patch>> patches = patch->shatter();

						std::vector<Relation> rels; 
						for (auto p : patches) {
							_active_mesh->select_vertex(_viewer, *(p->vids().begin()), { 0.0, 1.0, 0.0 });

							_crsolver->add_relation(Relation(p, Relation::Designation::Include), _active_set);
						}
					}
					else if (_match_mode == MatchMode::Vertex) {
						if (_current_vertex_patch->vids().size() <= 0) {
							return true;
						}

						_active_mesh->select_vertex(_viewer, *(_current_vertex_patch->vids().begin()), { 0.0, 1.0, 0.0 });

						_crsolver->add_relation(Relation(_current_vertex_patch, Relation::Designation::Include), _active_set);
					}
					else {
						// Mode not supported
						return false;
					}

					std::shared_ptr<SelfSimilarityMap> sim_map = std::make_shared<SelfSimilarityMap>(*_crsolver, _SDFParams._sig, _active_set);

					_active_mesh->set_scalar_vertex_color(_viewer, sim_map->similarity_ratings(), igl::COLOR_MAP_TYPE_VIRIDIS);
				}
			}

			if (ImGui::Button("Exclude Patch", ImVec2(-1, 0))) {
				if (_match_mode == MatchMode::Patch || _match_mode == MatchMode::Component) {
					std::shared_ptr<Patch> patch = _active_mesh->emit_selected_as_patch();
					//_active_mesh->select_patch(viewer, patch, { 1.0, 0.0, 0.0, 1.0 });

					if (patch == nullptr) {
						return true;
					}

					// EXPERIMENT: Submit each vertex of the patch individually
					std::vector<std::shared_ptr<Patch>> patches = patch->shatter();

					std::vector<Relation> rels;
					for (auto p : patches) {
						_active_mesh->select_vertex(_viewer, *(p->vids().begin()), { 1.0, 0.0, 0.0 });

						_crsolver->add_relation(Relation(p, Relation::Designation::Exclude), _active_set);
					}
				}
				else if (_match_mode == MatchMode::Vertex) {
					if (_current_vertex_patch != nullptr) {
						if (_current_vertex_patch->vids().size() <= 0) {
							return true;
						}

						_active_mesh->select_vertex(_viewer, *(_current_vertex_patch->vids().begin()), { 1.0, 0.0, 0.0 });

						_crsolver->add_relation(Relation(_current_vertex_patch, Relation::Designation::Exclude), _active_set);
					}
				}
				else {
					// Mode not supported
					return false;
				}

				std::vector<double> indices;
				_crsolver->solve(indices, _active_set);

				std::shared_ptr<SelfSimilarityMap> sim_map = std::make_shared<SelfSimilarityMap>(*_crsolver, _SDFParams._sig, _active_set);

				_active_mesh->set_scalar_vertex_color(_viewer, sim_map->similarity_ratings(), igl::COLOR_MAP_TYPE_VIRIDIS);
			}

			if (ImGui::Button("Suggest Stroke", ImVec2(-1, 0))) {
				/*if (_suggest_stroke_thread.joinable()) {
					_suggest_stroke_thread.join();
				}

				_suggest_stroke_thread = std::thread([this]()
				{*/
					std::shared_ptr<SelfSimilarityMap> ssm;
					_suggested_strokes = _stroke_transfer->suggested_transfers(_num_suggested, _cull_sample, &ssm);

					if (ssm != nullptr) {
						_active_mesh->set_scalar_vertex_color(_viewer, ssm->similarity_ratings(), igl::COLOR_MAP_TYPE_VIRIDIS);
					}

					_viewer.data().points.resize(0, 0);
					_viewer.data().lines.resize(0, 0);
					
					for (int i = 0; i < _suggested_strokes.size() && i < _num_suggested; ++i) {
						_suggested_strokes[i]->display(_viewer, true, _active_mesh, i == 0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), _suggested_strokes[i]);
					}
				//});
			}

			ImGui::SliderInt("Sample to Solve", &_cull_sample, 1, 100);

			if (ImGui::SliderInt("Suggestions to Find", &_num_suggested, 1, 100)) {
				for (int i = 0; i < _num_suggested && i < _suggested_strokes.size(); ++i) {
					_suggested_strokes[i]->display(_viewer, true, _active_mesh, i == 0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), _suggested_strokes[i]);
				}
			}

			if (ImGui::Button("Save Experiment", ImVec2(-1, 0))) {
				std::ofstream saved_exp(_active_mesh->resource_dir() + "ste.json");
				cereal::JSONOutputArchive output(saved_exp); // stream to cout

				StrokeTransferExperiment ste(_stroke_transfer);

				output(cereal::make_nvp("Stroke Transfer Experiment", ste));
			}

			if (ImGui::Button("Save Geometry", ImVec2(-1, 0))) {
				if (_active_mesh != nullptr) {
					_active_mesh->write_obj(_active_mesh->resource_dir() + "\\mesh_dump.obj");	
				}
			}
		}

		if (ImGui::CollapsingHeader("Surfacing by Numbers", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::InputDouble("eta", &_eta, 0, 0, "%.6f")) {
				_eta = (_eta > 0.001) ? _eta : 0.001;
			}

			if (ImGui::InputDouble("gamma", &_gamma, 0, 0, "%.6f")) {
				_gamma = (_eta > 0.001) ? _gamma : 0.001;
			}
		}



		ImGui::End();
	};

	_viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int) -> bool {
		bool handled = false;

		double x = viewer.current_mouse_x;
		double y = viewer.core.viewport(3) - viewer.current_mouse_y;

		switch (_click_mode) {
			case ClickMode::Select: {
				if (_active_mesh != nullptr) {
					int fid;
					Eigen::Vector3f bc;

					if (_match_mode == MatchMode::Component) {
						if (_active_mesh->unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view, viewer.core.proj, viewer.core.viewport, fid, bc, true)) {
							_active_mesh->select_curve_cover(viewer, fid);
						}
					} else {
						// Cast a ray in the view direction starting from the mouse position
						if (_active_mesh->unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view, viewer.core.proj, viewer.core.viewport, fid, bc)) {
							if (_match_mode == MatchMode::Vertex) {
								Eigen::DenseIndex cvid = _active_mesh->closest_vertex_id(fid, bc);

								_current_vertex_patch = Patch::instantiate(_active_mesh, _active_mesh->closest_vertex_id(fid, bc), 0.0);

								_active_mesh->select_vid_with_point(viewer, cvid);
							} else if (_match_mode == MatchMode::Patch) {
								_active_mesh->select_contiguous_face(viewer, fid);
							}
						}
					}

					handled = true;
				}

				break;
			}
			default: {
				break;
			}
		}

		_prev_mouse_pos << x, y;
		_mouse_down = true;

		return handled;
	};

	_viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y) -> bool {
		bool handled = false;

		double x = viewer.current_mouse_x;
		double y = viewer.core.viewport(3) - viewer.current_mouse_y;

		if (_mouse_down && _match_mode == MatchMode::Patch) {
			switch (_click_mode) {
				case ClickMode::Select: {
					if (_active_mesh != nullptr) {
						// Select more mesh
						int fid;
						Eigen::Vector3f bc;

						// Cast a ray in the view direction starting from the mouse position
						if (_active_mesh->unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view, viewer.core.proj, viewer.core.viewport, fid, bc)) {
							_active_mesh->select_contiguous_face(viewer, fid);
						}

						handled = true;
					}

					break;
				}
				default: {
					break;
				}
			}

			_prev_mouse_pos << mouse_x, mouse_y;
		} 

		if (_mouse_down && _click_mode == ClickMode::Draw) {
			if (_active_mesh != nullptr) {
				// Draw on the mesh
				int fid;
				Eigen::Vector3f bc;

				// Cast a ray in the view direction starting from the mouse position
				if (_active_mesh->unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view, viewer.core.proj, viewer.core.viewport, fid, bc)) {
					_stroke_transfer->add_to_source(fid, bc.cast<double>());
					_stroke_transfer->source()->display(_viewer, true, nullptr, true, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), _stroke_transfer->source());
				}

				handled = true;
			}
		}

		return handled;
	};

	_viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer& viewer, int, int ) -> bool {
		bool handled = false;
		_mouse_down = false;

		if (_click_mode == ClickMode::Draw) {
			// TODO: Visualize suggestions as selectable patches
			//std::vector<std::shared_ptr<SurfaceStroke>> suggested_strokes = _stroke_transfer->suggested_transfers(_crsolver);
		}

		return handled;
	};

	// Keyboard callbacks
	_viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers) -> bool {
		static int sig_num = 0;

		if (key == '.') {
			return step_hks_t(_SDFParams._tstep);
		} else if (key == ',') {
			return step_hks_t(-1.0 * _SDFParams._tstep);
		} else if (key == 'q') { 
			if (_click_mode == ClickMode::View) {
				_click_mode = ClickMode::Select;
			} else if (_click_mode == ClickMode::Select) {
				_click_mode = ClickMode::Draw;
			} else {
				_click_mode = ClickMode::View;
			}
		} else if (key == 'w') {
			if (_match_mode == MatchMode::Vertex) {
				_match_mode = MatchMode::Patch;
			}
			else if (_match_mode == MatchMode::Patch) {
				_match_mode = MatchMode::Component;
			} else {
				_match_mode = MatchMode::Vertex;
			}
		} else if (key == 'k') {
			// Cycle different signature types
			sig_num = (sig_num + 1) % 2;

			switch (sig_num) {
				case 0: {
					_SDFParams._sig = ShapeDiameterSignature::instantiate(_active_mesh, 0.0);
					_SDFParams._t = 0.0;
					break;
				}
				case 1: {
					_SDFParams._sig = HeatKernelSignature::instantiate(_active_mesh, 1000);
					_SDFParams._t = 0.000143;
					break;
				}
				case 2: {
					_SDFParams._sig = WaveKernelSignature::instantiate(_active_mesh, 1000);
					_SDFParams._t = 0.000143;
				}
			}

			_SDFParams._W = _SDFParams._sig->get_signature_values();
			_SDFParams._maxVal = std::max(1.0, _SDFParams._sig->get_signature_values().maxCoeff());
			_SDFParams._minVal = std::min(0.0, _SDFParams._sig->get_signature_values().minCoeff());

			_crsolver = std::make_shared<CRSolver>();
			_crsolver->add_signature(_SDFParams._sig);
			return step_hks_t(0.0);
		} else if (key == 'c') {
			_cm = static_cast<igl::ColorMapType>((_cm + 1) % 6);
			return step_hks_t(0.0);
		} /*else if (key == '1' || key == '2' || key == '3' || key == '4') {
			std::cout << "Changing desire set to [ " << key << " ]!" << std::endl;

			std::stringstream ss; ss << key;
			_active_set = std::stoi(ss.str());

			_SDFParams._sig->set_active_set(_active_set);

			std::shared_ptr<SelfSimilarityMap> sim_map = std::make_shared<SelfSimilarityMap>(_SDFParams._sig);

			_active_mesh->set_scalar_vertex_color(_viewer, sim_map->similarity_ratings(), igl::COLOR_MAP_TYPE_VIRIDIS);
		}*/

		return true;
	};
}

bool ControlDisplay::load_texture() {
	if (_active_mesh != nullptr) {
		if (_active_mesh->load_texture(_active_mesh_texture_path)) {
			//_viewer.data.clear();

			// reload active mesh
			_active_mesh->display(_viewer);

			return true;
		}
	}

	return true;
}

ControlDisplay::~ControlDisplay() {
	if (_suggest_stroke_thread.joinable()) {
		_suggest_stroke_thread.join();
	}
}

void ControlDisplay::run() {
	_viewer.launch();
}

bool ControlDisplay::load_mesh() {
	if (_active_mesh_path.compare("") == 0) {
		return false;
	}

	std::shared_ptr<Mesh> loaded_mesh = nullptr;

	try {
		loaded_mesh = Mesh::instantiate(_active_mesh_path);
	} catch (std::exception e) {
		return false;
	}

	unload_mesh();

	_active_mesh = loaded_mesh;

	// load active mesh
	_active_mesh->display(_viewer);

	_crsolver = std::make_shared<CRSolver>();
	_SDFParams._sig = nullptr;

	if (!step_hks_t(0.0)) {
		return false;
	}

	_stroke_transfer = std::make_shared<StrokeTransfer>(_active_mesh, _SDFParams._sig, &_viewer);

	return true;
}

void ControlDisplay::unload_mesh() {
	_viewer.data().clear();

	_active_mesh = nullptr;
	_stroke_transfer = nullptr;

	_SDFParams = SDF();
}

bool ControlDisplay::step_hks_t(double step) {
	if (_active_mesh == nullptr) {
		return false;
	}

	if (_SDFParams._sig == nullptr) {
		_SDFParams._sig = ShapeDiameterSignature::instantiate(_active_mesh, 0.0);
		_SDFParams._W = _SDFParams._sig->get_signature_values();
		_SDFParams._maxVal = std::max(1.0, _SDFParams._sig->get_signature_values().maxCoeff());
		_SDFParams._minVal = std::min(0.0, _SDFParams._sig->get_signature_values().minCoeff());
		_SDFParams._t = 0.0;

		_crsolver->add_signature(_SDFParams._sig);
	} 

	if (_SDFParams._t + step < _SDFParams._sig->param_lower_bound()) {
		_SDFParams._t = _SDFParams._sig->param_lower_bound();
	} else if (_SDFParams._t + step > _SDFParams._sig->param_upper_bound()) {
		_SDFParams._t = _SDFParams._sig->param_upper_bound();
	} else {
		_SDFParams._t += step;
	}

	_SDFParams._sig->resample_at_param(_SDFParams._t);

	//_SDFParams._k = _SDFParams._sig->get_k_pairs_used();

	_SDFParams._W_t = _SDFParams._sig->get_signature_values();

	_active_mesh->set_scalar_vertex_color(_viewer, _SDFParams._W_t, _cm);

	return true;
}