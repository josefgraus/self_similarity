#include "surface_by_numbers.h"

#include <set>
#include <queue>

#include <maxflow/maxflow.h>

#include <geometry/mesh.h>
#include <geometry/patch.h>
#include <matching/threshold.h>

using maxflow::Graph_DDD;

SurfaceByNumbers::SurfaceByNumbers(const CRSolver& crsolver, double eta, double gamma) : _eta(eta), _gamma(gamma) {
	// For a given set of relations, determine the model segmentation via the algorithm presented in Section 3 of Surfacing by Numbers
	// All included patches will be used part of the source node, and all excluded part of the sink. 
	// In the case where we only have include patches, we choose the sink to be the furthest away signature-wise from the source
	if (crsolver.signatures().size() != 1) {
		throw std::domain_error("This implementation of Surface By Numbers currently only supports a single bound signature!");
	}

	std::shared_ptr<ShapeSignature> signature = crsolver.signatures()[0];

	if (!signature) {
		return;
	}

	const std::vector<Relation>& rels = crsolver.relations();
	_mesh = signature->origin_mesh();

	// Set up graph for a triangle mesh (worst case sizes)
	Graph_DDD graph(_mesh->vertices().size(), 3 * _mesh->faces().size());
	graph.add_node(_mesh->vertices().size());

	const trimesh::trimesh_t& halfedge_mesh = _mesh->halfedge();

	// Add edge weights
	double capacity = 0.0;

	// Calculate average edge length
	Eigen::DenseIndex v_count = _mesh->vertices().rows();
	Eigen::DenseIndex f_count = _mesh->faces().rows();

	double avg_edge_length = 0.0;
	unsigned int edge_count = 0;
	for (Eigen::DenseIndex fi = 0; fi < f_count; ++fi) {
		for (int vi = 0; vi < _mesh->faces().cols(); ++vi) {
			Eigen::DenseIndex i = _mesh->faces()(fi, vi);
			Eigen::DenseIndex j = _mesh->faces()(fi, (vi + 1) % _mesh->faces().cols());

			if (i >= j) {
				long he_index = halfedge_mesh.directed_edge2he_index(i, j);

				if (he_index >= 0 && halfedge_mesh.halfedge(he_index).opposite_he >= 0) {
					continue;
				}
			}

			avg_edge_length += (_mesh->vertices().row(i) - _mesh->vertices().row(j)).norm();
			edge_count++;
		}
	}

	if (edge_count != 0) {
		avg_edge_length /= edge_count;
	} else {
		throw std::logic_error("There are no edges in this mesh?!");
	}

	edge_count = 0;

	for (Eigen::DenseIndex fi = 0; fi < f_count; ++fi) {
		for (int vi = 0; vi < _mesh->faces().cols(); ++vi) {
			Eigen::DenseIndex i = _mesh->faces()(fi, vi);
			Eigen::DenseIndex j = _mesh->faces()(fi, (vi + 1) % _mesh->faces().cols());

			long he_index = halfedge_mesh.directed_edge2he_index(i, j);
			long ophe_index = halfedge_mesh.directed_edge2he_index(j, i);

			if (he_index < 0 && ophe_index < 0) {
				throw std::logic_error("There's no halfedge containing this face!");
			}

			if (i >= j) {
				if (!(he_index < 0 || ophe_index < 0)) {
					continue;
				}
			}

			double d_angle = dihedral_angle(_mesh, halfedge_mesh.halfedge(he_index).face, halfedge_mesh.halfedge(ophe_index).face);
			double length_ratio = (_mesh->vertices().row(i) - _mesh->vertices().row(j)).norm() / avg_edge_length;

			double w = _eta * (2 * M_PI - d_angle) / (2 * M_PI) + _gamma * length_ratio;

			graph.add_edge(i, j, w, w);
			edge_count++;
			capacity += w;
		}
	}

	// Add source/sink capacity
	std::vector<double> indices;
	std::shared_ptr<Threshold> inc_threshold = crsolver.solve(indices)[0];

	if (inc_threshold != nullptr) {
		std::vector<std::shared_ptr<Threshold>> exc_thresholds;
		Eigen::VectorXd vertex_sig = signature->get_signature_values(indices[0]);

		int exc_ctr = 0;
		double exc_min = std::numeric_limits<double>::max();
		double exc_max = std::numeric_limits<double>::lowest();
		for (const Relation& rel : rels) {
			if (rel._designation == Relation::Designation::Exclude) {
				for (Eigen::DenseIndex vid : rel._patch->vids()) {
					if (vertex_sig(vid) < exc_min) {
						exc_min = vertex_sig(vid);
					}

					if (vertex_sig(vid) > exc_max) {
						exc_max = vertex_sig(vid);
					}
				}

				exc_ctr++;
			}
		}

		if (inc_threshold->contains((Eigen::VectorXd(1) << exc_min).finished())) {
			exc_min = inc_threshold->max() + std::numeric_limits<double>::epsilon();
		}
		
		if (inc_threshold->contains((Eigen::VectorXd(1) << exc_max).finished())) {
			exc_max = inc_threshold->min() + std::numeric_limits<double>::epsilon();
		}

		exc_thresholds.push_back(std::make_shared<Threshold>(exc_min, exc_max));

		if (exc_thresholds[0]->contains((Eigen::VectorXd(1) << inc_threshold->midpoint()).finished())) {
			// Split exclude interval
			auto ex = exc_thresholds[0];
			exc_thresholds.clear();

			Eigen::VectorXd min_mid(1); min_mid << ((inc_threshold->min() - ex->min()) / 2.0) + ex->min();
			Eigen::VectorXd max_mid(1); max_mid << ((inc_threshold->max() - ex->max()) / 2.0) + inc_threshold->max();
			exc_thresholds.push_back(ex->padded_subinterval_about(min_mid));
			exc_thresholds.push_back(ex->padded_subinterval_about(max_mid));
		}

		if (exc_ctr > 0) {
			unsigned int source_nodes = 0;
			unsigned int sink_nodes = 0;
			for (Eigen::DenseIndex vi = 0; vi < vertex_sig.rows(); ++vi) {
				if (inc_threshold->contains(vertex_sig.row(vi))) {
					graph.add_tweights(vi, std::numeric_limits<double>::max(), 0.0);
					source_nodes++;
					continue;
				} 

				for (auto exc_threshold : exc_thresholds) {
					if (exc_threshold->contains(vertex_sig.row(vi))) {
						graph.add_tweights(vi, 0.0, std::numeric_limits<double>::max());
						sink_nodes++;
					}
				}
			}

			std::cout << "inc_threshold: (" << inc_threshold->min() << ", " << inc_threshold->max() << " );" << std::endl;
			for (auto exc_threshold : exc_thresholds) {
				std::cout << "exc_threshold: (" << exc_threshold->min() << ", " << exc_threshold->max() << " );" << std::endl;
			}

			std::cout << "source nodes: " << source_nodes << std::endl;
			std::cout << "sink nodes: " << sink_nodes << std::endl;
		}
	}

	// Solve max-flow/min-cut for graph to remove min-cut edges
	double flow = graph.maxflow();

	std::cout << "Max flow of graph: " << flow << std::endl;

	// Breadth-first search included components
	// Begin with any source connected node, and do a breadth-first search of it and its connected nodes to exhaustion, repeat until all nodes are explored
	// Each such iteration is a component of the mesh

	// Save components for query
	_segment_by_face = Eigen::VectorXi(f_count);

	std::set<Eigen::DenseIndex> inc;
	std::set<Eigen::DenseIndex> exc;
	for (const Relation& rel : rels) {
		if (rel._designation == Relation::Designation::Include) {
			inc.insert(rel._patch->vids().begin(), rel._patch->vids().end());
		} else if (rel._designation == Relation::Designation::Exclude) {
			exc.insert(rel._patch->vids().begin(), rel._patch->vids().end());
		}
	}

	for (Eigen::DenseIndex i = 0; i < f_count; ++i) {
		int source_ctr = 0;
		int sink_ctr = 0;

		for (Eigen::DenseIndex j = 0; j < _mesh->faces().cols(); ++j) {
			if (graph.what_segment(_mesh->faces()(i,j)) == Graph_DDD::SOURCE && exc.count(_mesh->faces()(i,j)) <= 0) {
				source_ctr++;
			} else {
				sink_ctr++;
			}
		}

		if (source_ctr > 0) {
			_segment_by_face(i) = -1;
		} else {
			_segment_by_face(i) = 0;
		}
	}

	int component = 1;

	while (true) {
		bool found = false;
		bool isolated = true;

		for (Eigen::DenseIndex fi = 0; fi < f_count; ++fi) {
			if (_segment_by_face(fi) > -1) {
				continue;
			} 

			found = true;

			std::set<Eigen::DenseIndex> deferred;
			std::queue<Eigen::DenseIndex> bfs_q;
			bfs_q.push(fi);

			while (!bfs_q.empty()) {
				Eigen::DenseIndex fid = bfs_q.front();

				if (deferred.count(fid) > 0) {
					bfs_q.pop();

					continue;
				}

				deferred.insert(fid);

				int source_vert = 0;

				for (int vi = 0; vi < _mesh->faces().cols(); ++vi) {
					Eigen::DenseIndex i = _mesh->faces()(fid, vi);
					Eigen::DenseIndex j = _mesh->faces()(fid, (vi + 1) % _mesh->faces().cols());

					if (graph.what_segment(i) == Graph_DDD::SOURCE) { 
						source_vert++;
					}

					long he_index = halfedge_mesh.directed_edge2he_index(i, j);
					long ophe_index = halfedge_mesh.directed_edge2he_index(j, i);

					if (he_index < 0 && ophe_index < 0) {
						throw std::logic_error("There's no halfedge containing this face!");
					}

					Eigen::DenseIndex f1 = halfedge_mesh.halfedge(he_index).face;
					Eigen::DenseIndex f2 = halfedge_mesh.halfedge(ophe_index).face;

					if (f1 != fid && f2 != fid) {
						continue;
					}

					if (f1 >= 0 && f1 != fid && _segment_by_face(f1) < 0 && deferred.count(f1) <= 0) {
						bfs_q.push(f1);
					} else if (f2 >= 0 && f2 != fid && _segment_by_face(f2) < 0 && deferred.count(f2) <= 0) {
						bfs_q.push(f2);
					}
				}

				if (source_vert > 1) {
					isolated = false;
				}

				if (!isolated) {
					for (auto def : deferred) {
						_segment_by_face(def) = component;
					}
					deferred.clear();
				}

				bfs_q.pop();
			}

			if (isolated) {
				for (auto def : deferred) {
					_segment_by_face(def) = 0;
				}
			}

			break;
		}

		if (!found) {
			break;
		} else {
			component++;
		}
	}
}

SurfaceByNumbers::~SurfaceByNumbers() {

}

const Eigen::VectorXi& SurfaceByNumbers::segment_by_face() const {
	return _segment_by_face;
}

std::vector<std::shared_ptr<Component>> SurfaceByNumbers::components() {
	std::vector<std::shared_ptr<Component>> comps;
	std::set<Eigen::DenseIndex> used = { 0 };

	// Quick and dirty way to create components from face assignments
	while (true) {
		Eigen::DenseIndex comp_nbr = -1;
		std::vector<int> fids;

		for (Eigen::DenseIndex i = 0; i < _segment_by_face.rows(); ++i) {
			if (comp_nbr < 0 && used.count(_segment_by_face[i]) <= 0) {
				comp_nbr = _segment_by_face[i];
			}

			if (_segment_by_face[i] == comp_nbr) {
				fids.push_back(i);
			}
		}

		if (comp_nbr > -1) {
			Eigen::Map<Eigen::VectorXi> v_fids(fids.data(), fids.size());

			comps.emplace_back(Component::instantiate(_mesh, v_fids));

			used.insert(comp_nbr);
		} else {
			break;
		}

		comp_nbr = -1;
	}

	return comps;
}

double SurfaceByNumbers::dihedral_angle(std::shared_ptr<Mesh> mesh, Eigen::DenseIndex f1, Eigen::DenseIndex f2) const {
	if (f1 < 0 && f2 < 0) {
		throw std::domain_error("Faces are out of range!");
	}

	if (f1 >= mesh->faces().size() || f2 >= mesh->faces().size()) {
		throw std::domain_error("Faces are out of range!");
	}

	const Eigen::MatrixXd V = mesh->vertices();
	Eigen::VectorXi face1 = (f1 < 0) ? mesh->faces().row(f2).transpose() : mesh->faces().row(f1).transpose();
	Eigen::VectorXi face2 = (f2 < 0) ? mesh->faces().row(f1).transpose() : mesh->faces().row(f2).transpose();

	if ((face1 - face2).sum() == 0) {
		// One of the faces is on a boundary, so return 90 degrees as its angle per the Surfacing by Numbers boundary condition
		return M_PI / 2.0;
	}

	if (face1.size() != 3 || face2.size() != 3) {
		throw std::domain_error("Faces are not triangles!");
	}

	Eigen::Vector3d e1 = (V.row(face1(0)) - V.row(face1(1))).block<1, 3>(0, 0).normalized().transpose();
	Eigen::Vector3d e2 = (V.row(face1(2)) - V.row(face1(1))).block<1, 3>(0, 0).normalized().transpose();

	Eigen::Vector3d face1_normal = e1.cross(e2).normalized();

	e1 = (V.row(face2(0)) - V.row(face2(1))).block<1, 3>(0, 0).normalized().transpose();
	e2 = (V.row(face2(2)) - V.row(face2(1))).block<1, 3>(0, 0).normalized().transpose();

	Eigen::Vector3d face2_normal = e1.cross(e2).normalized();

	Eigen::VectorXd face1_center = Eigen::VectorXd::Zero(V.row(face1(0)).cols());
	for (Eigen::DenseIndex i = 0; i < face1.size(); ++i) {
		face1_center += V.row(face1(i)).transpose();
	}
	face1_center /= face1.size();

	Eigen::VectorXd face2_center = Eigen::VectorXd::Zero(V.row(face2(0)).cols());
	for (Eigen::DenseIndex i = 0; i < face2.size(); ++i) {
		face2_center += V.row(face2(i)).transpose();
	}
	face2_center /= face2.size();

	return (1.0 - face1_normal.dot(face2_normal));
}