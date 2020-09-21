#include "spectral_clustering.h"

#include <set>
#include <array>
#include <algorithm>

#include <Eigen/Sparse>
#include <igl/eigs.h>

#include <algorithms/shortest_path.h>

using namespace shortest_path;

SpectralClustering::SpectralClustering(std::shared_ptr<Geometry> geometry, unsigned int k, double delta, double eta):
	_delta(delta),
	_eta(eta) {

	if (geometry == nullptr) {
		return;
	}

	// Generate affinity matrix from mesh faces
	Eigen::MatrixXd W = affinity_matrix(geometry);

	// Degree matrix
	Eigen::DiagonalMatrix<double, -1> Ds = W.rowwise().sum().cwiseInverse().cwiseSqrt().asDiagonal();

	// Graph Laplacian
	Eigen::MatrixXd L = Ds * (W * Ds);

	Eigen::MatrixXd M = Eigen::MatrixXd::Identity(L.rows(), L.cols());

	Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(L, M, Eigen::ComputeEigenvectors);
	Eigen::MatrixXd U = es.eigenvectors().real().rightCols(k);

	// kmeans clustering
	Eigen::MatrixXd V = U.colwise().normalized();
	Eigen::MatrixXd Q = V * V.transpose();

	// Group faces by clusters
	Eigen::VectorXi guess = initial_guess(Q, k);

	_segment_by_face = k_means_lloyds(guess, V, k);
}

SpectralClustering::~SpectralClustering() {
}

const Eigen::VectorXi& SpectralClustering::segment_by_face() const {
	return _segment_by_face;
}

Eigen::MatrixXd SpectralClustering::affinity_matrix(std::shared_ptr<Geometry> geometry) {
	Eigen::SparseMatrix<double> G(geometry->faces().rows(), geometry->faces().rows());
	Eigen::SparseMatrix<double> A(geometry->faces().rows(), geometry->faces().rows());

	std::set<Eigen::DenseIndex> eta_list;
	unsigned int num_adj = 0;
	const Eigen::SparseMatrix<int>& adj = geometry->adjacency_matrix();
	const trimesh::trimesh_t& halfedge_mesh = geometry->halfedge();

	unsigned int count = 0;

	std::vector<Eigen::Triplet<double>> g_triplets;
	std::vector<Eigen::Triplet<double>> a_triplets;
	Eigen::DenseIndex v_count = geometry->vertices().rows();
	for (Eigen::DenseIndex i = 0; i < v_count; ++i) {
		for (Eigen::DenseIndex j = i + 1; j < v_count; ++j) {
			trimesh::index_t he_index = halfedge_mesh.directed_edge2he_index(i, j);

			if (he_index < 0) {
				// Invalid vertex pair??
				continue;
			}

			const trimesh::trimesh_t::halfedge_t& he = halfedge_mesh.halfedge(he_index);
			const trimesh::trimesh_t::halfedge_t& op_he = halfedge_mesh.halfedge(he.opposite_he);

			if (he.face < 0 || op_he.face < 0) {
				// This edge is on a boundary!
				continue;
			}

			Eigen::DenseIndex r = he.face;
			Eigen::DenseIndex s = op_he.face;

			Eigen::VectorXi face1 = geometry->faces().row(r);
			Eigen::VectorXi face2 = geometry->faces().row(s);

			double gd = geodesic_distance({ he.to_vertex, op_he.to_vertex }, face1, face2, geometry->vertices());
			double ad = angular_distance(face1, face2, geometry->vertices());

			g_triplets.emplace_back(Eigen::Triplet<double>(r, s, gd));
			g_triplets.emplace_back(Eigen::Triplet<double>(s, r, gd));
			a_triplets.emplace_back(Eigen::Triplet<double>(r, s, ad));
			a_triplets.emplace_back(Eigen::Triplet<double>(s, r, ad));

			++count;
		}
	}

	double geodesic_avg = 0.0;
	double angular_avg = 0.0;

	G.setFromTriplets(g_triplets.begin(), g_triplets.end());
	A.setFromTriplets(a_triplets.begin(), a_triplets.end());

	if (count > 0) {
		geodesic_avg = G.sum() / static_cast<double>(2 * count);
		angular_avg = A.sum() / static_cast<double>(2 * count);
	}

	G = G * (_delta / geodesic_avg);
	A = A * ((1.0 - _delta) / angular_avg);

	Eigen::MatrixXd W = floyd_warshall(G + A);

	std::vector<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>> inf_entries;
	for (Eigen::DenseIndex i = 0; i < W.rows(); ++i) {
		for (Eigen::DenseIndex j = 0; j < W.cols(); ++j) {
			if (W(i, j) > std::numeric_limits<double>::max()) {
				W(i, j) = 0.0;
				inf_entries.emplace_back(std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(i, j));
			}
		}
	}

	double sigma = W.sum() / (std::pow(geometry->faces().rows(), 2.0));
	double den = 2 * std::pow(sigma, 2.0);
	W = -1.0 * W / den;

	for (Eigen::DenseIndex i = 0; i < W.rows(); ++i) {
		for (Eigen::DenseIndex j = 0; j < W.cols(); ++j) {
			W(i, j) = std::exp(W(i, j));
		}
	}

	for (auto it : inf_entries) {
		W(it.first, it.second) = 0.0;
	}

	for (Eigen::DenseIndex i = 0; i < W.rows(); ++i) {
		W(i, i) = 1.0;
	}

	return W;
}

double SpectralClustering::geodesic_distance(std::array<Eigen::DenseIndex,2> edge_verts, const Eigen::VectorXi& face1, const Eigen::VectorXi& face2, const Eigen::MatrixXd& V) const {
	if (face1.size() <= 0 || face2.size() <= 0) {
		return 0.0;
	}

	Eigen::VectorXd edge_center = (V.row(edge_verts[0]) + V.row(edge_verts[1])).transpose() / 2.0;

	Eigen::VectorXd face1_center = Eigen::VectorXd::Zero(edge_center.rows());
	for (Eigen::DenseIndex i = 0; i < face1.size(); ++i) {
		face1_center += V.row(face1(i)).transpose();
	}
	face1_center /= face1.size();

	Eigen::VectorXd face2_center = Eigen::VectorXd::Zero(edge_center.rows());
	for (Eigen::DenseIndex i = 0; i < face2.size(); ++i) {
		face2_center += V.row(face2(i)).transpose();
	}
	face2_center /= face2.size();

	return (edge_center - face1_center).norm() + (edge_center - face2_center).norm();
}

double SpectralClustering::angular_distance(const Eigen::VectorXi& face1, const Eigen::VectorXi& face2, const Eigen::MatrixXd& V) const {
	if (face1.size() != 3 || face2.size() != 3) {
		return 0.0;
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

	bool use_eta = face1_normal.dot((face2_center - face1_center).block<3, 1>(0, 0).normalized()) < 0.0;
	
	return ((use_eta) ? _eta : 1.0) * (1.0 - face1_normal.dot(face2_normal));
}

Eigen::VectorXi SpectralClustering::k_means_lloyds(const Eigen::VectorXi& guess, const Eigen::MatrixXd& V, unsigned int k) {
	// guess
	Eigen::VectorXi means = guess; //initial_guess(V, k);
	Eigen::VectorXi prev_means = means;

	std::cout << means << std::endl; 

	Eigen::VectorXi assignment = Eigen::VectorXi::Constant(V.rows(),-1);
	Eigen::VectorXi next_assignment = assignment;

	double stop_threshold = 0.001 * means.rowwise().norm().minCoeff();

	do {
		// assign
		for (Eigen::DenseIndex i = 0; i < V.rows(); ++i) {
			double min_dist = std::numeric_limits<double>::max();
			Eigen::DenseIndex min_index = -1;

			for (Eigen::DenseIndex j = 0; j < means.rows(); ++j) {
				double dist = (V.row(means(j)) - V.row(i)).norm();

				if (dist < min_dist) {
					min_dist = dist;
					min_index = j;
				}
			}

			if (min_index < 0) {
				// There was no nearest centroid?? Something is wrong..
				throw std::logic_error("Finding nearest centroid failed??");
			}

			next_assignment(i) = min_index;
		}

		if ((assignment - next_assignment).sum() == 0) {
			// converged
			std::cout << "Converged!" << std::endl;
			break;
		}

		// update
		assignment = next_assignment;

		Eigen::MatrixXd numeric_means = Eigen::MatrixXd::Zero(means.rows(), V.cols());
		Eigen::VectorXi c_count = Eigen::VectorXi::Zero(means.rows());
		means = Eigen::VectorXi::Zero(k);

		for (Eigen::DenseIndex i = 0; i < assignment.rows(); ++i) {
			numeric_means.row(assignment(i)) += V.row(i);
			c_count(assignment(i))++;
		}

		for (Eigen::DenseIndex i = 0; i < c_count.rows(); ++i) {
			if (c_count(i) > 0) {
				numeric_means.row(i) /= c_count(i);
			}
		}

		// Find nearest vector in V closest to each numeric mean
		for (Eigen::DenseIndex i = 0; i < numeric_means.rows(); ++i) {
			double min_dist = std::numeric_limits<double>::max();
			Eigen::DenseIndex min_index = -1;

			for (Eigen::DenseIndex j = 0; j < V.rows(); ++j) {
				double dist = (V.row(j) - numeric_means.row(i)).norm();

				if (dist < min_dist) {
					min_dist = dist;
					min_index = j;
				}
			}

			if (min_index < 0) {
				// There was no nearest vector?? Something is wrong..
				throw std::logic_error("Finding nearest point to centroid failed??");
			}

			means(i) = min_index;
		}

		prev_means = means;

	} while (true);

	std::cout << means << std::endl;

	return next_assignment;
}

Eigen::VectorXi SpectralClustering::initial_guess(const Eigen::MatrixXd& Q, unsigned int k) {
	//"""Computes an initial guess for the cluster-centers"""
	int n = Q.rows();
	double min_value = std::numeric_limits<double>::max();
	std::array<int, 2> min_indices = { -1, -1 };

	for (int i = 0; i < Q.rows(); ++i) {
		for (int j = 0; j < Q.cols(); ++j) {
			if (i != j && Q(i, j) < min_value) {
				min_value = Q(i, j);
				min_indices = { i, j };
			}
		}
	}
	
	std::set<int> chosen = { min_indices[0], min_indices[1] };

	while (chosen.size() < k) {
		double min_max = std::numeric_limits<double>::max();
		double cur_max = 0.0;
		int new_index = -1;

		for (Eigen::DenseIndex i = 0; i < n; ++i) {
			if (chosen.count(i) <= 0) {
				cur_max = std::numeric_limits<double>::min();
				for (auto c : chosen) {
					if (cur_max < Q(c, i)) {
						cur_max = Q(c, i);
					}
				}

				if (min_max - cur_max > 1e-6) {
					min_max = cur_max;
					new_index = i;
				}
			}
		}

		chosen.insert(new_index);
	}

	Eigen::VectorXi guess(chosen.size());
	Eigen::DenseIndex j = 0;
	for (auto c : chosen) {
		guess(j++) = c;
	}

	return guess;
}