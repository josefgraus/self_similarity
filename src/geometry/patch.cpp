#include "patch.h"

#include <queue>
#include <map>
#include <memory>
#include <limits>
#include <stack>
#include <iostream>
#include <algorithm>

#include <igl/adjacency_matrix.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

#include <Eigen\Dense>
#include <Eigen\Sparse>
#include <Eigen\Geometry>

//#include <trimesh.h>
#include <geometry/mesh.h>
#include <matching/geodesic_fan.h>


struct PatchInstancer : public Patch, std::enable_shared_from_this<Patch> { 
	PatchInstancer(std::shared_ptr<Mesh> origin_mesh) : Patch(origin_mesh) {}
	PatchInstancer(std::shared_ptr<Mesh> origin_mesh, Eigen::DenseIndex center_vid) : Patch(origin_mesh, center_vid) {}
	PatchInstancer(std::shared_ptr<Mesh> mesh, const Eigen::VectorXi& fids) : Patch(mesh, fids) {}
	PatchInstancer(std::shared_ptr<Mesh> mesh, Eigen::DenseIndex center_vid, double geodesic_radius) : Patch(mesh, center_vid, geodesic_radius) {}
};

struct Vertex {
	Vertex(Eigen::DenseIndex vid, double path_length) : _vid(vid), _path_length(path_length) { };
	Eigen::DenseIndex _vid;
	double _path_length;
};

struct DEMPoint {
	double _x;
	double _y;
};

class PatchCompare {
	public:
		bool operator()(std::pair<double, int> left, std::pair<double, int> right) { return left.first > right.first; };
};

Patch::Patch(std::shared_ptr<Mesh> origin_mesh): 
	_origin_mesh(origin_mesh) {
}

Patch::Patch(std::shared_ptr<Mesh> origin_mesh, Eigen::DenseIndex vid):
	_origin_mesh(origin_mesh) {
	// Special constructor to create a patch containing only a single vertex
	_vids.insert(vid);

	reindex_submatrices();
}

Patch::Patch(std::shared_ptr<Mesh> origin_mesh, const Eigen::VectorXi& fids):
	_origin_mesh(origin_mesh) {

	const Eigen::MatrixXi& faces = _origin_mesh->faces();

	for (Eigen::DenseIndex i = 0; i < fids.size(); ++i) {
		const Eigen::DenseIndex fid = static_cast<Eigen::DenseIndex>(fids(i));

		_fids.insert(fid);

		for (Eigen::DenseIndex k = 0; k < faces.cols(); ++k) {
			_vids.insert(static_cast<Eigen::DenseIndex>(faces(fid, k)));
		}
	}

	reindex_submatrices();
}

Patch::Patch(std::shared_ptr<Mesh> origin_mesh, Eigen::DenseIndex center_vid, double geodesic_radius):
	_origin_mesh(origin_mesh) {
	if (!_origin_mesh->loaded()) {
		return;
	}

	// Run BFS with max Euclidean distance termination
	const Eigen::MatrixXd& V = _origin_mesh->vertices();
	const Eigen::SparseMatrix<int>& adj = _origin_mesh->adjacency_matrix();

	std::queue<std::shared_ptr<Vertex>> q;

	q.push(std::make_shared<Vertex>(center_vid, 0.0));
	_vids.insert(center_vid);

	while (!q.empty()) {
		std::shared_ptr<Vertex> p = q.front();
		q.pop();

		for (Eigen::SparseMatrix<int>::InnerIterator it(adj, static_cast<int>(p->_vid)); it; ++it) {
			Eigen::DenseIndex neighbor = it.row();   // neighbor vid

			if (_vids.count(neighbor) == 0) {
				double dist = (V.row(p->_vid) - V.row(neighbor)).norm() + p->_path_length;

				if (dist < geodesic_radius) {
					q.push(std::make_shared<Vertex>(neighbor, dist));
					_vids.insert(neighbor);
				}
			}
		}
	}

	// Found all the vertex ids within range, now select faces containing only selected vertices
	const Eigen::MatrixXi& faces = _origin_mesh->faces();
	std::set<Eigen::DenseIndex> discovered_vids;

	for (Eigen::DenseIndex i = 0; i < faces.rows(); ++i) {
		bool include = false;

		for (Eigen::DenseIndex j = 0; j < faces.cols(); ++j) {
			if (_vids.count(faces(i, j)) > 0) {
				include = true;

				// Insert all other vids of the face
				for (Eigen::DenseIndex k = 0; k < faces.cols(); ++k) {
					discovered_vids.insert(faces(i, k));
				}

				break;
			}
		}

		if (include) {
			_fids.insert(i);
		}
	}

	_vids.insert(discovered_vids.begin(), discovered_vids.end());

	reindex_submatrices();
}

Patch::~Patch() {
}

std::shared_ptr<Patch> Patch::instantiate(std::shared_ptr<Mesh> origin) {
	return std::make_shared<PatchInstancer>(origin);
}

std::shared_ptr<Patch> Patch::instantiate(std::shared_ptr<Mesh> origin, Eigen::DenseIndex center_vid) {
	return std::make_shared<PatchInstancer>(origin, center_vid);
}

std::shared_ptr<Patch> Patch::instantiate(std::shared_ptr<Mesh> origin, const Eigen::VectorXi& fids) {
	return std::make_shared<PatchInstancer>(origin, fids);
}

std::shared_ptr<Patch> Patch::instantiate(std::shared_ptr<Mesh> origin, Eigen::DenseIndex center_vid, double geodesic_radius) {
	return std::make_shared<PatchInstancer>(origin, center_vid, geodesic_radius);
}

void Patch::reindex_submatrices() {
	if (_origin_mesh == nullptr || _vids.size() <= 0) {
		return;
	}

	_vertices = Eigen::MatrixXd(_vids.size(), _origin_mesh->vertices().cols());
	_vertex_normals = Eigen::MatrixXd(_vids.size(), _origin_mesh->vertex_normals().cols());
	_faces = Eigen::MatrixXi(_fids.size(), _origin_mesh->faces().cols());
	_adj = Eigen::SparseMatrix<int>(_vids.size(), _vids.size());

	std::vector<Eigen::DenseIndex> vid_order(_vids.begin(), _vids.end());
	_remapping_vid.clear();
	_remapping_vid_rev.clear();
	_remapping_fid.clear();
	_remapping_fid_rev.clear();

	// Gather up vertex data
	for (Eigen::DenseIndex i = 0; i < vid_order.size(); ++i) {
		_vertices.row(i) = _origin_mesh->vertices().row(vid_order[i]);
		_vertex_normals.row(i) = _origin_mesh->vertex_normals().row(vid_order[i]);
		_remapping_vid[vid_order[i]] = i;
		_remapping_vid_rev[i] = vid_order[i];
	}

	// Reindex faces
	std::vector<Eigen::DenseIndex> fid_order(_fids.begin(), _fids.end());

	for (Eigen::DenseIndex i = 0; i < fid_order.size(); i++) {
		for (Eigen::DenseIndex j = 0; j < _faces.cols(); j++) {
			_faces(i, j) = _remapping_vid[_origin_mesh->faces()(fid_order[i], j)];
			_remapping_fid[fid_order[i]] = i;
			_remapping_fid_rev[i] = fid_order[i];
		}
	}

	if (_faces.rows() > 0) {
		// Build adjacency matrix
		igl::adjacency_matrix(_faces, _adj);

		// Create halfedge structure
		if (_faces.cols() == 3) {
			// Create halfedge structure
			std::vector<trimesh::triangle_t> triangles;
			triangles.resize(_faces.rows());

			for (Eigen::DenseIndex i = 0; i < _faces.rows(); ++i) {
				for (Eigen::DenseIndex j = 0; j < 3; ++j) {
					triangles[i].v[j] = _faces(i, j);
				}
			}

			std::vector<trimesh::edge_t> edges;
			trimesh::unordered_edges_from_triangles(triangles.size(), &triangles[0], edges);

			_halfedge_patch.build(_vertices.rows(), triangles.size(), &triangles[0], edges.size(), &edges[0]);
		}
	}
}

std::shared_ptr<Patch> Patch::one_ring(std::shared_ptr<Mesh> mesh, Eigen::DenseIndex center_vid) {
	if (center_vid < 0 || center_vid > mesh->vertices().rows()) {
		return nullptr;
	}

	std::shared_ptr<Patch> patch = Patch::instantiate(mesh);

	patch->_vids.insert(center_vid);

	const Eigen::SparseMatrix<int>& adj = mesh->adjacency_matrix();

	for (Eigen::SparseMatrix<int>::InnerIterator it(adj, static_cast<int>(center_vid)); it; ++it) {
		Eigen::DenseIndex neighbor = it.row();   // neighbor vid

		patch->_vids.insert(neighbor);
	}

	// Found all the vertex ids within range, now select faces containing only selected vertices
	const Eigen::MatrixXi& faces = mesh->faces();

	for (Eigen::DenseIndex i = 0; i < faces.rows(); ++i) {
		bool include = true;

		for (Eigen::DenseIndex j = 0; j < faces.cols(); ++j) {
			if (patch->_vids.count(faces(i, j)) <= 0) {
				include = false;
				break;
			}
		}

		if (include) {
			patch->_fids.insert(i);
		}
	}

	return patch;
}

std::vector<Eigen::DenseIndex> Patch::one_ring(Eigen::DenseIndex center_vid) {
	std::vector<Eigen::DenseIndex> ring;

	if (_remapping_vid.find(center_vid) == _remapping_vid.end()) {
		return ring;
	}

	Eigen::DenseIndex patch_vid = _remapping_vid[center_vid];

	ring.push_back(center_vid);

	for (Eigen::SparseMatrix<int>::InnerIterator it(_adj, static_cast<int>(patch_vid)); it; ++it) {
		Eigen::DenseIndex neighbor = it.row();   // neighbor vid

		ring.push_back(vid_to_origin_mesh(neighbor));
	}

	return ring;
}

std::vector<Eigen::DenseIndex> Patch::valence_faces(Eigen::DenseIndex vid) {
	std::vector<Eigen::DenseIndex> valence;

	for (Eigen::DenseIndex i = 0; i < _faces.rows(); ++i) {
		for (Eigen::DenseIndex j = 0; j < _faces.cols(); ++j) {
			if (vid == _faces(i, j)) {
				valence.push_back(i);
			}
		}
	}

	return valence;
}

const Eigen::MatrixXd& Patch::vertices() const {
	return _vertices;
}

const Eigen::MatrixXd& Patch::vertex_normals() const {
	return _vertex_normals;
}

const Eigen::MatrixXi& Patch::faces() const {
	return _faces;
}

const Eigen::SparseMatrix<int>& Patch::adjacency_matrix() const {
	return _adj;
}

const trimesh::trimesh_t& Patch::halfedge() {
	return _halfedge_patch;
}

bool Patch::add(Eigen::DenseIndex fid) {
	std::set<Eigen::DenseIndex> fids = { fid };

	return add(fids);
}

bool Patch::add(std::set<Eigen::DenseIndex> fids) {
	bool added = false;

	for (auto fid : fids) {
		auto ret = _fids.insert(fid);

		if (ret.second) {
			const Eigen::MatrixXi& faces = _origin_mesh->faces();

			for (Eigen::DenseIndex i = 0; i < faces.cols(); ++i) {
				_vids.insert(static_cast<Eigen::DenseIndex>(faces(fid, i)));
			}

			added = ret.second;
		}
	}

	if (added) {
		reindex_submatrices();
	}

	return added;
}

bool Patch::remove(Eigen::DenseIndex fid) {
	auto ret = _fids.erase(fid);

	if (ret > 0) {
		// TODO: Waaay too expensive for a removal -- consider another way to keep the vertex ids consistent with the face ids
		_vids.clear();

		const Eigen::MatrixXi& faces = _origin_mesh->faces();

		for (auto fid = _fids.cbegin(); fid != _fids.cend(); ++fid) {
			for (Eigen::DenseIndex k = 0; k < faces.cols(); ++k) {
				_vids.insert(static_cast<Eigen::DenseIndex>(faces(*fid, k)));
			}
		}

		reindex_submatrices();
	}

	return (ret > 0);
}

bool Patch::contains(Eigen::DenseIndex fid) {
	for (auto face : _fids) {
		if (fid == face) {
			return true;
		}
	}
	
	return false;
}

bool Patch::is_disjoint_from(std::shared_ptr<Patch> other) const {
	for (auto it = other->_vids.cbegin(); it != other->_vids.cend(); ++it) {
		if (_vids.count(*it) > 0) {
			return false;
		}
	}

	return true;
}

Eigen::DenseIndex Patch::get_centroid_vid() const {
	Eigen::Vector3d centroid(0.0, 0.0, 0.0);

	for (Eigen::DenseIndex i = 0; i < _vertices.rows(); ++i) {
		centroid += _vertices.block<1, 3>(i, 0).transpose() / static_cast<double>(_vertices.rows());
	}

	return nearest_vid(centroid);
}

Eigen::DenseIndex Patch::get_centroid_vid_on_origin_mesh() const {
	Eigen::DenseIndex centroid_vid = get_centroid_vid();

	for (auto p : _remapping_vid) {
		if (p.second == centroid_vid) {
			return p.first;
		}
	}

	return -1;
}

Eigen::DenseIndex Patch::nearest_vid(Eigen::VectorXd point) const {
	Eigen::DenseIndex nearest_vid = -1;
	double dist = std::numeric_limits<double>::max();

	for (Eigen::DenseIndex i = 0; i < _vertices.rows(); ++i) {
		double vDist = (_vertices.block<1, 3>(i, 0).transpose() - point).norm();

		if (vDist < dist) {
			dist = vDist;
			nearest_vid = i;
		}
	}

	return nearest_vid;
}

Eigen::DenseIndex Patch::get_centroid_fid() const {
	if (_faces.size() == 0) {
		return -1;
	}

	Eigen::DenseIndex centroid_vid = get_centroid_vid();

	Eigen::Vector3d centroid(0.0, 0.0, 0.0);

	for (Eigen::DenseIndex i = 0; i < _vertices.rows(); ++i) {
		centroid += _vertices.block<1, 3>(i, 0).transpose() / static_cast<double>(_vertices.rows());
	}

	std::vector<std::pair<Eigen::DenseIndex, Eigen::Vector3d>> face_centers;

	for (Eigen::DenseIndex i = 0; i < _faces.rows(); ++i) {
		bool contains_center = false;
		
		for (Eigen::DenseIndex j = 0; j < _faces.cols(); ++j) {
			if (centroid_vid == _faces(i, j)) {
				contains_center = true;
				break;
			}
		}

		if (contains_center) {
			Eigen::Vector3d c = Eigen::Vector3d::Zero();

			for (Eigen::DenseIndex j = 0; j < _faces.cols(); ++j) {
				c += _vertices.row(_faces(i, j)) / static_cast<double>(_faces.cols());
			}

			face_centers.push_back(std::pair<Eigen::DenseIndex, Eigen::Vector3d>(i, c));
		}
	}

	Eigen::DenseIndex centroid_fid = -1;
	double dist = std::numeric_limits<double>::max();

	for (Eigen::DenseIndex i = 0; i < face_centers.size(); ++i) {
		double c_dist = (face_centers[i].second - centroid).norm();

		if (c_dist < dist) {
			dist = c_dist;
			centroid_fid = face_centers[i].first;
		}
	}

	return centroid_fid;
}

Eigen::DenseIndex Patch::get_centroid_fid_on_origin_mesh() const {
	Eigen::DenseIndex centroid_fid = get_centroid_fid();

	for (auto p : _remapping_fid) {
		if (p.second == centroid_fid) {
			return p.first;
		}
	}

	return -1;
}

Eigen::DenseIndex Patch::vid_to_origin_mesh(Eigen::DenseIndex vid) {
	auto ovid = _remapping_vid_rev.find(vid);

	if (ovid == _remapping_vid_rev.end()) {
		throw std::domain_error("Invalid vid to origin mesh!");
	}

	return ovid->second;
}

Eigen::DenseIndex Patch::vid_to_local_patch(Eigen::DenseIndex origin_vid) {
	auto lvid = _remapping_vid.find(origin_vid);

	if (lvid == _remapping_vid.end()) {
		throw std::domain_error("Invalid vid to origin mesh!");
	}

	return lvid->second;
}

Eigen::DenseIndex Patch::fid_to_origin_mesh(Eigen::DenseIndex fid) {
	auto ofid = _remapping_fid_rev.find(fid);

	if (ofid == _remapping_fid_rev.end()) {
		throw std::domain_error("Invalid vid to origin mesh!");
	}

	return ofid->second;
}

double Patch::get_geodesic_extent(Eigen::DenseIndex center_vid) {
	if (_vids.count(center_vid) == 0) {
		throw std::invalid_argument("Patch::get_geodesic_extent(): center_vid is not in Patch!");
	}

	const Eigen::MatrixXd& V = _origin_mesh->vertices();

	const Eigen::SparseMatrix<int>& adj = _origin_mesh->adjacency_matrix();
	double geodesic_radius = 0.0;

	std::queue<std::shared_ptr<Vertex>> q;
	std::set<Eigen::DenseIndex> vids;

	q.push(std::make_shared<Vertex>(center_vid, 0.0));
	vids.insert(center_vid);

	while (!q.empty()) {
		std::shared_ptr<Vertex> p = q.front();
		q.pop();

		for (Eigen::SparseMatrix<int>::InnerIterator it(adj, static_cast<int>(p->_vid)); it; ++it) {
			Eigen::DenseIndex neighbor = it.row();   // neighbor vid

			if (_vids.count(neighbor) == 1 && vids.count(neighbor) == 0) {
				double dist = (V.row(p->_vid) - V.row(neighbor)).norm() + p->_path_length;

				if (dist > geodesic_radius) {
					geodesic_radius = dist;
				}

				q.push(std::make_shared<Vertex>(neighbor, dist));
				vids.insert(neighbor);
			}
		}
	}

	return geodesic_radius;
}

// A couple of assumptions
// 1) The mesh is triangular
// That's it, probably
std::vector<std::shared_ptr<Patch>> Patch::find_similar(const std::shared_ptr<ShapeSignature> signature, Eigen::DenseIndex& feature_vid, std::vector<Eigen::DenseIndex>& feature_matches) {
	std::vector<std::shared_ptr<Patch>> similar;

	if (signature == nullptr || _vids.size() <= 0) {
		return similar;
	}

	const Eigen::MatrixXd& features = signature->get_signature_values();

	double threshold = automatic_threshold(features);

	// Pick most uncommon patch value
	Eigen::MatrixXd matches;
	feature_vid = approximate_patch_feature(features, threshold, matches);

	if (feature_vid < 0 || feature_vid >= features.rows()) {
		return similar;
	}

	// Create priority queue, with lowest difference being at the front
	std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, PatchCompare> feature_diff;

	for (unsigned int i = 0; i < matches.rows(); ++i) {
		feature_diff.push(std::pair<double, int>(matches(i,1), static_cast<int>(matches(i,0))));
	}

	Eigen::VectorXd feature_compare(features.rows());
	feature_compare = (features.rowwise() - features.row(feature_vid)).rowwise().norm();

	// Determine some sort of geodesic radius based off of this patch
	double geodesic_radius = get_geodesic_extent(feature_vid);
	
	feature_matches.clear();

	// Construct geometric patch for current feature patch
	//Eigen::VectorXd feature_normal = N.row(feature_vid).transpose();
	std::shared_ptr<DiscreteExponentialMap> feature_dem = discrete_exponential_map(get_centroid_vid_on_origin_mesh());

	double radius = geodesic_radius / 2.0; // TODO: should scale this to the discrete exponential map instead of patch
	double radius_step = radius / 3.0;
	double angle_step = M_PI / 10.0; // 20 spokes
	
	GeodesicFan feature_fan(angle_step, radius, radius_step, feature_dem, signature);

	// Sanity check
	double madness_double;
	double better_be_damn_close_to_zero = feature_fan.compare(feature_fan, madness_double);

	std::cout << "Close to zero? " << better_be_damn_close_to_zero << std::endl;
	std::cout << "Orientation?   " << madness_double << std::endl;

	assert(std::fabs(better_be_damn_close_to_zero) < std::numeric_limits<double>::epsilon());

	const Eigen::SparseMatrix<int>& adj = _origin_mesh->adjacency_matrix();

	// Find the closest matches
	std::vector<std::pair<double, std::shared_ptr<Patch>>> match_metrics;
	while (!feature_diff.empty()) {
		auto match_vid = feature_diff.top();
		feature_diff.pop();

		if (match_vid.first > threshold) {
			break;
		}

		std::shared_ptr<Patch> match = Patch::instantiate(_origin_mesh, match_vid.second, geodesic_radius);

		if (match->_vids.count(feature_vid) != 0) {
			// this patch contains the original feature point, so discard
			continue;
		}

		// Align patches and determine mutli-scale matching functions
		// Construct geometric patch for match extent
		std::shared_ptr<DiscreteExponentialMap> match_dem = match->discrete_exponential_map(match->get_centroid_vid_on_origin_mesh());

		// Create multi-scaled scalar valued function for m discrete rotations of this patch on potential match patch, finding orientation of lowest feature difference 
		// (align by feature-match anchor and interpolate feature values between connected vertices)
		GeodesicFan match_fan(angle_step, radius, radius_step, match_dem, signature);

		double match_orientation = 0.0;
		double l1_match_metric = match_fan.compare(feature_fan, match_orientation);

		// TODO: Consider some different threshold to compare two patches -- what does the l1_match_metric conceptually represent?
		std::cout << "thresholding: ( " << l1_match_metric << " < " << threshold << " ): " << ((l1_match_metric < threshold) ? "TRUE" : "FALSE") << std::endl;

		if (l1_match_metric < threshold) {
			// Check that our match is disjoint from all other matches
			// If it isn't, use the patch with the best match metric
			std::set<unsigned int> overlap_indices;
			for (unsigned int i = 0; i < match_metrics.size(); ++i) {
				if (!match_metrics[i].second->is_disjoint_from(match)) {
					overlap_indices.insert(i);
				}
			}

			unsigned int i;
			for (i = 0; i < overlap_indices.size(); ++i) {
				if (match_metrics[i].first < l1_match_metric) {
					break;
				}
			}

			if (i < overlap_indices.size()) {
				// area is already covered by a better patch
				continue;
			}

			// remove old patches if necessary, and add better match
			std::vector<std::pair<double, std::shared_ptr<Patch>>> match_metrics_reduced;
			std::vector<Eigen::DenseIndex> feature_matches_reduced;
			for (unsigned int i = 0; i < match_metrics.size(); ++i) {
				if (overlap_indices.count(i) > 0) {
					continue;
				}

				feature_matches_reduced.push_back(feature_matches[i]);
				match_metrics_reduced.push_back(match_metrics[i]);
			}

			feature_matches = std::move(feature_matches_reduced);
			match_metrics = std::move(match_metrics_reduced);

			feature_matches.push_back(match_vid.second);
			match_metrics.push_back(std::pair<double, std::shared_ptr<Patch>>(l1_match_metric, match));
		}
	}

	for (auto it = match_metrics.begin(); it != match_metrics.end(); ++it) {
		similar.push_back(it->second);
	}

	return similar;
}

double Patch::automatic_threshold(const Eigen::MatrixXd& features) {
	// Instead of setting threshold, select the threshold that allows all patch points to have at least one match, and then multiply it by some factor (or transform it some other way??)
	double threshold = 0.0;

	for (auto it = _vids.cbegin(); it != _vids.cend(); ++it) {
		Eigen::VectorXd compare = (features.rowwise() - features.row(*it)).rowwise().norm();

		for (auto itt = _vids.cbegin(); itt != _vids.cend(); ++itt) {
			compare(*itt) = std::numeric_limits<double>::infinity();
		}

		double min = compare.minCoeff();

		if (min > threshold) {
			threshold = min;
		}
	}

	// TODO: Magic factor? Consider some smart replacement
	threshold *= 4.0;

	return threshold;
}

// Find "most unique" point within patch based on feature difference with other mesh vertices
Eigen::DenseIndex Patch::approximate_patch_feature(const Eigen::MatrixXd& features, double threshold, Eigen::MatrixXd& matches) {
	Eigen::DenseIndex feature_vid = -1;
	Eigen::MatrixXd compare(features.rows(), _vids.size());
	Eigen::MatrixXi threshold_count(_vids.size(),2);

	Eigen::DenseIndex i = 0;
	for (auto it = _vids.cbegin(); it != _vids.cend(); ++it) {
		compare.col(i) = (features.rowwise() - features.row(*it)).rowwise().norm();

		// Set all vertices in the patch to have an infinite compare value 
		// This will cause them to not be counted as a match
		for (auto itt = _vids.cbegin(); itt != _vids.cend(); ++itt) {
			compare.col(i)(*itt) = std::numeric_limits<double>::infinity();
		}

		// count similar, excluding self-similar
		threshold_count(i, 0) = static_cast<int>((compare.col(i).array() < threshold).count());
		threshold_count(i, 1) = static_cast<int>(*it);
		++i;
	}

	// Find vertex with fewest matches under threshold
	int feature_index = -1;
	int match_nbr = 0;
	for (unsigned int i = 0; i < threshold_count.size(); ++i) {
		match_nbr = threshold_count.col(0).minCoeff(&feature_index);

		if (match_nbr <= 0) {
			threshold_count(feature_index, 0) = std::numeric_limits<int>::max();
		}
		else {
			break;
		}
	}

	if (match_nbr <= 0) {
		return false;
	}

	feature_vid = threshold_count(feature_index, 1);

	int match_index = 0;
	matches.resize(threshold_count(feature_index, 0), 2);
	for (unsigned int i = 0; i < features.rows(); ++i) {
		if (compare(i, feature_index) < threshold && i != feature_vid) {
			matches(match_index, 0) = i;
			matches(match_index, 1) = compare(i, feature_index);
			match_index++;
		}
	}

	return feature_vid;
}

std::unordered_map<Eigen::DenseIndex, std::vector<Eigen::DenseIndex>> Patch::bipartite_threshold_matching(std::shared_ptr<Patch> other, const Eigen::MatrixXd& features, double threshold) {
	std::unordered_map<Eigen::DenseIndex, std::vector<Eigen::DenseIndex>> bipartite_map;

	if (other == nullptr) {
		return bipartite_map;
	}

	for (auto it = _vids.cbegin(); it != _vids.cend(); ++it) {
		std::vector<Eigen::DenseIndex> edge_to;

		for (auto itt = _vids.cbegin(); itt != _vids.cend(); ++itt) {
			double dist = (features.row(*it) - features.row(*itt)).norm();

			if (dist < threshold) {
				edge_to.push_back(*itt);
			}
		}

		bipartite_map.insert(std::pair<Eigen::DenseIndex, std::vector<Eigen::DenseIndex>>(*it, edge_to));
	}

	return bipartite_map;
}

std::vector<std::shared_ptr<Patch>> Patch::shatter() { 
	std::vector<std::shared_ptr<Patch>> patches;

	for (auto vid : _vids) {
		patches.push_back(Patch::instantiate(origin_mesh(), vid));
	}

	return patches;
}

// TODO: Update to a As-Rigid-As-Possible patch parameterization
std::shared_ptr<DiscreteExponentialMap> Patch::discrete_exponential_map(Eigen::DenseIndex center) {
	return std::make_shared<DiscreteExponentialMap>(dynamic_cast<PatchInstancer*>(this)->shared_from_this(), center);

	// This was pulled directly from the libigl tutorial for ARAP parameterization
	/*Eigen::MatrixXd vertex_uv;

	if (_vids.size() >= 3) {
		Eigen::MatrixXd initial_guess;

		// Compute the initial solution for ARAP (harmonic parametrization)
		std::vector<Eigen::DenseIndex> loop;
		igl::boundary_loop(_faces, loop);

		Eigen::VectorXi bnd;
		bnd.resize(loop.size());
		for (size_t i = 0; i < loop.size(); ++i)
			bnd(i) = loop[i];

		Eigen::MatrixXd bnd_uv;
		igl::map_vertices_to_circle(_vertices, bnd, bnd_uv);

		igl::harmonic(_vertices, _faces, bnd, bnd_uv, 1, initial_guess);

		// Add dynamic regularization to avoid to specify boundary conditions
		igl::ARAPData arap_data;
		arap_data.with_dynamics = true;
		Eigen::VectorXi b = Eigen::VectorXi::Zero(0);
		Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0, 0);

		// Initialize ARAP
		arap_data.max_iter = 100;
		// 2 means that we're going to *solve* in 2d
		arap_precomputation(_vertices, _faces, 2, b, arap_data);

		// Solve arap using the harmonic map as initial guess
		vertex_uv = initial_guess;

		igl::arap_solve(bc, arap_data, vertex_uv);
	}

	// Pack points for map representation
	std::map<Eigen::DenseIndex, Eigen::Vector2d> DEM_points;
	for (Eigen::DenseIndex i = 0; i < vertex_uv.rows(); ++i) {
		// Subtract center vid UV
		DEM_points.insert(std::pair<Eigen::DenseIndex, Eigen::Vector2d>(vid_to_origin_mesh(i), vertex_uv.row(i) - vertex_uv.row(_remapping_vid[center])));
	}

	// Scale UV to make it representative of edge distances
	if (DEM_points.size() > 0) {
		auto ring = one_ring(center);

		double scale = 0.0;
		for (auto vid : ring) {
			if (vid == center) {
				continue;
			}

			scale += ((_origin_mesh->vertices().block<1, 3>(vid, 0) - _origin_mesh->vertices().block<1, 3>(center, 0)).norm() / vertex_uv.row(_remapping_vid[vid]).norm()) / static_cast<double>(ring.size());
		}

		if (scale > std::numeric_limits<double>::epsilon()) {
			for (auto it = DEM_points.begin(); it != DEM_points.end(); ++it) {
				it->second = scale * it->second;
			}
		}
	} else {
		DEM_points.insert(std::pair<Eigen::DenseIndex, Eigen::Vector2d>(center, Eigen::Vector2d::Zero()));
	}

	// Create a TBN for the parameterization
	Eigen::Matrix3d TBN = basis_from_plane_normal(_origin_mesh->vertex_normals().row(center));

	return std::make_shared<DiscreteExponentialMap>(_remapping_vid[center], TBN, DEM_points, dynamic_cast<PatchInstancer*>(this)->shared_from_this());*/
}
