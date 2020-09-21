#ifndef SHORTEST_PATH_H_
#define SHORTEST_PATH_H_

#include <set>
#include <map>
#include <memory>

#include <Eigen/Sparse>

class Patch;
class Mesh;

namespace shortest_path {
	struct DjikstraNode {
		public:
			double _dist;

		protected:
			DjikstraNode(): _dist(std::numeric_limits<double>::infinity()) { }
			DjikstraNode(double dist): _dist(dist) { }
	};

	struct DjikstraVertexNode: public DjikstraNode {
		DjikstraVertexNode(): _prev(nullptr), _vid(-1) { }
		DjikstraVertexNode(std::shared_ptr<DjikstraVertexNode> prev, double dist, Eigen::DenseIndex vid): DjikstraNode(dist), _prev(prev), _vid(vid) { }

		std::shared_ptr<DjikstraVertexNode> _prev;
		
		Eigen::DenseIndex _vid;
	};

	struct DjikstraFaceNode: public DjikstraNode {
		DjikstraFaceNode() : _prev(nullptr), _edge(std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(-1, -1)), _midpoint(Eigen::VectorXd::Zero(0)) { }
		DjikstraFaceNode(std::shared_ptr<DjikstraFaceNode> prev, double dist, std::pair<Eigen::DenseIndex, Eigen::DenseIndex> edge, const Eigen::VectorXd& midpoint): DjikstraNode(dist), _prev(prev), _edge(edge), _midpoint(midpoint) { }

		std::shared_ptr<DjikstraFaceNode> _prev;
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex> _edge;
		Eigen::VectorXd _midpoint;
	};

	class DjikstraDist {
		public:
			bool operator()(std::shared_ptr<DjikstraNode> left, std::shared_ptr<DjikstraNode> right) { return left->_dist > right->_dist; };
	};

	std::map<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>> djikstras_algorithm(std::shared_ptr<Patch> patch, Eigen::DenseIndex source, std::set<Eigen::DenseIndex> exclude = { });
	Eigen::MatrixXd floyd_warshall(const Eigen::SparseMatrix<double> edge_weights);
	std::vector<Eigen::DenseIndex> face_to_face(Eigen::DenseIndex source, Eigen::DenseIndex sink, std::shared_ptr<Mesh> mesh);
}

#endif
