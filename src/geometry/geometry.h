#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>

#undef min
#undef max

#include <halfedge/trimesh.h>

typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> ColorChannel;
typedef std::tuple<std::shared_ptr<ColorChannel>, std::shared_ptr<ColorChannel>, std::shared_ptr<ColorChannel>, std::shared_ptr<ColorChannel>> ColorTexture;

Eigen::Matrix3d basis_from_plane_normal(const Eigen::Vector3d plane_normal);
bool point_in_triangle(Eigen::Vector2d pt, Eigen::Vector2d v1, Eigen::Vector2d v2, Eigen::Vector2d v3);
Eigen::Vector3d local_log_map(const Eigen::Vector3d& origin, const Eigen::Matrix3d& origin_TBN, const Eigen::Vector3d& point);

class Geometry {
	public:
		virtual ~Geometry();

		virtual const Eigen::MatrixXd& vertices() const = 0;
		virtual const Eigen::MatrixXd& vertex_normals() const = 0;
		virtual const Eigen::MatrixXi& faces() const = 0;
		virtual const Eigen::SparseMatrix<int>& adjacency_matrix() const = 0;

		virtual ColorTexture& color_texture() = 0;

		virtual const trimesh::trimesh_t& halfedge() = 0;

		virtual std::vector<Eigen::DenseIndex> one_ring(Eigen::DenseIndex center_vid) = 0;

		virtual Eigen::DenseIndex vid_to_origin_mesh(Eigen::DenseIndex vid) = 0;
		virtual Eigen::DenseIndex fid_to_origin_mesh(Eigen::DenseIndex fid) = 0;

	protected:
		Geometry();
};

struct BarycentricCoord {
	public:
		BarycentricCoord() : _fid(-1), _coeff(Eigen::VectorXd::Zero(3)) { }
		BarycentricCoord(Eigen::DenseIndex fid, Eigen::VectorXd barycentric_coeff) : _fid(fid), _coeff(barycentric_coeff) { }

		Eigen::VectorXd to_world(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) const {
			return resolve(F, V);
		}

		Eigen::VectorXd to_world(std::shared_ptr<Geometry> geo) const {
			return resolve(geo->faces(), geo->vertices());
		}

		Eigen::VectorXd to_sig_value(std::shared_ptr<Geometry> geo, const Eigen::VectorXd& sig_values) const {
			return resolve(geo->faces(), sig_values);
		}

		Eigen::DenseIndex _fid;
		Eigen::VectorXd _coeff;

		template<class Archive>
		void save(Archive & archive) const {
			std::array<double, 3> bc_coeff = { _coeff(0), _coeff(1), _coeff(2) };

			archive(CEREAL_NVP(_fid), CEREAL_NVP(bc_coeff));
		}

		template<class Archive>
		void load(Archive & archive) {
			std::array<double, 3> bc_coeff;

			archive(_fid, bc_coeff);

			_coeff = (Eigen::VectorXd(3) << bc_coeff[0], bc_coeff[1], bc_coeff[2]).finished();
		}

	protected:
		Eigen::VectorXd resolve(const Eigen::MatrixXi& F, const Eigen::MatrixXd& signal) const {
			if (_fid < 0) {
				return Eigen::VectorXd::Zero(0);
			}

			Eigen::VectorXd resolved = Eigen::VectorXd::Zero(signal.cols());

			for (Eigen::DenseIndex i = 0; i < F.cols(); ++i) {
				resolved += signal.row(F(_fid, i)) * _coeff(i);
			}

			return resolved;
		}
};



#endif