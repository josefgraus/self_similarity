#include "geodesic_fan.h"

#include <chrono>

#include <utilities/units.h>
#include <shape_signatures/heat_kernel_signature.h>
#include <matching/parameterization/discrete_exponential_map.h>
#include <shape_signatures/shape_signature.h>
#include <geometry/geometry.h>
#include <geometry/patch.h>

#undef min
#undef max

GeodesicFan::GeodesicFan(const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature):
	_custom_blade(nullptr),
	_origin_map(dem),
	_radius(0.0),
	_radius_step(0.0),
	_angle_step(0.0),
	_normalized(false) {

	if (dem == nullptr || signature == nullptr) {
		return;
	}

	_radius = dem->get_radius();
	_radius_step = _radius / 3.0;
	_angle_step = M_PI / 10.0; // 20 spokes

	populate(dem, signature);
}

GeodesicFan::GeodesicFan(double angle_step, double radius, double radius_step, const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature, bool normalized):
	_custom_blade(nullptr),
	_origin_map(dem),
	_angle_step(angle_step),
	_radius(radius),
	_radius_step(radius_step),
	_normalized(normalized) {

	if (dem == nullptr || signature == nullptr) {
		return;
	}

	populate(dem, signature);
}

GeodesicFan::GeodesicFan(double angle_step, double radius, double radius_step, const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature, Eigen::Vector3d encode_up, bool normalized):
	_custom_blade(nullptr),
	_origin_map(dem),
	_angle_step(angle_step),
	_radius(radius),
	_radius_step(radius_step),
	_normalized(normalized) {
	
	if (dem == nullptr || signature == nullptr) {
		return;
	}

	populate(dem, signature, encode_up);
}

GeodesicFan::GeodesicFan(std::shared_ptr<GeodesicFanBlade> custom_blade, double angle_step, const std::shared_ptr<ShapeSignature> signature) {
	_custom_blade = custom_blade;
	_fan = custom_blade->blade_values(angle_step, signature, &_origin_map);

	// Let the GeodesicFanBlade handle how values are sampled from the signature
	_normalized = false; 
	_sig_min = signature->lower_bound();
	_sig_max = signature->upper_bound();

	// Not really needed since a custom blade is supplied
	_radius = 0.0; // custom_blade->radius();
	_radius_step = 0.0; // custom_blade->average_sample_spacing();
	_angle_step = angle_step;

	if (_origin_map != nullptr) {
		_TBN << _origin_map->get_tangent(), _origin_map->get_bitangent(), _origin_map->get_normal();
	} else {
		_TBN = Eigen::Matrix3d::Identity();
	}
}

GeodesicFan::~GeodesicFan() {

}

std::ostream& operator<<(std::ostream &out, const GeodesicFan &fan) {
	out << "[ " << fan._center.transpose();

	for (Eigen::DenseIndex j = 0; j < fan.spokes(); ++j) {
		out << std::endl;
		for (Eigen::DenseIndex i = 0; i < fan.levels(); ++i) {
			out << fan._fan(i, j).transpose() << " ";
		}
	}

	out << " ]" << std::endl;

	return out;
}

void GeodesicFan::populate(const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature, Eigen::Vector3d encode_up) {
	if (signature->feature_count() <= 0) {
		return;
	}

	//_sig_min = signature->get_signature_values().minCoeff();
	//_sig_max = signature->get_signature_values().maxCoeff();
	_sig_min = signature->lower_bound();
	_sig_max = signature->upper_bound();

	// Query discrete exponential map for interpolated values
	Eigen::Vector2d dem_center = (Eigen::Vector2d() << 0.0, 0.0).finished();
	
	if (dem->get_center_vid() < 0) {
		return;
	}

	Eigen::DenseIndex fid = dem->get_center_fid();
	Eigen::DenseIndex vid = dem->get_center_vid();
	Eigen::VectorXd lc = signature->lerpable_coord(fid, vid);

	_center = signature->lerpable_to_signature_value(lc);

	if (_normalized) {
		for (Eigen::DenseIndex index = 0; index < _center.rows(); ++index) {
			_center(index) = Units::normalize(_center(index), _sig_min, _sig_max);
		}
	}

	if (!encode_up.isZero()) {
		encode_up.normalize();

		double ev_dot = encode_up.dot(signature->origin_mesh()->vertex_normals().row(vid).normalized());

		Eigen::VectorXd val_up(_center.size() + 1); val_up << _center, Units::normalize(ev_dot, -1.0, 1.0); ev_dot;

		_center = val_up;
	}

	if (std::fabs(_radius) < std::numeric_limits<double>::epsilon() ||
		std::fabs(_radius_step) < std::numeric_limits<double>::epsilon()) {
		_fan.resize(0, 0);

		return;
	}

	Eigen::DenseIndex rows = static_cast<Eigen::DenseIndex>(std::floor(_radius / _radius_step));
	Eigen::DenseIndex cols = static_cast<Eigen::DenseIndex>(std::floor(2.0 * M_PI / _angle_step));

	_fan.resize(rows, cols);
	for (unsigned int i = 0; i < _fan.size(); ++i) {
		_fan(i).resize(signature->feature_dimension());
	}

	for (unsigned int j = 0; j < rows; ++j) {
		double r_step = static_cast<double>(j + 1) * _radius_step;

		for (unsigned int k = 0; k < cols; ++k) {
			double t_step = static_cast<double>(k) * _angle_step;
			Eigen::Vector2d polar_coord(r_step, t_step);
			Eigen::VectorXd value;

			if (dem != nullptr) {
				Eigen::VectorXd val = dem->query_map_value_polar(polar_coord, signature);

				if (_normalized && val.sum() >= 0.0) {
					for (Eigen::DenseIndex index = 0; index < val.rows(); ++index) {
						val(index) = Units::normalize(val(index), _sig_min, _sig_max);
					}
				}

				_fan(j, k) = val;
			}
		}
	}

	if (dem != nullptr) {
		_TBN << dem->get_tangent(), dem->get_bitangent(), dem->get_normal();
	} else {
		_TBN = Eigen::Matrix3d::Identity();
	}
}

const Eigen::VectorXd& GeodesicFan::operator()(Eigen::DenseIndex i, Eigen::DenseIndex j) {
	if (i >= _fan.rows() || j >= _fan.cols()) {
		std::stringstream err;
		err << "GeodesicFan::operator(): indices (" << i << ", " << j << ") exceed range of fan!";
		throw std::out_of_range(err.str());
	}

	return _fan(i, j);
}

double GeodesicFan::angle_step() {
	return _angle_step;
}

double GeodesicFan::radius() {
	return _radius;
}

double GeodesicFan::radius_step() {
	return _radius_step;
}

unsigned int GeodesicFan::spokes() const {
	return _fan.cols();
}

unsigned int GeodesicFan::levels() const {
	return _fan.rows();
}

double GeodesicFan::lower_bound() {
	if (_normalized) {
		return 0.0;
	} 

	return _sig_min;
}

double GeodesicFan::upper_bound() {
	if (_normalized) {
		return 1.0;
	}

	return _sig_max;
}

// Fans must match in number of spokes, samples along spoke, and layers (radius_step, angle_step, and columns of the signature values)
double GeodesicFan::compare(const GeodesicFan& other, double& orientation) {
	if (other._fan.rows() != _fan.rows() || other._fan.cols() != _fan.cols() || other._center.size() != _center.size()) {
		throw std::invalid_argument("The two geodesic fans are different sizes!");
	}

	orientation = 0.0;

	Eigen::VectorXd center_diff = _center - other._center;
	double center_norm_sq = center_diff.squaredNorm();

	if (_fan.size() <= 0) {
		// Should we throw an exception here instead? Or perhaps two empty fans should be identical
		return std::sqrt(center_norm_sq);
	} 

	unsigned int elements = _fan(0, 0).size() * _fan.size();

	if (elements <= 0) {
		// The elements of the fan contains zero size vectors?
		return std::sqrt(center_norm_sq);
	}

	Eigen::DenseIndex spokes = _fan.cols();
	Eigen::VectorXd metrics = Eigen::VectorXd::Zero(spokes);

	// Rotate the other fan matrix by reordering the columns, and compare the reordering each iteration to this->_fan
	// L2-norm metric as comparision operator
	for (Eigen::DenseIndex j = 0; j < spokes; ++j) {
		for (Eigen::DenseIndex k = 0; k < spokes; ++k) {
			Eigen::DenseIndex spoke_offset = (j + k) % spokes;

			if (_fan.col(k).size() != other._fan.col(spoke_offset).size()) {
				throw std::invalid_argument("The two geodesic fans are different sizes!");
			}

			Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> layer_diffs = (_fan.col(k) - other._fan.col(spoke_offset));

			Eigen::VectorXd layer_norm = Eigen::VectorXd::Zero(_fan(0,0).size());

			for (Eigen::DenseIndex i = 0; i < layer_diffs.rows(); ++i) {
				layer_norm += layer_diffs(i).cwiseProduct(layer_diffs(i));	// L2-norm
			}

			metrics(j) += layer_norm.sum();
		}

		metrics(j) += center_norm_sq;

		metrics(j) = std::sqrt(metrics(j));
	}

	Eigen::DenseIndex min_spoke_index;

	double metric_avg = metrics.minCoeff(&min_spoke_index) / std::sqrt(static_cast<double>(elements + _center.size()));

	for (Eigen::DenseIndex i = 0; i < metrics.size(); ++i) {
		if ((metrics(i) - metrics(min_spoke_index)) < std::numeric_limits<double>::epsilon()) {
			min_spoke_index = i;
			break;
		}
	}

	orientation = min_spoke_index * other._angle_step;

	if (orientation > M_PI) {
		orientation -= 2.0 * M_PI;
	}

	return metric_avg;
}

double GeodesicFan::compare(const GeodesicFanBlade::SignatureTensor& blade_tensor, double& orientation) {
	if (_custom_blade == nullptr) {
		throw std::logic_error("This fan doesn't contain its own custom blade for a valid comparison!");
	}

	if (blade_tensor.rows() != _fan.rows()) {
		throw std::invalid_argument("The two geodesic fans are different sizes!");
	}

	orientation = 0.0;

	if (_fan.size() <= 0) {
		// Should we throw an exception here instead? Or perhaps two empty fans should be identical
		orientation = 0.0;
		return 0.0;
	}

	unsigned int elements = _fan(0, 0).size() * _fan.size();

	if (elements <= 0) {
		// The elements of the fan contains zero size vectors?
		orientation = 0.0;
		return 0.0;
	}

	Eigen::DenseIndex spokes = _fan.cols();
	Eigen::VectorXd metrics = Eigen::VectorXd::Zero(spokes);

	// Rotate the other fan matrix by reordering the columns, and compare the reordering each iteration to this->_fan
	// L2-norm metric as comparision operator
	for (Eigen::DenseIndex i = 0; i < spokes; ++i) {
		if (_fan.col(i).rows() != blade_tensor.rows()) {
			throw std::invalid_argument("The two geodesic fans are different sizes!");
		}

		Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> layer_diffs = (_fan.col(i) - blade_tensor);

		Eigen::VectorXd layer_norm = Eigen::VectorXd::Zero(_fan(0, 0).size());

		for (Eigen::DenseIndex j = 0; j < layer_diffs.rows(); ++j) {
			layer_norm += layer_diffs(j).cwiseProduct(layer_diffs(j));	// L2-norm
		}

		metrics(i) = std::sqrt(layer_norm.sum());
	}

	Eigen::DenseIndex min_spoke_index;

	double metric_avg = metrics.minCoeff(&min_spoke_index) / std::sqrt(static_cast<double>(elements + _center.size()));

	for (Eigen::DenseIndex i = 0; i < metrics.size(); ++i) {
		if ((metrics(i) - metrics(min_spoke_index)) < std::numeric_limits<double>::epsilon()) {
			min_spoke_index = i;
			break;
		}
	}

	orientation = min_spoke_index * _angle_step;

	if (orientation > M_PI) {
		orientation -= 2.0 * M_PI;
	}

	return metric_avg;
}

Eigen::VectorXd GeodesicFan::aligned_vector(double orientation) {
	Eigen::DenseIndex n = _center.size();
	for (Eigen::DenseIndex i = 0; i < _fan.rows(); ++i) {
		for (Eigen::DenseIndex j = 0; j < _fan.cols(); ++j) {
			n += _fan(i, j).size();
		}
	}

	Eigen::DenseIndex cursor = 0;
	Eigen::VectorXd aligned_fan_vec(n);

	unsigned int start_spoke = 0;
	if (orientation < 0.0) {
		start_spoke = static_cast<unsigned int>(std::round((orientation + 2.0 * M_PI) / _angle_step));
	} else {
		start_spoke = static_cast<unsigned int>(std::round(orientation / _angle_step));
	}

	for (Eigen::DenseIndex i = 0; i < _center.size(); ++i, ++cursor) {
		aligned_fan_vec(cursor) = _center(i);
	}

	for (Eigen::DenseIndex j = 0; j < _fan.cols(); ++j) {
		for (Eigen::DenseIndex i = 0; i < _fan.rows(); ++i) {
			Eigen::DenseIndex spoke = (start_spoke + j) % spokes();

			for (Eigen::DenseIndex k = 0; i < _fan(i, j).size(); ++k, ++cursor) {
				aligned_fan_vec(cursor) = _fan(i, spoke)(k);
			}
		}
	}

	return std::move(aligned_fan_vec);
}

std::shared_ptr<GeodesicFan> GeodesicFan::from_aligned_vector(Eigen::VectorXd vec, unsigned int spokes, unsigned int levels, unsigned int sig_dim) {
	if (vec.size() != (spokes * levels * sig_dim) + sig_dim) {
		return nullptr;
	}

	vec.normalize();

	/*std::shared_ptr<GeodesicFan> fan = std::make_shared<GeodesicFan>();

	fan->_center = vec.topRows(sig_dim);

	fan->_fan.resize(levels, spokes);
	for (unsigned int i = 0; i < fan->_fan.size(); ++i) {
		fan->_fan(i).resize(sig_dim);
	}

	Eigen::DenseIndex offset = sig_dim;
	for (Eigen::DenseIndex j = 0; j < fan->_fan.cols(); ++j) {
		for (Eigen::DenseIndex i = 0; i < fan->_fan.rows(); ++i) {
			fan->_fan(i, j) = vec.block(offset, 0, sig_dim, 1);
		}
	}

	fan->_angle_step = 2.0 * M_PI / spokes;
	fan->_radius = 1.0;
	fan->_radius_step = 1.0 / 3.0;
	Eigen::Matrix3d _TBN = Eigen::Matrix3d::Identity();

	fan->_sig_min = 0.0;
	fan->_sig_max = 1.0;
	fan->_normalized = true;*/

	return nullptr;
}

double GeodesicFan::l2_norm() {
	double sum = 0.0;

	for (unsigned int i = 0; i < _fan.rows(); ++i) {
		for (unsigned int j = 0; j < _fan.cols(); ++j) {
			for (unsigned int k = 0; k < _fan(i, j).size(); ++k) {
				double sig_value = _fan(i, j)(k);

				if (sig_value > 0.0) {
					sum += std::pow(sig_value, 2.0);
				}
			}
		}
	}

	for (Eigen::DenseIndex i = 0; i < _center.size(); ++i) {
		double sig_value = _center(i);

		if (sig_value > 0.0) {
			sum += std::pow(sig_value, 2.0);
		}
	}

	return std::sqrt(sum);
}

double GeodesicFan::scaled_l2_norm() {
	double sum = 0.0;
	unsigned int n = 0;

	for (unsigned int i = 0; i < _fan.rows(); ++i) {
		for (unsigned int j = 0; j < _fan.cols(); ++j) {
			for (unsigned int k = 0; k < _fan(i, j).size(); ++k) {
				double sig_value = _fan(i, j)(k);

				if (sig_value > 0.0) {
					sum += std::pow(sig_value, 2.0);
					n++;
				}
			}
		}
	}

	for (Eigen::DenseIndex i = 0; i < _center.size(); ++i) {
		double sig_value = _center(i);

		if (sig_value > 0.0) {
			sum += std::pow(sig_value, 2.0);
			n++;
		}
	}

	if (n == 0) {
		return 0.0;
	} else {
		return std::sqrt(sum) / static_cast<double>(n);
	}
}

double GeodesicFan::geometric_mean() {
	unsigned int entries = 0;
	double product = 1.0;

	for (unsigned int i = 0; i < _fan.rows(); ++i) {
		for (unsigned int j = 0; j < _fan.cols(); ++j) {
			for (unsigned int k = 0; k < _fan(i, j).size(); ++k) {
				double sig_value = _fan(i, j)(k);

				if (sig_value > 0.0) {
					product *= sig_value;

					entries++;
				}
			}
		}
	}

	for (Eigen::DenseIndex i = 0; i < _center.size(); ++i) {
		double sig_value = _center(i);

		if (sig_value > 0.0) {
			product *= sig_value;

			entries++;
		}
	}

	return std::pow(product, 1.0 / static_cast<double>(entries));
}

Eigen::MatrixXd GeodesicFan::get_fan_vertices() const {
	if (_fan.size() <= 0) {
		return Eigen::MatrixXd();
	}

	if (_custom_blade != nullptr) {
		Eigen::MatrixXd planar_points = _custom_blade->parameterized_space_points_2d();

		Eigen::MatrixXd V_3d(_fan.size(), 3);

		Eigen::DenseIndex vIndex = 0;
		for (Eigen::DenseIndex i = 0; i < planar_points.cols(); ++i) {
			for (int j = 0; j < _fan.cols(); ++j) {
				Eigen::AngleAxis<double> R(static_cast<double>(j) * _angle_step, Eigen::Vector3d::UnitZ());

				Eigen::Vector3d tbn_point = R * (Eigen::Vector3d() << planar_points(0, i), planar_points(1, i), 0.0).finished();

				V_3d.row(vIndex++) = _TBN * tbn_point;
			}
		}

		return V_3d;
 	} 

	Eigen::MatrixXd V_3d(_fan.size() + 1, 3);

	// Center point
	V_3d.row(0) = (Eigen::Vector3d() << 0.0, 0.0, 0.0).finished();

	Eigen::DenseIndex vIndex = 1;
	for (unsigned int i = 0; i < _fan.rows(); ++i) {
		for (unsigned int j = 0; j < _fan.cols(); ++j) {
			double r = static_cast<double>(i + 1) * _radius_step;
			double t = static_cast<double>(j) * _angle_step;
			double x = r * std::cos(t);
			double y = r * std::sin(t);

			Eigen::Vector3d tbn_point = (Eigen::Vector3d() << x, y, 0.0).finished();

			V_3d.row(vIndex++) = _TBN * tbn_point;
		}
	}

	return V_3d;
}

Eigen::VectorXd GeodesicFan::get_fan_values(unsigned int layer) const {
	if (layer >= _fan.size()) {
		throw std::range_error("Layer does not exist within geodesic fan!");
	}

	Eigen::VectorXd fan_vert_order(_fan.size() /* + 1 */);

	//fan_vert_order(0) = _center(layer);

	Eigen::DenseIndex vIndex = 0; // 1;
	for (unsigned int i = 0; i < _fan.rows(); ++i) {
		for (unsigned int j = 0; j < _fan.cols(); ++j) {
			fan_vert_order(vIndex++) = _fan(i, j)(layer);
		}
	}

	return fan_vert_order;
}

bool GeodesicFan::layer_over(std::shared_ptr<GeodesicFan> other, double orientation) {
	// Fans must have either the same number of spokes, or one must have zero spokes
	if (other->spokes() != spokes() && (other->spokes() != 0 && spokes() != 0)) {
		return false;
	}

	// Combine center vectors
	//Eigen::VectorXd c(other->_center.size() + _center.size()); c << other->_center, _center;
	// Actually, completely replace for now -- it's a special MeshMatch reason this function was created
	other->_center = _center;

	if (spokes() > 0 && other->spokes() > 0) {
		Eigen::DenseIndex spoke_offset = std::round(orientation / (2.0 * M_PI / spokes()));

		// spoke numbers match, just determine if we need to allocate more levels
		int lv_diff = other->levels() - levels();

		if (lv_diff > 0) {
			other->_fan.conservativeResize(levels(), Eigen::NoChange);	
		}

		for (Eigen::DenseIndex j = 0; j < spokes(); ++j) {
			Eigen::DenseIndex oj = (j + spoke_offset) % spokes();

			for (Eigen::DenseIndex i = 0; i < levels(); ++i) {
				Eigen::VectorXd layered(other->_fan(i, j).size() + _fan(i, oj).size()); layered << other->_fan(i, j), _fan(i, oj);
				other->_fan(i, j) = layered;
			}
		}
	}

	return true;
}