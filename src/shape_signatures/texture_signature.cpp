#include "texture_signature.h"

#include <matching/constrained_relation_solver.h>
#include <matching/geodesic_fan.h>

struct TSInstancer : public TextureSignature {
	TSInstancer(std::shared_ptr<Mesh> mesh) : TextureSignature(mesh) {}
};

TextureSignature::TextureSignature(const std::shared_ptr<Mesh> mesh): ShapeSignature(mesh, std::make_shared<TextureParameterOptimization>()) {
	auto ct = _mesh->color_texture();

	if (std::get<0>(ct) == nullptr ||
		std::get<1>(ct) == nullptr ||
		std::get<2>(ct) == nullptr ||
		std::get<3>(ct) == nullptr) {
		throw std::domain_error("Color texture of mesh is invalid!! (was it loaded??)");
	}
}

TextureSignature::~TextureSignature() {
}

std::shared_ptr<TextureSignature> TextureSignature::instantiate(std::shared_ptr<Mesh> mesh) {
	return std::make_shared<TSInstancer>(mesh);
}

const Eigen::VectorXd TextureSignature::lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid) {
	Eigen::DenseIndex index = -1;

	for (Eigen::DenseIndex i = 0; i < _mesh->faces().cols(); ++i) {
		if (vid == _mesh->faces()(fid, i)) {
			index = _mesh->faces_uv()(fid, i);
		}
	}

	if (index < 0) {
		throw std::domain_error("lerpable_coord(): vid is not part of the face referenced by fid!");
	}

	return _mesh->vertex_uv().row(index);
}

Eigen::VectorXd TextureSignature::lerpable_to_signature_value(const Eigen::VectorXd& lerped) {
	auto ct = _mesh->color_texture();

	double w = std::get<0>(ct)->cols();
	double h = std::get<0>(ct)->rows();

	Eigen::DenseIndex x = std::floor((w - 1.0) * lerped(0));
	Eigen::DenseIndex y = std::floor((h - 1.0) * lerped(1));

	Eigen::VectorXd c(3);
	c << (*std::get<0>(ct))(x,y),
		 (*std::get<1>(ct))(x,y), 
		 (*std::get<2>(ct))(x,y);

	return c;
}

Eigen::VectorXi TextureSignature::uv_to_pixel_coord(Eigen::Vector2d uv) {
	auto ct = _mesh->color_texture();

	double w = std::get<0>(ct)->cols();
	double h = std::get<0>(ct)->rows();

	Eigen::DenseIndex x = std::floor((w - 1.0) * uv(0));
	Eigen::DenseIndex y = std::floor((h - 1.0) * uv(1));

	return Eigen::Vector2i(x, y);
}

Eigen::VectorXd TextureSignature::pixel_coord_to_uv(Eigen::Vector2i pixel_coord) {
	auto ct = _mesh->color_texture();

	double w = std::get<0>(ct)->cols();
	double h = std::get<0>(ct)->rows();

	double u = pixel_coord(0) / w;
	double v = pixel_coord(1) / h;

	return Eigen::Vector2d(u, v);
}

bool TextureSignature::set_signature_value(Eigen::Vector2d uv, Eigen::VectorXd value) {
	auto ct = _mesh->color_texture();

	double w = std::get<0>(ct)->cols();
	double h = std::get<0>(ct)->rows();

	Eigen::DenseIndex x = std::floor((w - 1.0) * uv(0));
	Eigen::DenseIndex y = std::floor((h - 1.0) * uv(1));

	(*std::get<0>(ct))(x,y) = std::floor(value(0));
	(*std::get<1>(ct))(x,y) = std::floor(value(1));
	(*std::get<2>(ct))(x,y) = std::floor(value(2));

	return true;
}

Eigen::VectorXd TextureSignature::get_signature_values(double index) {
	throw std::logic_error("get_signature_values is not implemented for TextureSignature yet!");
}

unsigned long TextureSignature::feature_count() {
	// The number of "points" the signature defines values for across the mesh 
	return _mesh->vertex_uv().rows();
}

unsigned long TextureSignature::feature_dimension() {
	// The dimensionality of the feature vector defined for each "point" the signature defines values for across the mesh 
	return 3; 
}

Eigen::MatrixXd TextureSignature::sig_steps() {
	throw std::domain_error("TextureSignature does not have a tunable parameter space!");
}

const double TextureSignature::step_width(double param) {
	throw std::domain_error("TextureSignature does not have a tunable parameter space!");
}

double TextureSignature::lower_bound() {
	return 0.0;
}

double TextureSignature::upper_bound() {
	return 255.0;
}

double TextureSignature::param_lower_bound() {
	return 0.0;
}

double TextureSignature::param_upper_bound() {
	return 0.0;
}

void TextureSignature::resample_at_param(double param) {
	return;
}

TextureSignature::TextureParameterOptimization::TextureParameterOptimization() {

}

TextureSignature::TextureParameterOptimization::~TextureParameterOptimization() {

}

double TextureSignature::TextureParameterOptimization::value(const TVector &x) {
	return 0.0;
}

cppoptlib::Problem<double>::TVector TextureSignature::TextureParameterOptimization::upperBound() const {
	TVector upper_bound(1);
	upper_bound << 0.0;

	return upper_bound;
}

cppoptlib::Problem<double>::TVector TextureSignature::TextureParameterOptimization::lowerBound() const {
	TVector upper_bound(1);
	upper_bound << 0.0;

	return upper_bound;
}

Eigen::MatrixXd TextureSignature::TextureParameterOptimization::param_steps(unsigned int steps) {
	throw std::domain_error("TextureSignature does not have a tunable parameter space!");
}

double TextureSignature::u_step() {
	auto color_texture = _mesh->color_texture();
	
	double w = std::get<0>(color_texture)->cols();
	
	return 1.0 / w;
}

double TextureSignature::v_step() {
	auto color_texture = _mesh->color_texture();

	double h = std::get<0>(color_texture)->rows();

	return 1.0 / h;
}

std::shared_ptr<GeodesicFan> TextureSignature::TextureParameterOptimization::geodesic_fan_from_relation(const Relation& r) {
	// There seems to be no saved metrics, so generate them here
	std::shared_ptr<TextureSignature> sig = std::dynamic_pointer_cast<TextureSignature>(_optimizing_sig.lock());

	if (sig == nullptr) {
		return nullptr;
	}

	auto patch = r._patch;

	if (patch == nullptr) {
		return nullptr;
	}

	// There seems to be no saved metrics, so generate them here
	Eigen::DenseIndex centroid_vid = patch->get_centroid_vid_on_origin_mesh();
	double geodesic_radius = patch->get_geodesic_extent(centroid_vid);

	auto dem = patch->discrete_exponential_map(centroid_vid);

	// Fans are based off of value inputs, so can't be pre-computed
	auto fan = std::make_shared<GeodesicFan>(dem, sig);

	return fan;
}