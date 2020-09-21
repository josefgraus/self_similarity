#ifndef TEXTURE_SIGNATURE_H_
#define TEXTURE_SIGNATURE_H_

#include <memory>
#include <string>

#include <Eigen/Sparse>

#include <geometry/mesh.h>
#include <shape_signatures/shape_signature.h>
#include <geometry/component.h>

class TextureSignature : public ShapeSignature {
	public:
		virtual ~TextureSignature();

		static std::shared_ptr<TextureSignature> instantiate(std::shared_ptr<Mesh> mesh);

		virtual const Eigen::VectorXd lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid);
		virtual Eigen::VectorXd lerpable_to_signature_value(const Eigen::VectorXd& lerped);

		Eigen::VectorXi uv_to_pixel_coord(Eigen::Vector2d uv);
		Eigen::VectorXd pixel_coord_to_uv(Eigen::Vector2i pixel_coord);

		bool set_signature_value(Eigen::Vector2d uv, Eigen::VectorXd value);
		virtual Eigen::VectorXd get_signature_values(double index);	// Get raw matrix of signature values
		virtual unsigned long feature_count();						// The number of "points" the signature defines values for across the mesh 
		virtual unsigned long feature_dimension();					// The dimensionality of the feature vector defined for each "point" the signature defines values for across the mesh 
		virtual double lower_bound();
		virtual double upper_bound();
		double u_step();
		double v_step();
		virtual Eigen::MatrixXd sig_steps();
		virtual const double step_width(double param);

		virtual double param_lower_bound();
		virtual double param_upper_bound();
		virtual void resample_at_param(double param);

	protected:
		TextureSignature(const std::shared_ptr<Mesh> mesh);

		class TextureParameterOptimization : public ParameterOptimization {
			public:
				TextureParameterOptimization();
				~TextureParameterOptimization();

				virtual double value(const TVector &x);
				virtual TVector upperBound() const;
				virtual TVector lowerBound() const;

				virtual Eigen::MatrixXd param_steps(unsigned int steps);
				virtual std::shared_ptr<GeodesicFan> geodesic_fan_from_relation(const Relation& r);
		};
};

#endif