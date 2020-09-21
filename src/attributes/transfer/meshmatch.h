#ifndef MESHMATCH_H_
#define MESHMATCH_H_

#include <memory>
#include <map>

#include <Eigen\Dense>

#include <geometry/component.h>
#include <matching/geodesic_fan.h>
#include <shape_signatures/heat_kernel_signature.h>
#include <shape_signatures/texture_signature.h>
 
struct MergeNode;

class MeshMatch {
	public:
		MeshMatch(std::shared_ptr<Component> source, std::shared_ptr<Component> target);
		~MeshMatch();

		//Avoid transfer_color_texture();
		//void transfer_geometric_detail();

	private:
		// Duplicate code from QuadraticBump
		double falloff(double t, double mean, double width);

		// Duplicate code from DiscreteExponentialMap
		bool point_in_triangle(Eigen::Vector2d pt, Eigen::Vector2d v1, Eigen::Vector2d v2, Eigen::Vector2d v3) const;
		double sign(Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3) const;

		std::tuple<std::vector<std::shared_ptr<MergeNode>>, std::vector<std::shared_ptr<Component>>> decimate_target(std::shared_ptr<Component> target);
		std::map<Eigen::DenseIndex, std::shared_ptr<GeodesicFan>> preprocess_fans(std::shared_ptr<Component> component, std::shared_ptr<HeatKernelSignature> hks_sig, std::shared_ptr<TextureSignature> texture_sig, double radius_step);	// MeshMatch uses 3 levels and 18 spokes

		std::map<Eigen::DenseIndex, Eigen::DenseIndex> _NNF;
};

#endif