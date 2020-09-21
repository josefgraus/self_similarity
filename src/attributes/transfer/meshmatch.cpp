#include "meshmatch.h"

#include <random>

#include <igl\decimate.h>

#include <shape_signatures/heat_kernel_signature.h>

struct MergeNode {
	MergeNode(Eigen::DenseIndex vid, std::shared_ptr<MergeNode> parent, std::shared_ptr<MergeNode> child1, std::shared_ptr<MergeNode> child2): _vid(vid), _parent(parent), _child1(child1), _child2(child2) { }

	Eigen::DenseIndex _vid;
	std::shared_ptr<MergeNode> _parent;
	std::shared_ptr<MergeNode> _child1;
	std::shared_ptr<MergeNode> _child2;
};

MeshMatch::MeshMatch(std::shared_ptr<Component> source, std::shared_ptr<Component> target) {
	// Calculate target average edge length
	const trimesh::trimesh_t& halfedge_mesh = target->halfedge();
	double avg_edge_length = 0.0;
	unsigned int edge_count = 0;
	for (Eigen::DenseIndex fi = 0; fi <  target->faces().rows(); ++fi) {
		for (int vi = 0; vi < target->faces().cols(); ++vi) {
			Eigen::DenseIndex i = target->faces()(fi, vi);
			Eigen::DenseIndex j = target->faces()(fi, (vi + 1) % target->faces().cols());

			if (i >= j) {
				long he_index = halfedge_mesh.directed_edge2he_index(i, j);

				if (he_index >= 0 && halfedge_mesh.halfedge(he_index).opposite_he >= 0) {
					continue;
				}
			}

			avg_edge_length += (target->vertices().row(i) - target->vertices().row(j)).norm();
			edge_count++;
		}
	}

	if (edge_count != 0) {
		avg_edge_length /= edge_count;
	} else {
		throw std::logic_error("There are no edges in this mesh?!");
	}

	std::shared_ptr<HeatKernelSignature> source_hks_sig = HeatKernelSignature::instantiate(source->origin_mesh(), 8); // 8 steps per MeshMatch paper
	std::shared_ptr<TextureSignature> source_texture_sig = TextureSignature::instantiate(source->origin_mesh());
	std::shared_ptr<HeatKernelSignature> target_hks_sig = HeatKernelSignature::instantiate(target->origin_mesh(), 8);
	std::shared_ptr<TextureSignature> target_texture_sig = TextureSignature::instantiate(target->origin_mesh());

	std::cout << "Preprocess source fans..." << std::endl;
	auto source_fans = preprocess_fans(source, source_hks_sig, source_texture_sig, avg_edge_length / 12.0);

	/*for (auto sfan : source_fans) {
		std::cout << "vid " << sfan.first << std::endl;
		std::cout << *sfan.second << std::endl;
	}*/
	
	auto multires_mesh = decimate_target(target);

	std::cout << "Preprocess target fans..." << std::endl;
	std::vector<std::map<Eigen::DenseIndex, std::shared_ptr<GeodesicFan>>> target_fans;
	for (std::size_t i = 0; i < std::get<1>(multires_mesh).size(); ++i) {
		target_fans.push_back(preprocess_fans(std::get<1>(multires_mesh)[i], target_hks_sig, target_texture_sig, avg_edge_length / 12.0));

		/*for (auto tfan : target_fans[i]) {
			std::cout << "vid " << tfan.first << std::endl;
			std::cout << *tfan.second << std::endl;
		}*/
	}

	// Load source texture and create a copy of it for the target
	auto source_texture = source->color_texture();
	auto target_texture = target->color_texture();

	// Create Nearest-Neighbor field across target mesh levels, updating and refining according to MeshMatch algorithm
	std::vector<std::unordered_map<Eigen::DenseIndex, Eigen::DenseIndex>> NNF;

	// Randomly initialize
	std::cout << "Randomly initialize NNF..." << std::endl;
	for (std::size_t i = 0; i < std::get<1>(multires_mesh).size(); ++i) {
		std::shared_ptr<Component> target_level = std::get<1>(multires_mesh)[i];
		NNF.push_back(std::unordered_map<Eigen::DenseIndex, Eigen::DenseIndex>());

		std::random_device rd;   // Will be used to obtain a seed for the random number engine
		std::mt19937 fgen(rd()); // Standard mersenne_twister_engine seeded with rd()
		std::mt19937 vgen(rd()); // Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> fdis(0, source->faces().rows() - 1);
		std::uniform_int_distribution<> vdis(0, source->faces().cols() - 1);

		for (Eigen::DenseIndex j = 0; j < target_level->vertices().rows(); ++j) {
			Eigen::DenseIndex fid = fdis(fgen);
			Eigen::DenseIndex vid = source->faces()(fid, vdis(vgen));

			NNF.back()[target_level->vid_to_origin_mesh(j)] = source->vid_to_origin_mesh(vid);
		}
	}

	// EM loop
	double v_step = target_texture_sig->v_step();
	double u_step = target_texture_sig->u_step();

	int mesh_level = 0;
	int level_ctr = 1;
	bool fpo = false;	// Used for flipping the order in which vertices are visited
	double radius = avg_edge_length / 3.0;

	while (mesh_level < std::get<1>(multires_mesh).size()) {
		//source_fans = preprocess_fans(source, source_hks_sig, source_texture_sig, avg_edge_length / 3.0);
		target_fans[mesh_level] = preprocess_fans(std::get<1>(multires_mesh)[mesh_level], target_hks_sig, target_texture_sig, avg_edge_length / 3.0);

		std::shared_ptr<Component> target_level = std::get<1>(multires_mesh)[mesh_level];

		// NNF propagation
		std::cout << "NNF Propagation..." << std::endl; 
		for (Eigen::DenseIndex vid = (fpo ? target_level->vertices().rows() - 1 : 0); (fpo ? vid >= 0 : vid < target_level->vertices().rows());  (fpo ? --vid :++vid)) {
			Eigen::DenseIndex ovid = target_level->vid_to_origin_mesh(vid);
			std::vector<Eigen::DenseIndex> target_one_ring = target_level->one_ring(ovid);
			
			// For each neighbor of vid, see if vid's match has a neighbor that better matches it (optimization by locality)
			for (auto tv : target_one_ring) {
				double orientation = 0.0;	// TODO: Store orientation of the match as well for texture synthesis??
				double best_match = std::numeric_limits<double>::max();
				Eigen::DenseIndex best_match_index = -1;

				std::vector<Eigen::DenseIndex> source_one_ring = source->one_ring(NNF[mesh_level][tv]);

				for (auto sv : source_one_ring) {
					double comp = target_fans[mesh_level][tv]->compare(*source_fans[sv], orientation);

					if (comp < best_match) {
						best_match = comp;
						best_match_index = sv;
					}
				}

				if (best_match_index > -1) {
					NNF[mesh_level][tv] = best_match_index;
				}
			}
		}

		fpo = !fpo;

		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);

		// NNF Randomization
		std::cout << "NNF Randomization..." << std::endl;
		for (Eigen::DenseIndex vid : target_level->vids()) {
			auto fan = target_fans[mesh_level][vid];
			std::shared_ptr<DiscreteExponentialMap> dem = fan->origin_map();
			
			double orientation = 0.0;
			double cur_comp = fan->compare(*source_fans[NNF[mesh_level][vid]], orientation);

			double dem_radius = dem->get_radius();
			for (double step = radius; step < dem_radius; step += radius) {
				Eigen::Vector2d polar_point; polar_point << step, dis(gen);

				Eigen::DenseIndex closest_vid = source_fans[NNF[mesh_level][vid]]->origin_map()->nearest_vertex_by_polar(polar_point);

				if (closest_vid > -1) {
					double comp = fan->compare(*source_fans[NNF[mesh_level][vid]], orientation);

					if (comp < cur_comp) {
						NNF[mesh_level][vid] = closest_vid;
						cur_comp = comp;
					}
				}
			}
		}

		double v_step = target_texture_sig->v_step();
		double u_step = target_texture_sig->u_step();

		// Texture Reconstruction
		std::cout << "Texture reconstruction..." << std::endl;

		for (Eigen::DenseIndex i = 0; i < target_level->faces().rows(); ++i) { 
			Eigen::DenseIndex origin_fid = target_level->fid_to_origin_mesh(i);
			Eigen::Matrix3d uv_coords;

			std::cout << "Face [ " << origin_fid << " ].. ";

			std::map<Eigen::DenseIndex, double> angles;
			for (Eigen::DenseIndex j = 0; j < 3; ++j) { // Assuming triangle -- will break if not
				Eigen::DenseIndex origin_vid = target_level->vid_to_origin_mesh(target_level->faces()(i, j));
				Eigen::VectorXd lc = target_texture_sig->lerpable_coord(origin_fid, origin_vid);
				uv_coords.row(j) << origin_vid, lc(0), lc(1);

				double orientation = 0.0;
				double comp = target_fans[mesh_level][origin_vid]->compare(*source_fans[NNF[mesh_level][origin_vid]], orientation);

				angles[origin_vid] = orientation;
			}

			double xmax = uv_coords.col(1).maxCoeff();
			double ymax = uv_coords.col(2).maxCoeff();

			long long texel_count = 0;

			for (double y = uv_coords.col(2).minCoeff(); y <= ymax; y += v_step) {
				for (double x = uv_coords.col(1).minCoeff(); x <= xmax; x += u_step) {
					Eigen::Vector2d uv; uv << x, y;

					if (!point_in_triangle(uv, uv_coords.block<1, 2>(0, 1), uv_coords.block<1, 2>(1, 1), uv_coords.block<1, 2>(2, 1))) {
						continue;
					}

					// Find barycentric coordinates of the point within the triangle
					Eigen::Vector2d v0 = (uv_coords.block<1, 2>(1, 1) - uv_coords.block<1, 2>(0, 1)).transpose();
					Eigen::Vector2d v1 = (uv_coords.block<1, 2>(2, 1) - uv_coords.block<1,2>(0, 1)).transpose();
					Eigen::Vector2d v2 = uv - uv_coords.block<1, 2>(0, 1).transpose();
					double d00 = v0.dot(v0);
					double d01 = v0.dot(v1);
					double d11 = v1.dot(v1);
					double d20 = v2.dot(v0);
					double d21 = v2.dot(v1);
					double denom = d00 * d11 - d01 * d01;
					double v = (d11 * d20 - d01 * d21) / denom;
					double w = (d00 * d21 - d01 * d20) / denom;
					double u = 1.0f - v - w;

					if (u + v + w - 1.0 > std::numeric_limits<double>::epsilon()) {
						throw std::domain_error("Final texture construction: Invalid barycentric coordinates!");
					}

					// Find polar coordinates on the discrete exponential map associated with each target face vertex's geodesic fan	
					std::vector<Eigen::DenseIndex> tvids = { static_cast<Eigen::DenseIndex>(uv_coords(0,0)), static_cast<Eigen::DenseIndex>(uv_coords(1,0)), static_cast<Eigen::DenseIndex>(uv_coords(2,0)) };

					std::vector<std::pair<double, Eigen::Vector3d>> colors_at_range;
					for (Eigen::DenseIndex j = 0; j < 3; ++j) {
						Eigen::DenseIndex tvid = static_cast<Eigen::DenseIndex>(uv_coords(j, 0));

						std::shared_ptr<GeodesicFan> fan = target_fans[mesh_level][tvid];
						std::shared_ptr<DiscreteExponentialMap> target_dem = fan->origin_map();

						Eigen::Vector2d polar = target_dem->interpolated_polar(Eigen::Vector3d(u, v, w), tvids);

						// Rotate each polar coordinate by the best fit comparison angle from the target vertex's geodesic fan to its NNF geodesic fan
						std::shared_ptr<DiscreteExponentialMap> source_dem = source_fans[NNF[mesh_level][tvid]]->origin_map();

						polar(1) += angles[tvid];

						if (polar(1) > 2.0 * M_PI) {
							polar(1) -= 2.0 * M_PI;
						}

						// Query the NNF vertex discrete exponential map for a color
						Eigen::Vector3d source_color; source_color << source_dem->query_map_value_polar(polar, source_texture_sig).block<3,1>(0,0);

						if (source_color.sum() < 0.0) {
							// Sampled point is off the map, so there is no color contribution
							colors_at_range.emplace_back(std::pair<double, Eigen::Vector3d>(std::numeric_limits<double>::max(), Eigen::Vector3d::Zero()));
						} else {
							colors_at_range.emplace_back(std::pair<double, Eigen::Vector3d>((uv_coords.block<1, 2>(j, 1).transpose() - uv).norm(), source_color));
						}
					}

					// Blend all the colors together with a gaussian falloff proprotional to the radius value of the polar coordinate
					Eigen::Vector3d final_color = Eigen::Vector3d::Zero();

					double cutoff = std::max((uv_coords.block<1,2>(0 ,1) - uv_coords.block<1, 2>(1, 1)).norm(), (uv_coords.block<1, 2>(0, 1) - uv_coords.block<1, 2>(2, 1)).norm());
		
					double falloff_sum = 0.0;

					for (std::size_t i = 0; i < colors_at_range.size(); ++i) {
						colors_at_range[i].first = falloff(colors_at_range[i].first, 0.0, cutoff);
						falloff_sum += colors_at_range[i].first;
					}

					if (falloff_sum > 0.0) {
						for (auto color : colors_at_range) {
							final_color += (color.first / falloff_sum) * color.second;
						}
					} else {
						continue;
					}

					// Assign blended value to target texture uv
					target_texture_sig->set_signature_value(uv, final_color);

					texel_count++;
				}
			}

			std::cout << "Done! [ " << texel_count << " ] texels modified!" << std::endl;
		}

		std::cout << "EM Loop level [ " << mesh_level << " ], iteration [" << level_ctr << " ] complete!" << std::endl;

		// Check for convergence
		// TODO: Just going to run 5 iterations as admitted to in the PatchMatch paper and see what happens
		if (level_ctr < 5) {
			level_ctr++;
		} else {
			mesh_level++;
			level_ctr = 1;
		}
	}

	// Final texture reconstruction
	// Same as the E-step in the EM loop, but instead of a Guassian falloff, just do nearest neighbor voting and save the resultant texture
	mesh_level = std::get<1>(multires_mesh).size() - 1;

	/*NNF[mesh_level] = {
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(69,	137),
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(71,	138),
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(70,	139),
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(72,	140),
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(43,	114),
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(6,	91),
		std::pair<Eigen::DenseIndex, Eigen::DenseIndex>(7,	90)
	};*/

	std::shared_ptr<Component> target_level = std::get<1>(multires_mesh)[mesh_level];
	for (Eigen::DenseIndex i = 0; i < target_level->faces().rows(); ++i) {
		Eigen::DenseIndex origin_fid = target_level->fid_to_origin_mesh(i);
		Eigen::Matrix3d uv_coords;

		std::cout << "Face [ " << origin_fid << " ].. ";

		Eigen::MatrixXi pixel_coords(3, 2);

		std::map<Eigen::DenseIndex, double> angles;
		for (Eigen::DenseIndex j = 0; j < 3; ++j) { // Assuming triangle -- will break if not
			Eigen::DenseIndex origin_vid = target_level->vid_to_origin_mesh(target_level->faces()(i, j));
			Eigen::VectorXd lc = target_texture_sig->lerpable_coord(origin_fid, origin_vid);
			uv_coords.row(j) << origin_vid, lc(0), lc(1);

			double orientation = 0.0;
			double comp = target_fans[mesh_level][origin_vid]->compare(*source_fans[NNF[mesh_level][origin_vid]], orientation);

			angles[origin_vid] = orientation;
			std::cout << "[ " << angles[origin_vid] << " ] ";

			pixel_coords.row(j) = target_texture_sig->uv_to_pixel_coord(lc);
		}

		Eigen::DenseIndex xmax = pixel_coords.col(0).maxCoeff();
		Eigen::DenseIndex ymax = pixel_coords.col(1).maxCoeff();

		long long texel_count = 0;

		for (Eigen::DenseIndex y = pixel_coords.col(1).minCoeff(); y <= ymax; ++y) {
			for (Eigen::DenseIndex x = pixel_coords.col(0).minCoeff(); x <= xmax; ++x) {
				Eigen::Vector2d uv = target_texture_sig->pixel_coord_to_uv(Eigen::Vector2i(x, y));

				if (!point_in_triangle(uv, uv_coords.block<1, 2>(0, 1), uv_coords.block<1, 2>(1, 1), uv_coords.block<1, 2>(2, 1))) {
					continue;
				}

				// Find barycentric coordinates of the point within the triangle
				// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
				Eigen::Vector2d v0 = (uv_coords.block<1, 2>(1, 1) - uv_coords.block<1, 2>(0, 1)).transpose();
				Eigen::Vector2d v1 = (uv_coords.block<1, 2>(2, 1) - uv_coords.block<1, 2>(0, 1)).transpose();
				Eigen::Vector2d v2 = uv - uv_coords.block<1, 2>(0, 1).transpose();
				double d00 = v0.dot(v0);
				double d01 = v0.dot(v1);
				double d11 = v1.dot(v1);
				double d20 = v2.dot(v0);
				double d21 = v2.dot(v1);
				double denom = d00 * d11 - d01 * d01;
				double v = (d11 * d20 - d01 * d21) / denom;
				double w = (d00 * d21 - d01 * d20) / denom;
				double u = 1.0f - v - w;

				if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || w < 0.0 || w > 1.0) {
					// coordinate not in triangle
					continue;
				}

				// Find polar coordinates on the discrete exponential map associated with each target face vertex's geodesic fan	
				std::vector<Eigen::DenseIndex> tvids = { static_cast<Eigen::DenseIndex>(uv_coords(0,0)), static_cast<Eigen::DenseIndex>(uv_coords(1,0)), static_cast<Eigen::DenseIndex>(uv_coords(2,0)) };

				std::vector<std::pair<double, Eigen::Vector3d>> colors_at_range;
				for (Eigen::DenseIndex j = 0; j < 3; ++j) {
					Eigen::DenseIndex tvid = static_cast<Eigen::DenseIndex>(uv_coords(j, 0));

					std::shared_ptr<GeodesicFan> fan = target_fans[mesh_level][tvid];
					std::shared_ptr<DiscreteExponentialMap> target_dem = fan->origin_map();

					Eigen::Vector2d polar = target_dem->interpolated_polar(Eigen::Vector3d(u, v, w), tvids);

					// Rotate each polar coordinate by the best fit comparison angle from the target vertex's geodesic fan to its NNF geodesic fan
					std::shared_ptr<DiscreteExponentialMap> source_dem = source_fans[NNF[mesh_level][tvid]]->origin_map();

					polar(1) += angles[tvid];

					if (polar(1) > 2.0 * M_PI) {
						polar(1) -= 2.0 * M_PI;
					}

					// Query the NNF vertex discrete exponential map for a color
					Eigen::Vector3d source_color; source_color << source_dem->query_map_value_polar(polar, source_texture_sig).block<3, 1>(0, 0);

					if (source_color.sum() < 0.0) {
						// Sampled point is off the map, so there is no color contribution
						colors_at_range.emplace_back(std::pair<double, Eigen::Vector3d>(std::numeric_limits<double>::max(), Eigen::Vector3d::Zero()));
					} else {
						colors_at_range.emplace_back(std::pair<double, Eigen::Vector3d>((uv_coords.block<1, 2>(j, 1).transpose() - uv).norm(), source_color));
					}
				}

				// Blend all the colors together with a gaussian falloff proprotional to the radius value of the polar coordinate
				Eigen::Vector3d final_color = Eigen::Vector3d::Zero();

				double color_dist = std::numeric_limits<double>::max();
				Eigen::DenseIndex color_index = -1;

				for (std::size_t k = 0; k < colors_at_range.size(); ++k) {
					if (colors_at_range[k].first < color_dist) {
						color_dist = colors_at_range[k].first;
						color_index = k;
					}
				}

				if (color_index >= 0) {
					final_color += colors_at_range[color_index].second;
				} else {
					std::cout << "Bad Color Index!" << std::endl;
					continue;
				}

				// Assign blended value to target texture uv
				target_texture_sig->set_signature_value(uv, final_color);

				texel_count++;
			}
		}

		std::cout << ".. Done!" << std::endl;
	}

	std::cout << "Final Texture Finished!" << std::endl;

	std::cout << "Final vertex assignments [ source -> target ]:" << std::endl;

	for (auto match : NNF.back()) {
		std::cout << "[ " << match.second << " ] -> [ " << match.first << " ]" << std::endl;
	}
}

MeshMatch::~MeshMatch() {

}

std::tuple<std::vector<std::shared_ptr<MergeNode>>, std::vector<std::shared_ptr<Component>>> MeshMatch::decimate_target(std::shared_ptr<Component> target) {
	// Extract component as its own mesh for decimation
	std::tuple<std::vector<std::shared_ptr<MergeNode>>, std::vector<std::shared_ptr<Component>>> multires_mesh;

	std::vector<std::shared_ptr<MergeNode>> vertex_tree;
	vertex_tree.resize(target->vids().size());
	for (Eigen::DenseIndex i = 0; i < vertex_tree.size(); ++i) {
		vertex_tree[i] = std::make_shared<MergeNode>(i, nullptr, nullptr, nullptr);
	}

	// According to MeshMatch paper, meshes should be decimated to about 128 vertices or fewer
	if (target->vids().size() <= 128) {
		// Component is already coarse enough!
		std::vector<std::shared_ptr<Component>> dec;
		dec.push_back(target);
		multires_mesh = std::make_tuple(vertex_tree, dec);

		return multires_mesh;
	}

	// TODO: How well does decimation help performance for meshes with fewer than 1000 faces? Components used shouldn't breach that limit. Maybe increase the above threshold?
	throw std::logic_error("Proper tracked decimation is not yet implemented!");
}

std::map<Eigen::DenseIndex, std::shared_ptr<GeodesicFan>> MeshMatch::preprocess_fans(std::shared_ptr<Component> component, std::shared_ptr<HeatKernelSignature> hks_sig, std::shared_ptr<TextureSignature> texture_sig, double radius_step) {	
	std::map<Eigen::DenseIndex, std::shared_ptr<GeodesicFan>> fans;

	// Preprocess the geodesic fan for each vertex in the component
	for (Eigen::DenseIndex vid : component->vids()) {
		double radius = 3.0 * radius_step;

		// TODO: Will the distortion introduced by DEM cause issues with large components that loop around?
		std::shared_ptr<DiscreteExponentialMap> dem = component->discrete_exponential_map(vid);

		// MeshMatch uses a hybrid comparision feature vector made up of 8 HKS levels, the normal dotted with the prescribed up, and a full 18 spoke 3 level fan of sampled texture values about the vertex
		//auto hks_fan = std::make_shared<GeodesicFan>(0.0, 0.0, 0.0, dem, hks_sig, Eigen::Vector3d::UnitZ());
		auto hks_fan = std::make_shared<GeodesicFan>(M_PI / 9.0, radius, radius_step, dem, hks_sig, Eigen::Vector3d::UnitZ());
		auto texture_fan = std::make_shared<GeodesicFan>(M_PI / 9.0, radius, radius_step, dem, texture_sig);

		// Combine these two fans into a (18 spoke * 3 level * 3 color channel) + 8 HKS value + 1 dot product = 171 dimension feature vector
		hks_fan->layer_over(texture_fan);

		fans.insert(std::pair<Eigen::DenseIndex, std::shared_ptr<GeodesicFan>>(vid, texture_fan));
		//fans.insert(std::pair<Eigen::DenseIndex, std::shared_ptr<GeodesicFan>>(vid, hks_fan));
	}

	return fans;
}

// Quick point-in-triangle test taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
double MeshMatch::sign(Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3) const {
	return (p1(0) - p3(0)) * (p2(1) - p3(1)) - (p2(0) - p3(0)) * (p1(1) - p3(1));
}

bool MeshMatch::point_in_triangle(Eigen::Vector2d pt, Eigen::Vector2d v1, Eigen::Vector2d v2, Eigen::Vector2d v3) const {
	bool b1, b2, b3;

	b1 = sign(pt, v1, v2) < std::numeric_limits<double>::epsilon();
	b2 = sign(pt, v2, v3) < std::numeric_limits<double>::epsilon();
	b3 = sign(pt, v3, v1) < std::numeric_limits<double>::epsilon();

	return ((b1 == b2) && (b2 == b3));
}


double MeshMatch::falloff(double t, double mean, double width) {
	t -= mean;

	double f;
	const double a = 1.0;
	t = t / width;

	double p = std::fabs(t);

	if (p < a / 3.0) {
		f = 3.0 * (p*p) / (a*a);
	} else if (p < a) {
		f = -1.5 * (p*p) / (a*a) + 3.0 * p / a - 0.5;
	} else {
		f = 1.0;
	}

	return 1.0 - f;
}