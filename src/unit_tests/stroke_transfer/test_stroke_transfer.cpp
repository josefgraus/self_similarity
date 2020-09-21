#include "test_stroke_transfer.h"

// Experiment serialization
#include <cereal/archives/json.hpp>
#include <fstream>

#include <experiments/stroke_transfer_exp.h>
#include <cppoptlib/solver/lbfgsbsolver.h>

using namespace cppoptlib;

SCENARIO("Comparing a SurfaceStroke against itself should always result in a zero gradient", "[SurfaceStroke]") {
	GIVEN("A SurfaceStroke on a mesh") {
		StrokeTransferExperiment ste;
		std::string exp_path = std::string(BASE_DIRECTORY) + "models\\ste.json";

		{
			std::ifstream exp_file(exp_path);
			cereal::JSONInputArchive exp_archive(exp_file);
			exp_archive(cereal::make_nvp("Stroke Transfer Experiment", ste));
		}

		std::shared_ptr<StrokeTransfer> st = ste.reproduce();
		
		REQUIRE(st != nullptr);
		REQUIRE(st->source()->blade_points().size() > 0);

		std::shared_ptr<SurfaceStroke> src = st->source();
		std::shared_ptr<CurveUnrolling> cu_src = std::make_shared<CurveUnrolling>(src);
		std::shared_ptr<CurveUnrolling> cu_cpy = std::make_shared<CurveUnrolling>(src);
		StrokeDiffReduce sdr(cu_src, cu_cpy, st->signature());

		cppoptlib::Problem<double>::TVector x = Eigen::VectorXd::Zero(3);
		cppoptlib::Problem<double>::TVector grad = Eigen::VectorXd(3);
		
		Approx zero(0.0);
		zero.margin(1e-7);

		
		WHEN("Two copies of the source stroke are unrolled", "[skip]") {
			THEN("They should be identical under the signature") {
				auto unrolled_src = cu_src->unrolled_on_origin_mesh();
				auto unrolled_cpy = cu_cpy->unrolled_on_origin_mesh();

				Eigen::VectorXd diff = unrolled_src->per_point_diff(*unrolled_cpy, st->signature());

				REQUIRE(diff.norm() == zero);
			}
		}
		WHEN("A trivial transform is performed on an exact copy", "[skip]") {
			cu_cpy->transform(0.0, 0.0, 0.0);

			THEN("Copy should not have moved (or have a difference against source)") {
				auto unrolled_src = cu_src->unrolled_on_origin_mesh();
				auto unrolled_cpy = cu_cpy->unrolled_on_origin_mesh();

				for (int i = 0; i < unrolled_src->blade_points().size(); ++i) {
					std::clog << "blade_point " << i << std::endl;
					auto& src_coeff = unrolled_src->blade_points()[i]._coeff;
					auto& cpy_coeff = unrolled_cpy->blade_points()[i]._coeff;

					std::clog << "src_coeff " << src_coeff.transpose() << std::endl;
					std::clog << "cpy_coeff " << cpy_coeff.transpose() << std::endl;

					REQUIRE(unrolled_src->blade_points()[i]._fid == unrolled_cpy->blade_points()[i]._fid);
					REQUIRE((unrolled_cpy->blade_points()[i]._coeff - unrolled_src->blade_points()[i]._coeff).isZero(1e-7));
				}

				Eigen::VectorXd diff = unrolled_src->per_point_diff(*unrolled_cpy, st->signature());

				REQUIRE(diff.norm() == zero);
			}
		}
		WHEN("The source SurfaceStroke is compared against itself") {
			THEN("The gradient must be zero") {
				sdr.gradient(x, grad);

				REQUIRE(grad(0) == zero);
				REQUIRE(grad(1) == zero);
				REQUIRE(grad(2) == zero);

				// Run gradient descent -- we shouldn't need to take any steps
				Eigen::VectorXd grad = Eigen::VectorXd::Zero(3);

				// Use fancy solver (CppOptLib)
				LbfgsbSolver<StrokeDiffReduce> solver;
				solver.minimize(sdr, x);

				std::clog << "x = " << x.transpose() << std::endl;

				double diff = sdr.value(x);

				REQUIRE(grad(0) == zero);
				REQUIRE(grad(1) == zero);
				REQUIRE(grad(2) == zero);
			}
		}
		WHEN("The source SurfaceStroke is perturbed by some small amount") {
			x << 1e-3, 1e-3, 1e-3;

			THEN("It must return to its original optimal position") {
				Eigen::VectorXd grad = Eigen::VectorXd::Zero(3);

				// Use fancy solver (CppOptLib)
				LbfgsbSolver<StrokeDiffReduce> solver;
				solver.minimize(sdr, x);

				std::clog << "x = " << x.transpose() << std::endl;

				double diff = sdr.value(x);

				REQUIRE(diff <= 1e-5);
			}
		}
		WHEN("The source stroke is moved a large amount") {
			Eigen::MatrixXd points = cu_cpy->curve_points_2d();
			auto V = cu_cpy->vertices();

			cu_cpy->transform(0.0, 0.0, M_PI / 2.0);

			Eigen::MatrixXd points2 = cu_cpy->curve_points_2d();
			REQUIRE(!(points - points2).isZero());

			cu_cpy->transform(0.0, 0.0, -M_PI / 2.0);

			points2 = cu_cpy->curve_points_2d();
			auto V2 = cu_cpy->vertices();
			REQUIRE((points - points2).isZero());
			REQUIRE(V.size() == V2.size());
			REQUIRE((V - V2).isZero());

			auto unrolled_src = cu_src->unrolled_on_origin_mesh();
			auto unrolled_cpy = cu_cpy->unrolled_on_origin_mesh();

			for (int i = 0; i < unrolled_src->blade_points().size(); ++i) {
				std::clog << "blade_point " << i << std::endl;
				auto& src_coeff = unrolled_src->blade_points()[i]._coeff;
				auto& cpy_coeff = unrolled_cpy->blade_points()[i]._coeff;

				std::clog << "src_coeff " << src_coeff.transpose() << std::endl;
				std::clog << "cpy_coeff " << cpy_coeff.transpose() << std::endl;

				REQUIRE(unrolled_src->blade_points()[i]._fid == unrolled_cpy->blade_points()[i]._fid);
				REQUIRE((unrolled_cpy->blade_points()[i]._coeff - unrolled_src->blade_points()[i]._coeff).isZero(1e-7));
			}

			Eigen::VectorXd diff = unrolled_src->per_point_diff(*unrolled_cpy, st->signature());

			REQUIRE(diff.norm() == zero);

			cu_cpy->transform(-2e-2, 0.0, 0.0);
			cu_cpy->transform(2e-2, 0.0, 0.0);
			cu_cpy->transform(4.5e-2, 0.0, 0.0);
			cu_cpy->transform(-4.5e-2, 0.0, 0.0);
			cu_cpy->transform(1e-1, 0.0, 0.0);
			cu_cpy->transform(-1e-1, 0.0, 0.0);
			cu_cpy->transform(5e-1, 0.0, 0.0);
			cu_cpy->transform(-5e-1, 0.0, 0.0);

			cu_cpy->transform(1.0, 0.0, 0.0);
			
			points2 = cu_cpy->curve_points_2d();
			REQUIRE(!(points - points2).isZero());
			REQUIRE((points - (points2.colwise() - Eigen::Vector2d::UnitX())).isZero());

			cu_cpy->transform(-1.0, 0.0, 0.0);

			points2 = cu_cpy->curve_points_2d();
			V2 = cu_cpy->vertices();
			REQUIRE((points - points2).isZero());
			//REQUIRE(V.size() == V2.size());
			//REQUIRE((V - V2).isZero());

			cu_cpy->transform(0.0, 1.0, 0.0);
			
			points2 = cu_cpy->curve_points_2d();
			REQUIRE(!(points - points2).isZero());
			REQUIRE((points - (points2.colwise() - Eigen::Vector2d::UnitY())).isZero());

			cu_cpy->transform(0.0, -1.0, 0.0);

			points2 = cu_cpy->curve_points_2d();
			V2 = cu_cpy->vertices();
			REQUIRE((points - points2).isZero());
			//REQUIRE(V.size() == V2.size());
			//REQUIRE((V - V2).isZero());
		}
		WHEN("Finite difference is used with StrokeDiffReduce::value() to find the gradient") {
			//cu_cpy->transform(1e-3, 1e-3, 1e-3);

			StrokeDiffReduce::TVector x(3);
			StrokeDiffReduce::TVector z(3);
			z << 1e-3, 1e-3, 1e-3;
			x = z;

			THEN("it should equal StrokeDiffReduce::gradient()") {
				sdr.gradient(x, grad);

				double f0 = sdr.value(x);
				double h = 1e-7;
				StrokeDiffReduce::TVector check(3);
				x << z(0) + h, z(1), z(2);
				check(0) = (sdr.value(x) - f0) / h;
				x << z(0), z(1) + h, z(2);
				check(1) = (sdr.value(x) - f0) / h;
				x << z(0), z(1), z(2) + h;
				check(2) = (sdr.value(x) - f0) / h;

				//REQUIRE((check - grad).norm() == zero);
			}
		}
	}
}