#ifndef STROKE_TRANSFER_EXP_H_
#define STROKE_TRANSFER_EXP_H_

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/access.hpp>

#include <matching/stroke_transfer.h>
#include <matching/surface_stroke.h>
#include <shape_signatures/shape_diameter_signature.h>

class StrokeTransferExperiment {
	public:
		StrokeTransferExperiment();
		StrokeTransferExperiment(std::shared_ptr<StrokeTransfer>);
		~StrokeTransferExperiment();

		std::string mesh_path() const { return _mesh_path; }
		std::shared_ptr<StrokeTransfer> reproduce();

		// This method lets cereal know which data members to serialize
		template<class Archive>
		void serialize(Archive& archive) {
			archive(CEREAL_NVP(_mesh_path), CEREAL_NVP(_bc)); // serialize things by passing them to the archive
		}

	protected:
		std::string _mesh_path;
		std::vector<BarycentricCoord> _bc;
};

#endif