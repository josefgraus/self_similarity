#include "stroke_transfer_exp.h"

#include <shape_signatures/shape_diameter_signature.h>

StrokeTransferExperiment::StrokeTransferExperiment() {

}

StrokeTransferExperiment::StrokeTransferExperiment(std::shared_ptr<StrokeTransfer> st) {
	_mesh_path = st->source()->origin_mesh()->model_path();
	_bc = st->source()->blade_points();
}

StrokeTransferExperiment::~StrokeTransferExperiment() {

}

std::shared_ptr<StrokeTransfer> StrokeTransferExperiment::reproduce() {
	std::shared_ptr<Mesh> mesh = Mesh::instantiate(_mesh_path);
	std::shared_ptr<ShapeDiameterSignature> sig = ShapeDiameterSignature::instantiate(mesh, 0.0);
	std::shared_ptr<StrokeTransfer> stroke_transfer = std::make_shared<StrokeTransfer>(mesh, sig);

	for (BarycentricCoord& bc : _bc) {
		stroke_transfer->add_to_source(bc._fid, bc._coeff);
	}

	return stroke_transfer;
}