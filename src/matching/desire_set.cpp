//#include "desire_set.h" 

#include <shape_signatures/shape_signature.h>

template <class T>
DesireSet<T>::DesireSet(T signature_index, std::weak_ptr<ShapeSignature> bound_signature): _index(signature_index), _bound_signature(bound_signature) {
}

template <class T>
DesireSet<T>::~DesireSet() {
}

template <class T>
T DesireSet<T>::signature_index() {
	return _index;
}

template <class T>
void DesireSet<T>::signature_index(T index) {
	_index = index;
}

template <class T>
Eigen::VectorXd DesireSet<T>::signature_values_at_index() {
	std::shared_ptr<ShapeSignature> sig = _bound_signature.lock();

	if (sig == nullptr) {
		return Eigen::VectorXd();
	}

	return std::move(sig->get_signature_values(_index));
}

template <class T>
std::vector<Relation> DesireSet<T>::relations() {
	return _relations;
}

template <class T>
std::weak_ptr<ShapeSignature> DesireSet<T>::bound_signature() const {
	return _bound_signature;
}

template <class T>
bool DesireSet<T>::add_relation(Relation r) {
	if (r._patch == nullptr) {
		return false;
	}

	// It makes no sense to add multiple relations for the same patch -- or contradictory ones
	if (std::find_if(_relations.begin(), _relations.end(), [&r](const Relation& rel) { return r._patch == rel._patch; }) != _relations.end()) {
		return false;
	}

	_relations.push_back(r);

	return true;
}

template <class T>
bool DesireSet<T>::remove_relation(Relation r) {
	if (r._patch == nullptr) {
		return false;
	}

	auto it = std::find_if(_relations.begin(), _relations.end(), [&r](const Relation& rel) { return r._patch == rel._patch; });

	if (it == _relations.end()) {
		return false;
	}

	_relations.erase(it);

	return true;
}

template <class T>
void DesireSet<T>::add_quadratic_bump(QuadraticBump<T> bump) {
	_bumps.emplace_back(bump);
}

template <class T>
double DesireSet<T>::energy_shift_by_index(T index) {
	std::shared_ptr<ShapeSignature> sig = std::dynamic_pointer_cast<ShapeSignature>(_bound_signature.lock());
	
	if (sig == nullptr) {
		throw std::logic_error("Bound signature is no longer valid!");
	}

	Eigen::VectorXd shift = Eigen::VectorXd::Zero(sig->feature_count());

	for (auto bump : _bumps) {
		shift(bump.feature_index()) += bump.energy_shift_by_paramter(index);
	}

	return shift;
}