#ifndef DESIRE_SET
#define DESIRE_SET

#include <memory>
#include <vector>

#include <matching/quadratic_bump.h>
#include <matching/constrained_relation_solver.h>

class ShapeSignature;
struct Relation;

template <class T>
class DesireSet {
	public:
		DesireSet(T signature_index, std::weak_ptr<ShapeSignature> bound_signature);
		~DesireSet();

		T signature_index();
		void signature_index(T index);

		Eigen::VectorXd signature_values_at_index();

		std::vector<Relation> relations();

		std::weak_ptr<ShapeSignature> bound_signature() const;

		bool add_relation(Relation r);
		bool remove_relation(Relation r);

		double energy_shift_by_index(T index);

		void add_quadratic_bump(QuadraticBump<T> quadratic_bump);

	private:
		DesireSet();

		T _index;
		std::weak_ptr<ShapeSignature> _bound_signature;
		std::vector<Relation> _relations;
		std::vector<QuadraticBump<T>> _bumps;
};

#include <matching/desire_set.cpp>

#endif