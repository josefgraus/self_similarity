#ifndef CONSTRAINED_RELATION_SOLVER_H_
#define CONSTRAINED_RELATION_SOLVER_H_

#include <vector>
#include <memory>
#include <map>

#include <utilities/multithreading.h>
#include <geometry/geometry.h>

class Patch;
class Threshold;
class ShapeSignature;

template <class T> class DesireSet;

struct Relation {
	public:
		enum class Designation {
			Include,
			Exclude
		};

		Relation(std::shared_ptr<Patch> patch, Designation designation): _patch(patch), _designation(designation) { };
		Relation(BarycentricCoord bc, Designation designation): _patch(nullptr), _bc(bc), _designation(designation) { };

		std::shared_ptr<Patch> _patch;
		BarycentricCoord _bc;
		Designation _designation;
};

class CRSolver {
	public:
		CRSolver();
		~CRSolver();

		bool add_signature(std::shared_ptr<ShapeSignature> sig);
		std::vector<std::shared_ptr<ShapeSignature>> signatures() const;

		bool add_relation(Relation rel, unsigned int desire_set = 0);
		std::vector<Relation> relations(unsigned int desire_set = 0) const;

		std::vector<std::shared_ptr<Threshold>> solve(std::vector<double>& indices, unsigned int desire_set = 0) const;

		// Debug and Analysis
		//virtual bool patch_entanglement(std::string path, unsigned int set = 0);

	protected:
		std::shared_ptr<Threshold> optimal_relation_threshold(double optimal_t, unsigned int set) const;
		void exception_interval(Threshold interval, double mean, unsigned int set) const;

		std::map<unsigned int, std::vector<Relation>> _relations;
		std::map<unsigned int, std::shared_ptr<DesireSet<double>>> _desire_sets;
		std::vector<std::shared_ptr<ShapeSignature>> _sigs;
};

#endif
