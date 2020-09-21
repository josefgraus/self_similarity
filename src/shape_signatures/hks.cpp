#include "hks.h"

#include <igl/eigs.h>
#include <iostream>

template <typename DerivedVal, typename DerivedVec, typename DerivedHKS>
  inline bool hks(
	const Eigen::MatrixBase<DerivedVal> & eVecs,
	const Eigen::MatrixBase<DerivedVec> & eVals,
	double tmin,
	double tmax,
	unsigned int steps,
	Eigen::MatrixBase<DerivedHKS>& HKS) {

	  if (eVecs.rows() < 2 || eVals.rows() < eVecs.cols()) {
		  return false;
	  }

	  if (steps < 1) {
		  steps = eVals.rows();
	  }

	  double stepsize = (std::log(tmax) - std::log(tmin)) / steps;

	  HKS = Eigen::MatrixXd::Zero(eVecs.rows(), steps);
	  for (unsigned int j = 0; j < steps; ++j) {
		  double t = std::exp(std::log(tmin) + stepsize * static_cast<double>(j));
		  double exp_sum = 0.0;

		  for (unsigned int i = 1; i < eVecs.cols(); ++i) {
			  double exp = std::exp(-std::fabs(eVals(i)) * t);
			  HKS.col(j) = HKS.col(j).array() + eVecs.col(i).array().square() * exp;
			  exp_sum += exp;
		  }

		  // HKS scaling
		  HKS.col(j) = HKS.col(j) / exp_sum;
	  }

	  return true;
} 

template <typename DerivedV, typename DerivedF, typename DerivedHKS>
  inline bool hks(
	const Eigen::MatrixBase<DerivedV>& eVecs,
	const Eigen::MatrixBase<DerivedF>& eVals,
	double t,
	Eigen::MatrixBase<DerivedHKS>& HKS) {
	  HKS = Eigen::MatrixXd::Zero(eVecs.rows(), 1);
	  double exp_sum = 0.0;
	  
	  for (unsigned int i = 1; i < eVecs.cols(); ++i) {
		double exp = std::exp(-std::fabs(eVals(i)) * t);

		HKS = HKS.array() + eVecs.col(i).array().square() * exp;
		exp_sum += exp;
	  }

	  // HKS scaling
	  HKS.col(0) = HKS.col(0) / exp_sum;

	  return true;
}

 template <typename DerivedVal, typename DerivedVec, typename DerivedHKS>
   inline bool hks(
     const Eigen::MatrixBase<DerivedVal>& eVecs,
	 const Eigen::MatrixBase<DerivedVec>& eVals,
	 Eigen::MatrixBase<DerivedHKS>& HKS,
	 int steps) {
	   
	 if (eVecs.rows() < 2 || eVals.rows() < eVecs.cols()) {
	   return false;
	 }

	 if (steps < 1) {
		 steps = eVals.rows();
	 }

	 // Heat Kernel Signature
	 // [tmin, tmax] as suggested in SOG09: http://dl.acm.org/citation.cfm?id=1735603.1735621
	 double tmin = 4.0 * std::log(10) / std::fabs(eVals(eVals.rows() - 1));
	 double tmax = 4.0 * std::log(10) / std::fabs(eVals(1));

	 double stepsize = (std::log(tmax) - std::log(tmin)) / steps;

	  HKS = Eigen::MatrixXd::Zero(eVecs.rows(), steps);
	  for (unsigned int j = 0; j < steps; ++j) {
	  	double t = std::exp(std::log(tmin) + stepsize * static_cast<double>(j));
		double exp_sum = 0.0;
		
		for (unsigned int i = 1; i < eVecs.cols(); ++i) {
		  double exp = std::exp(-std::fabs(eVals(i)) * t);
			HKS.col(j) = HKS.col(j).array() + eVecs.col(i).array().square() * exp;
			exp_sum += exp;
		}

		// HKS scaling
		HKS.col(j) = HKS.col(j) / exp_sum;
	}

	return true;
  }