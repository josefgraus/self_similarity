#ifndef HKS_H_
#define HKS_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

  template <typename DerivedVal, typename DerivedVec, typename DerivedHKS>
  inline bool hks(
	  const Eigen::MatrixBase<DerivedVal>& eVecs,
	  const Eigen::MatrixBase<DerivedVec>& eVals,
	  double tmin,
	  double tmax,
	  unsigned int steps,
	  Eigen::MatrixBase<DerivedHKS>& HKS);

  template <typename DerivedVal, typename DerivedVec, typename DerivedHKS>
  inline bool hks(
	  const Eigen::MatrixBase<DerivedVal>& eVecs,
	  const Eigen::MatrixBase<DerivedVec>& eVals,
	  double t,
	  Eigen::MatrixBase<DerivedHKS>& HKS);

  template <typename DerivedV, typename DerivedF, typename DerivedHKS>
  inline bool hks(
	  const Eigen::MatrixBase<DerivedV>& eVecs,
	  const Eigen::MatrixBase<DerivedF>& eVals,
	  Eigen::MatrixBase<DerivedHKS>& HKS,
	  int steps = -1);

#include "hks.cpp"

#endif