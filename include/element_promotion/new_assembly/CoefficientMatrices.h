/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef CoefficientMatrices_h
#define CoefficientMatrices_h

#include <element_promotion/QuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <stk_util/environment/ReportHandler.hpp>

#include <KokkosInterface.h>
#include <Teuchos_LAPACK.hpp>

namespace sierra {
namespace naluUnit {
namespace CoefficientMatrices {
/* Computes 1D coefficient matrices (e.g. for the derivative) for CVFEM */

template <unsigned poly_order>
typename CoefficientViews<poly_order>::nodal_matrix_array
nodal_integration_weights(
  const double* nodeLocs,
  const double* scsLocs)
{
  // Compute integration weights for CVFEM
  // This routine calculates \int_{x_{scs,i}}^{x_{scs,i+1}} l_j(x') dx',
  // with x_scs padded to include the end-points -1 and +1 by using a
  // "moment-matching" algorithm.  Alternatively, one could determine the monomial
  // coefficients for the Lagrange polynomials and use those to integrate

  constexpr unsigned nodes1D = poly_order + 1;
  constexpr unsigned nodesPerElement = (poly_order + 1) * (poly_order + 1);

  using nodal_matrix_array = typename CoefficientViews<poly_order>::nodal_matrix_array;

  nodal_matrix_array weightLHS("vandermonde matrix");
  for (unsigned j = 0; j < nodes1D; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      weightLHS(j,i) = std::pow(nodeLocs[j], i);
    }
  }

  nodal_matrix_array weights("nodal integration weighting for each scv");
  // each node has a separate RHS
  for (unsigned i = 0; i < nodes1D; ++i) {
    weights(0,i) = (std::pow(scsLocs[0], i + 1) - std::pow(-1.0, i + 1)) / (i + 1.0);
  }

  for (unsigned j = 1; j < nodes1D-1; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      weights(j,i) = (std::pow(scsLocs[j], i + 1) - std::pow(scsLocs[j-1], i + 1)) / (i + 1.0);
    }
  }

  for (unsigned i = 0; i < nodes1D; ++i) {
    weights(poly_order,i) = (std::pow(+1.0, i + 1) - std::pow(scsLocs[poly_order-1], i + 1)) / (i + 1.0);
  }

  int info = 1;
  int ipiv[nodesPerElement];
  Teuchos::LAPACK<int, double>().GESV(nodes1D, nodes1D,
    &weightLHS(0,0), nodes1D,
    ipiv,
    &weights(0,0), nodes1D,
    &info
  );
  ThrowRequire(info == 0);

  // GESV overwrites the RHS with the solution
  return weights;
}
//--------------------------------------------------------------------------
template < unsigned poly_order >
typename CoefficientViews<poly_order>::scs_matrix_array
scs_interpolation_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  typename QuadViews<poly_order>::nodal_scalar_array scsInterp("subcontrol surface interpolation matrix");

  auto basis1D = Lagrange1D(nodeLocs, poly_order);
  for (unsigned j = 0; j < poly_order; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      scsInterp(j,i) = basis1D.interpolation_weight(scsLocs[j], i);
    }
  }
  return scsInterp;
}
//--------------------------------------------------------------------------
template < unsigned poly_order >
typename CoefficientViews<poly_order>::scs_matrix_array
scs_derivative_weights(
  const double* nodeLocs,
  const double* scsLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  typename QuadViews<poly_order>::nodal_scalar_array scsDeriv("subcontrol surface derivative matrix");

  auto basis1D = Lagrange1D(nodeLocs, poly_order);
  for (unsigned j = 0; j < poly_order; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      scsDeriv(j,i) = basis1D.derivative_weight(scsLocs[j], i);
    }
  }
  return scsDeriv;
}
////--------------------------------------------------------------------------
template < unsigned poly_order >
typename CoefficientViews<poly_order>::nodal_matrix_array
nodal_derivative_weights(const double* nodeLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  typename QuadViews<poly_order>::nodal_scalar_array nodalDeriv("nodal derivative matrix");

  auto basis1D = Lagrange1D(nodeLocs, poly_order);
  for (unsigned j = 0; j < nodes1D; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      nodalDeriv(j,i) = basis1D.derivative_weight(nodeLocs[j],i);
    }
  }
  return nodalDeriv;
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename QuadViews<poly_order>::nodal_scalar_array
nodal_integration_weights()
{
  std::vector<double> nodeLocs; std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  return nodal_integration_weights<poly_order>(nodeLocs.data(), scsLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename QuadViews<poly_order>::nodal_scalar_array
nodal_derivative_weights()
{
  std::vector<double> nodeLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);

  return nodal_derivative_weights<poly_order>(nodeLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientViews<poly_order>::scs_matrix_array
scs_derivative_weights()
{
  std::vector<double> nodeLocs;  std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  return scs_derivative_weights<poly_order>(nodeLocs.data(), scsLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientViews<poly_order>::scs_matrix_array
scs_interpolation_weights()
{
  std::vector<double> nodeLocs; std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  return scs_interpolation_weights<poly_order>(nodeLocs.data(), scsLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientViews<poly_order>::linear_scs_matrix_array
linear_scs_interpolation_weights()
{

  std::vector<double> nodeLocs; std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  typename CoefficientViews<poly_order>::linear_scs_matrix_array
  linear_scs_interp("linear interpolants evaluated at scs locations");

  for (unsigned j = 0; j < poly_order; ++j) {
    linear_scs_interp(0,j) = 0.5*(1 - scsLocs[j]);
    linear_scs_interp(1,j) = 0.5*(1 + scsLocs[j]);
  }

  return linear_scs_interp;
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientViews<poly_order>::linear_nodal_matrix_array
linear_nodal_interpolation_weights()
{
  std::vector<double> nodeLocs; std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  typename CoefficientViews<poly_order>::linear_nodal_matrix_array
  linear_nodal_interp("linear interpolants evaluated at node locations");

  for (unsigned j = 0; j < poly_order; ++j) {
    linear_nodal_interp(0,j) = 0.5*(1 - nodeLocs[j]);
    linear_nodal_interp(1,j) = 0.5*(1 + nodeLocs[j]);
  }
  linear_nodal_interp(0,poly_order) = 0.5*(1-nodeLocs[poly_order]);
  linear_nodal_interp(1,poly_order) = 0.5*(1+nodeLocs[poly_order]);

  return linear_nodal_interp;
}

//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientViews<poly_order>::linear_scs_matrix_array
linear_scs_interpolation_weights(const double* scsLocs)
{
  typename CoefficientViews<poly_order>::linear_scs_matrix_array
  linear_scs_interp("linear interpolants evaluated at scs locations");

  for (unsigned j = 0; j < poly_order; ++j) {
    linear_scs_interp(0,j) = 0.5*(1 - scsLocs[j]);
    linear_scs_interp(1,j) = 0.5*(1 + scsLocs[j]);
  }

  return linear_scs_interp;
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientViews<poly_order>::linear_nodal_matrix_array
linear_nodal_interpolation_weights(const double* nodeLocs)
{
  typename CoefficientViews<poly_order>::linear_nodal_matrix_array
  linear_nodal_interp("linear interpolants evaluated at node locations");

  for (unsigned j = 0; j < poly_order; ++j) {
    linear_nodal_interp(0,j) = 0.5*(1 - nodeLocs[j]);
    linear_nodal_interp(1,j) = 0.5*(1 + nodeLocs[j]);
  }
  linear_nodal_interp(0,poly_order) = 0.5*(1-nodeLocs[poly_order]);
  linear_nodal_interp(1,poly_order) = 0.5*(1+nodeLocs[poly_order]);

  return linear_nodal_interp;
}
//--------------------------------------------------------------------------

}

} // namespace naluUnit
} // namespace Sierra

#endif
