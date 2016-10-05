/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef CoefficientMatrices_h
#define CoefficientMatrices_h

#include <element_promotion/new_assembly/HighOrderCoefficients.h>

namespace sierra {
namespace naluUnit {

template <unsigned p>
struct CoefficientMatrices
{
  constexpr static int poly_order = p;

  CoefficientMatrices(const double* nodeLocs, const double* scsLocs)
  : scsDeriv(HighOrderCoefficients::scs_derivative_weights<poly_order>(nodeLocs, scsLocs)),
    scsInterp(HighOrderCoefficients::scs_interpolation_weights<poly_order>(nodeLocs, scsLocs)),
    nodalWeights(HighOrderCoefficients::nodal_integration_weights<poly_order>(nodeLocs, scsLocs)),
    nodalDeriv(HighOrderCoefficients::nodal_derivative_weights<poly_order>(nodeLocs)),
    linear_nodal_interp(HighOrderCoefficients::linear_nodal_interpolation_weights<poly_order>(nodeLocs)),
    linear_scs_interp(HighOrderCoefficients::linear_scs_interpolation_weights<poly_order>(scsLocs)) {};

  CoefficientMatrices()
  : scsDeriv(HighOrderCoefficients::scs_derivative_weights<poly_order>()),
    scsInterp(HighOrderCoefficients::scs_interpolation_weights<poly_order>()),
    nodalWeights(HighOrderCoefficients::nodal_integration_weights<poly_order>()),
    nodalDeriv(HighOrderCoefficients::nodal_derivative_weights<poly_order>()),
    linear_nodal_interp(HighOrderCoefficients::linear_nodal_interpolation_weights<poly_order>()),
    linear_scs_interp(HighOrderCoefficients::linear_scs_interpolation_weights<poly_order>()) {};

  const typename CoefficientViews<poly_order>::scs_matrix_array scsDeriv;
  const typename CoefficientViews<poly_order>::scs_matrix_array scsInterp;
  const typename CoefficientViews<poly_order>::nodal_matrix_array nodalWeights;
  const typename CoefficientViews<poly_order>::nodal_matrix_array nodalDeriv;
  const typename CoefficientViews<poly_order>::linear_nodal_matrix_array linear_nodal_interp;
  const typename CoefficientViews<poly_order>::linear_scs_matrix_array linear_scs_interp;
};

} // namespace naluUnit
} // namespace Sierra

#endif
