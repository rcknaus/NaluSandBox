/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderGeometryHex_h
#define HighOrderGeometryHex_h

#include <element_promotion/new_assembly/HighOrderOperatorsHex.h>
#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <element_promotion/new_assembly/DirectionEnums.h>
#include <TopologyViews.h>

#include <NaluEnv.h>

namespace sierra {
namespace naluUnit {
namespace HighOrderMetrics
{
  // compute the cofactors of a 3x3 matrix
  template <unsigned strideA = 1, unsigned strideB = 1>
  double cofactor_matrix33(
    const double* __restrict__ matrix,
    double* __restrict__ cofactor)
  {
    cofactor[0*strideB] = matrix[4*strideA] * matrix[8*strideA] - matrix[7*strideA] * matrix[5*strideA];
    cofactor[1*strideB] = matrix[6*strideA] * matrix[5*strideA] - matrix[3*strideA] * matrix[8*strideA];
    cofactor[2*strideB] = matrix[3*strideA] * matrix[7*strideA] - matrix[6*strideA] * matrix[4*strideA];
    cofactor[3*strideB] = matrix[7*strideA] * matrix[2*strideA] - matrix[1*strideA] * matrix[8*strideA];
    cofactor[4*strideB] = matrix[0*strideA] * matrix[8*strideA] - matrix[6*strideA] * matrix[2*strideA];
    cofactor[5*strideB] = matrix[6*strideA] * matrix[1*strideA] - matrix[0*strideA] * matrix[7*strideA];
    cofactor[6*strideB] = matrix[1*strideA] * matrix[5*strideA] - matrix[4*strideA] * matrix[2*strideA];
    cofactor[7*strideB] = matrix[3*strideA] * matrix[2*strideA] - matrix[0*strideA] * matrix[5*strideA];
    cofactor[8*strideB] = matrix[0*strideA] * matrix[4*strideA] - matrix[3*strideA] * matrix[1*strideA];

    double detj =
         ( matrix[0*strideA] * cofactor[0*strideB]
         + matrix[1*strideA] * cofactor[1*strideB]
         + matrix[2*strideA] * cofactor[2*strideB] );


    ThrowRequire(detj > 0.0);
    return (1.0 / detj);
  }

  template<unsigned poly_order>
  void compute_diffusion_metric(
    const CoefficientMatrices<poly_order>& mat,
    const typename HexViews<poly_order>::nodal_vector_array& coordinates,
    typename HexViews<poly_order>::scs_tensor_array& metric)
  {
    using TopoView = HexViews<poly_order>;
    typename TopoView::nodal_tensor_array jac("jacobian");
    typename TopoView::tensor_array coj("jacobian cofactors");

    double metricval = -0.5;

    // compute xhat surfaces
    HighOrderOperators::scs_xhat_grad<poly_order>(mat.scsInterp, mat.scsDeriv,  mat.nodalDeriv, coordinates, jac);
    for (unsigned k = 0; k < TopoView::nodes1D; ++k) {
      for (unsigned j = 0; j < TopoView::nodes1D; ++j) {
        for (unsigned i = 0; i < TopoView::poly_order; ++i) {
          double inv_detj = cofactor_matrix33<TopoView::nodesPerElement, 1>(&jac(0, 0, k, j, i), coj.ptr_on_device());
          metric(XH, XH, i,j,k) = -inv_detj*(coj(XH, XH) * coj(XH, XH) + coj(XH, YH) * coj(XH, YH) + coj(XH, ZH) * coj(XH, ZH));
          metric(XH, YH, i,j,k) = inv_detj*(coj(XH, XH) * coj(YH, XH) + coj(XH, YH) * coj(YH, YH) + coj(XH, ZH) * coj(YH, ZH));
          metric(XH, ZH, i,j,k) = inv_detj*(coj(XH, XH) * coj(ZH, XH) + coj(XH, YH) * coj(ZH, YH) + coj(XH, ZH) * coj(ZH, ZH));
          ThrowRequire(std::abs(metric(XH,XH,i,j,k) - metricval) < 0.01);
        }
      }
    }

    HighOrderOperators::scs_yhat_grad<poly_order>(mat.scsInterp, mat.scsDeriv, mat.nodalDeriv, coordinates, jac);
    for (unsigned k = 0; k < TopoView::nodes1D; ++k) {
      for (unsigned j = 0; j < TopoView::poly_order; ++j) {
        for (unsigned i = 0; i < TopoView::nodes1D; ++i) {
          double inv_detj = cofactor_matrix33<TopoView::nodesPerElement, 1>(&jac(0, 0, k, j, i), coj.ptr_on_device());
          metric(YH, XH, j,k,i) = inv_detj*(coj(YH, XH) * coj(XH, XH) + coj(YH, YH) * coj(XH, YH) + coj(YH, ZH) * coj(XH, ZH));
          metric(YH, YH, j,k,i) = -inv_detj*(coj(YH, XH) * coj(YH, XH) + coj(YH, YH) * coj(YH, YH) + coj(YH, ZH) * coj(YH, ZH));
          metric(YH, ZH, j,k,i) = inv_detj*(coj(YH, XH) * coj(ZH, XH) + coj(YH, YH) * coj(ZH, YH) + coj(YH, ZH) * coj(ZH, ZH));
          ThrowRequire(std::abs(metric(YH,YH,j,k,i) - metricval) < 0.01);
        }
      }
    }

    HighOrderOperators::scs_zhat_grad<poly_order>(mat.scsInterp, mat.scsDeriv, mat.nodalDeriv, coordinates, jac);
    for (unsigned k = 0; k < TopoView::poly_order; ++k) {
      for (unsigned j = 0; j < TopoView::nodes1D; ++j) {
        for (unsigned i = 0; i < TopoView::nodes1D; ++i) {
          double inv_detj = cofactor_matrix33<TopoView::nodesPerElement, 1>(&jac(0, 0, k, j, i), coj.ptr_on_device());
          metric(ZH, XH, k,j,i) = inv_detj*(coj(ZH, XH) * coj(XH, XH) + coj(ZH, YH) * coj(XH, YH) + coj(ZH, ZH) * coj(XH, ZH));
          metric(ZH, YH, k,j,i) = inv_detj*(coj(ZH, XH) * coj(YH, XH) + coj(ZH, YH) * coj(YH, YH) + coj(ZH, ZH) * coj(YH, ZH));
          metric(ZH, ZH, k,j,i) = -inv_detj*(coj(ZH, XH) * coj(ZH, XH) + coj(ZH, YH) * coj(ZH, YH) + coj(ZH, ZH) * coj(ZH, ZH));
          ThrowRequire(std::abs(metric(ZH,ZH,k,j,i) - metricval) < 0.01);
        }
      }
    }
  }
//--------------------------------------------------------------------------
  template <unsigned poly_order>
  void compute_volume_metric(
    const CoefficientMatrices<poly_order>& mat,
    typename HexViews<poly_order>::nodal_vector_array& coordinates,
    typename HexViews<poly_order>::nodal_scalar_array& vol)
  {
    // Computes det(J) at nodes using the full isoparametric formulation
    typename HexViews<poly_order>::nodal_tensor_array jac("jacobian");
    HighOrderOperators::nodal_grad<poly_order>(mat.nodalDeriv, coordinates, jac);

    for (unsigned k = 0; k <  poly_order+1; ++k) {
      for (unsigned j = 0; j <  poly_order+1; ++j) {
        for (unsigned i = 0; i <  poly_order+1; ++i) {
          const double minor1 = jac(YH,YH, k,j,i) * jac(ZH,ZH, k,j,i) - jac(YH,ZH,k,j,i) * jac(ZH,YH,k,j,i);
          const double minor2 = jac(XH,XH, k,j,i) * jac(ZH,ZH, k,j,i) - jac(XH,ZH,k,j,i) * jac(ZH,XH,k,j,i);
          const double minor3 = jac(XH,XH, k,j,i) * jac(YH,YH, k,j,i) - jac(XH,YH,k,j,i) * jac(YH,XH,k,j,i);
          vol(k,j,i) = jac(XH,XH,k,j,i) * minor1 + jac(YH,YH,k,j,i) * minor2 + jac(ZH,ZH,k,j,i) * minor3;
          ThrowRequire(std::abs(vol(k,j,i) - 0.125) < 0.01);
        }
      }
    }
  }

} // namespace HighOrderMetrics
} // namespace naluUnit
} // namespace Sierra

#endif
