/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderGeometry_h
#define HighOrderGeometry_h

#include <element_promotion/new_assembly/HighOrderOperators.h>
#include <KokkosInterface.h>

#include <tuple>
#include <vector>

namespace sierra {
namespace naluUnit {

template < unsigned poly_order>
class HighOrderGeometryQuad
{
  //==========================================================================
  // Class Definition
  //==========================================================================
  // HighOrderGeometryQuad - Computes coefficients involving mesh
  // geometric quantities for the assembly for quadrilateral elements
  //==========================================================================
public:
  //--------------------------------------------------------------------------
  HighOrderGeometryQuad(const double* nodeLocs, const double* scsLocs)
: ops(nodeLocs,scsLocs),
  jac("jacobian"),
  linear_nodal_interp(CoefficientMatrices::linear_nodal_interpolation_weights<poly_order>(nodeLocs)),
  linear_scs_interp(CoefficientMatrices::linear_scs_interpolation_weights<poly_order>(scsLocs))
 { };
  //--------------------------------------------------------------------------
  HighOrderGeometryQuad() :
    ops(),
    jac("jacobian"),
    linear_nodal_interp(CoefficientMatrices::linear_nodal_interpolation_weights<poly_order>()),
    linear_scs_interp(CoefficientMatrices::linear_scs_interpolation_weights<poly_order>())
  { };
  //--------------------------------------------------------------------------
  void diffusion_metric(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::scs_tensor_array& metric)
  {
    /*
     * Metric for the full isoparametric mapping (supports curved elements)
     * The metric is a combination of the inverse of the Jacobian and the area-vector (A^T J^-1),
     */

    ops.scs_xhat_grad(coordinates, jac);

    for (unsigned j = 0; j < poly_order; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        double inv_detj = 1.0 / (jac(XH,YH, j,i) * jac(YH,XH, j,i) - jac(XH,XH,j,i) * jac(YH,YH,j,i));
        metric(XH,XH,j,i) =  inv_detj * (jac(XH,YH,j,i) * jac(XH,YH,j,i) + jac(YH,YH,j,i) * jac(YH,YH,j,i));
        metric(XH,YH,j,i) = -inv_detj * (jac(XH,XH,j,i) * jac(XH,YH,j,i) + jac(YH,XH,j,i) * jac(YH,YH,j,i));

      }
    }

    ops.scs_yhat_grad(coordinates, jac);

    for (unsigned j = 0; j < poly_order; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        double inv_detj = 1.0 / (jac(XH,YH, j,i) * jac(YH,XH, j,i) - jac(XH,XH,j,i) * jac(YH,YH,j,i));
        metric(YH,XH,j,i) = -inv_detj * (jac(XH,XH,j,i) * jac(XH,YH,j,i) + jac(YH,XH,j,i) * jac(YH,YH,j,i));
        metric(YH,YH,j,i) =  inv_detj * (jac(XH,XH,j,i) * jac(XH,XH,j,i) + jac(YH,XH,j,i) * jac(YH,XH,j,i));
      }
    }
  }
  //--------------------------------------------------------------------------
  void volume_metric(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::nodal_scalar_array& vol)
  {
    // Computes det(J) at nodes using the full isoparametric formulation

    ops.nodal_grad(coordinates, jac);

    for (unsigned j = 0; j < poly_order+1; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        vol(j,i) = jac(XH,YH, j,i) * jac(YH,XH, j,i) - jac(XH,XH,j,i) * jac(YH,YH,j,i);
      }
    }
  }
  //--------------------------------------------------------------------------
  void diffusion_metric_linear(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::scs_tensor_array& metric)
  {
    /*
     * Faster metric computation for geometrically linear elements
     */

    const double dx_x0 = coordinates(XH, poly_order, 0) - coordinates(XH, 0, 0);
    const double dx_x1 = coordinates(XH, 0, poly_order) - coordinates(XH, 0, 0);
    const double dx_y0 = coordinates(XH, poly_order, poly_order) - coordinates(XH, poly_order, 0);
    const double dx_y1 = coordinates(XH, poly_order, poly_order) - coordinates(XH, 0, poly_order);

    const double dy_x0 = coordinates(YH, poly_order, 0) - coordinates(YH, 0, 0);
    const double dy_x1 = coordinates(YH, 0, poly_order) - coordinates(YH, 0, 0);
    const double dy_y0 = coordinates(YH, poly_order, poly_order) - coordinates(YH, poly_order, 0);
    const double dy_y1 = coordinates(YH, poly_order, poly_order) - coordinates(YH, 0, poly_order);

    for (unsigned j = 0; j < poly_order; ++j) {
      const double dx_dyh = linear_scs_interp(0,j) * dx_x0 + linear_scs_interp(1,j) * dx_y1;
      const double dy_dyh = linear_scs_interp(0,j) * dy_x0 + linear_scs_interp(1,j) * dy_y1;

      const double orth = dx_dyh * dx_dyh + dy_dyh * dy_dyh;
      for (unsigned i = 0; i < poly_order+1; ++i) {
        const double dx_dxh = linear_nodal_interp(0,i) * dx_x1 + linear_nodal_interp(1,i) * dx_y0;
        const double dy_dxh = linear_nodal_interp(0,i) * dy_x1 + linear_nodal_interp(1,i) * dy_y0;

        const double inv_detj = 1.0 / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(XH,XH,j,i) =  inv_detj * orth;
        metric(XH,YH,j,i) = -inv_detj * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
      }
    }

    for (unsigned j = 0; j < poly_order; ++j) {
      const double dx_dxh =  linear_scs_interp(0,j) * dx_x1 + linear_scs_interp(1,j) * dx_y0;
      const double dy_dxh =  linear_scs_interp(0,j) * dy_x1 + linear_scs_interp(1,j) * dy_y0;

      const double orth = dx_dxh * dx_dxh + dy_dxh * dy_dxh;
      for (unsigned i = 0; i < poly_order+1; ++i) {
        const double dx_dyh = linear_nodal_interp(0,i) * dx_x0 + linear_nodal_interp(1,i) * dx_y1;
        const double dy_dyh = linear_nodal_interp(0,i) * dy_x0 + linear_nodal_interp(1,i) * dy_y1;

        const double inv_detj = 1.0 / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(YH,XH,j,i) = -inv_detj * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
        metric(YH,YH,j,i) =  inv_detj * orth;
      }
    }
  }
  //--------------------------------------------------------------------------
  void volume_metric_linear(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::nodal_scalar_array& vol)
  {
    // Computes det(J) at nodes using a linear basis for element geometry
    const double dx_x0 = coordinates(XH, poly_order, 0) - coordinates(XH, 0, 0);
    const double dx_x1 = coordinates(XH, 0, poly_order) - coordinates(XH, 0, 0);
    const double dx_y0 = coordinates(XH, poly_order, poly_order) - coordinates(XH, poly_order, 0);
    const double dx_y1 = coordinates(XH, poly_order, poly_order) - coordinates(XH, 0, poly_order);

    const double dy_x0 = coordinates(YH, poly_order, 0) - coordinates(YH, 0, 0);
    const double dy_x1 = coordinates(YH, 0, poly_order) - coordinates(YH, 0, 0);
    const double dy_y0 = coordinates(YH, poly_order, poly_order) - coordinates(YH, poly_order, 0);
    const double dy_y1 = coordinates(YH, poly_order, poly_order) - coordinates(YH, 0, poly_order);

    for (unsigned j = 0; j < poly_order+1; ++j) {
      const double dx_dyh = linear_nodal_interp(0,j) * dx_x1 + linear_nodal_interp(1,j) * dx_y0;
      const double dy_dyh = linear_nodal_interp(0,j) * dy_x1 + linear_nodal_interp(1,j) * dy_y0;

      for (unsigned i = 0; i < poly_order+1; ++i) {
        const double dx_dxh = linear_nodal_interp(0,i) * dx_x0 + linear_nodal_interp(1,i) * dx_y1;
        const double dy_dxh = linear_nodal_interp(0,i) * dy_x0 + linear_nodal_interp(1,i) * dy_y1;

        // times divided by 4 for missing factor of a half in the derivatives
        vol(j,i) = 0.25 * (dx_dyh * dy_dxh  - dx_dxh * dy_dyh);
      }
    }
  }

private:
  HighOrderOperatorsQuad<poly_order> ops; // has a mutable scratch
  mutable typename QuadViews<poly_order>::nodal_tensor_array jac; // larger than necessary for diffusion metric eval.

  const typename CoefficientViews<poly_order>::linear_nodal_matrix_array linear_nodal_interp;
  const typename CoefficientViews<poly_order>::linear_scs_matrix_array linear_scs_interp;

  enum Direction {
    XH = 0,
    YH = 1
  };

};


} // namespace naluUnit
} // namespace Sierra

#endif
