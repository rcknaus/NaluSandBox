/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderLaplacianQuad_h
#define HighOrderLaplacianQuad_h

#include <element_promotion/new_assembly/HighOrderOperatorsQuad.h>
#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <TopologyViews.h>

namespace sierra {
namespace naluUnit {
namespace TensorAssembly {

  template <unsigned nodes1D>
  int idx(int i, int j) { return i*nodes1D+j; };

  enum Direction {
    XH = 0,
    YH = 1
  };

  template <unsigned poly_order>
  void add_elemental_laplacian_matrix(
    const CoefficientMatrices<poly_order>& mat,
    const typename QuadViews<poly_order>::scs_tensor_array& metric,
    typename QuadViews<poly_order>::matrix_array& lhs)
  {
    /*
     * Computes the elemental lhs for the Laplacian operator given
     * the correct grid metrics, split into boundary and interior terms.
     */
    using TopoView = QuadViews<poly_order>;

    static_assert (TopoView::dim == 2,"Only 2D implemented");

    // flux past constant yhat lines
    for (int n = 0; n < TopoView::nodes1D; ++n) {
      // x- element boundary
      constexpr int m_minus = 0;
      for (int j = 0; j < TopoView::nodes1D; ++j) {
        double orth = mat.nodalWeights(n, j) * metric(XH, XH, m_minus, j);
        double non_orth = 0.0;
        for (int k = 0; k < TopoView::nodes1D; ++k) {
          non_orth += mat.nodalWeights(n, k) * mat.nodalDeriv(k, j) * metric(XH, YH, m_minus, k);
        }

        for (int i = 0; i < TopoView::nodes1D; ++i) {
          lhs(idx<TopoView::nodes1D>(n, m_minus), idx<TopoView::nodes1D>(j, i)) +=
              orth * mat.scsDeriv(m_minus, i) + non_orth * mat.scsInterp(m_minus, i);
        }
      }

      // interior flux
      for (int m = 1; m < TopoView::nodes1D - 1; ++m) {
        for (int j = 0; j < TopoView::nodes1D; ++j) {
          const double w = mat.nodalWeights(n, j);
          const double orthm1 = w * metric(XH, XH, m - 1, j);
          const double orthp0 = w * metric(XH, XH, m + 0, j);

          double non_orthp0 = 0.0;
          double non_orthm1 = 0.0;
          for (int k = 0; k < TopoView::nodes1D; ++k) {
            const double wd = mat.nodalWeights(n, k) * mat.nodalDeriv(k, j);
            non_orthm1 += wd * metric(XH, YH, m - 1, k);
            non_orthp0 += wd * metric(XH, YH, m + 0, k);
          }

          for (int i = 0; i < TopoView::nodes1D; ++i) {
            const double fm = orthm1 * mat.scsDeriv(m - 1, i) + non_orthm1 * mat.scsInterp(m - 1, i);
            const double fp = orthp0 * mat.scsDeriv(m + 0, i) + non_orthp0 * mat.scsInterp(m + 0, i);
            lhs(idx<TopoView::nodes1D>(n, m), idx<TopoView::nodes1D>(j, i)) += (fp - fm);
          }
        }
      }

      // x+ element boundary
      constexpr int m_plus = TopoView::nodes1D - 1;
      for (int j = 0; j < TopoView::nodes1D; ++j) {
        const double orth = mat.nodalWeights(n, j) * metric(XH, XH, m_plus - 1, j);

        double non_orth = 0.0;
        for (int k = 0; k < TopoView::nodes1D; ++k) {
          non_orth += mat.nodalWeights(n, k) * mat.nodalDeriv(k, j) * metric(XH, YH, m_plus - 1, k);
        }
        for (int i = 0; i < TopoView::nodes1D; ++i) {
          lhs(idx<TopoView::nodes1D>(n, m_plus), idx<TopoView::nodes1D>(j, i)) -=
              orth * mat.scsDeriv(m_plus - 1, i) + non_orth * mat.scsInterp(m_plus - 1, i);
        }
      }
    }

    // flux past constant xhat lines
    for (int m = 0; m < TopoView::nodes1D; ++m) {
      // y+ boundary
      constexpr int n_minus = 0;
      for (int i = 0; i < TopoView::nodes1D; ++i) {
        const double orth = mat.nodalWeights(m, i) * metric(YH, YH, n_minus, i);

        double non_orth = 0.0;
        for (int k = 0; k < TopoView::nodes1D; ++k) {
          non_orth += mat.nodalWeights(m, k) * mat.nodalDeriv(k, i) * metric(YH, XH, n_minus, k);
        }
        for (int j = 0; j < TopoView::nodes1D; ++j) {
          lhs(idx<TopoView::nodes1D>(n_minus, m), idx<TopoView::nodes1D>(j, i)) +=
              orth * mat.scsDeriv(n_minus, j) + non_orth * mat.scsInterp(n_minus, j);
        }
      }

      // interior flux
      for (int n = 1; n < TopoView::nodes1D - 1; ++n) {
        for (int i = 0; i < TopoView::nodes1D; ++i) {
          const double w = mat.nodalWeights(m, i);
          const double orthm1 = w * metric(YH, YH, n - 1, i);
          const double orthp0 = w * metric(YH, YH, n + 0, i);

          double non_orthp0 = 0.0;
          double non_orthm1 = 0.0;
          for (int k = 0; k < TopoView::nodes1D; ++k) {
            const double wd = mat.nodalWeights(m, k) * mat.nodalDeriv(k, i);
            non_orthm1 += wd * metric(YH, XH, n - 1, k);
            non_orthp0 += wd * metric(YH, XH, n + 0, k);
          }

          for (int j = 0; j < TopoView::nodes1D; ++j) {
            const double fm = orthm1 * mat.scsDeriv(n - 1, j) + non_orthm1 * mat.scsInterp(n - 1, j);
            const double fp = orthp0 * mat.scsDeriv(n + 0, j) + non_orthp0 * mat.scsInterp(n + 0, j);
            lhs(idx<TopoView::nodes1D>(n, m), idx<TopoView::nodes1D>(j, i)) += (fp - fm);
          }
        }
      }

      // y+ boundary
      constexpr int n_plus = TopoView::nodes1D - 1;
      for (int i = 0; i < TopoView::nodes1D; ++i) {
        const double orth = mat.nodalWeights(m, i) * metric(YH, YH, n_plus - 1, i);

        double non_orth = 0.0;
        for (int k = 0; k < TopoView::nodes1D; ++k) {
          non_orth += mat.nodalWeights(m, k) * mat.nodalDeriv(k, i) * metric(YH, XH, n_plus - 1, k);
        }
        for (int j = 0; j < TopoView::nodes1D; ++j) {
          lhs(idx<TopoView::nodes1D>(n_plus, m), idx<TopoView::nodes1D>(j, i)) -=
              orth * mat.scsDeriv(n_plus - 1, j) + non_orth * mat.scsInterp(n_plus - 1, j);
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void add_elemental_laplacian_action(
    const CoefficientMatrices<poly_order>& mat,
    const typename QuadViews<poly_order>::scs_tensor_array& metric,
    const typename QuadViews<poly_order>::nodal_scalar_array& scalar,
    typename QuadViews<poly_order>::nodal_scalar_array& residual
  )
  {
    /*
     * Compute the action of the LHS on a scalar field as a sequence of small (N x N), dense matrix-matrix
     * multiplications instead of a large (N^2 x N^2) matvec
     */
    using TopoView = QuadViews<poly_order>;

    // gradient at constant xhat surfaces
    typename TopoView::nodal_vector_array grad_phi("gp");
    HighOrderOperators::scs_xhat_grad<poly_order>(mat.scsInterp, mat.scsDeriv, mat.nodalDeriv, scalar, grad_phi);

    // apply metric transformation
    typename TopoView::nodal_scalar_array integrand("");
    for (unsigned j = 0; j < TopoView::nodes1D - 1; ++j) {
      for (unsigned i = 0; i < TopoView::nodes1D; ++i) {
        integrand(j, i) = metric(XH,XH, j, i) * grad_phi(XH, j, i) + metric(XH,YH, j, i) * grad_phi(YH, j, i);
      }
    }

    // integration / scattering of surface fluxes
    typename TopoView::nodal_scalar_array flux("");
    HighOrderOperators::volume_1D<poly_order>(mat.nodalWeights, integrand, flux);
    HighOrderOperators::scatter_flux_xhat<poly_order>(flux, residual);

    // gradient at constant yhat surfaces
    HighOrderOperators::scs_yhat_grad<poly_order>(mat.scsInterp, mat.scsDeriv, mat.nodalDeriv, scalar, grad_phi);

    // apply metric transformation
    for (unsigned j = 0; j < TopoView::nodes1D - 1; ++j) {
      for (unsigned i = 0; i < TopoView::nodes1D; ++i) {
        integrand(j, i) = metric(YH,XH, j, i) * grad_phi(XH, j, i) + metric(YH,YH, j, i) * grad_phi(YH, j, i);
      }
    }

    // integration / scattering of surface fluxes
    HighOrderOperators::volume_1D<poly_order>(mat.nodalWeights, integrand, flux);
    HighOrderOperators::scatter_flux_yhat<poly_order>(flux, residual);
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void add_volumetric_source(
    const CoefficientMatrices<poly_order>& mat,
    const typename QuadViews<poly_order>::nodal_scalar_array& volume_metric,
    const typename QuadViews<poly_order>::nodal_scalar_array& nodal_source,
    typename QuadViews<poly_order>::nodal_scalar_array& rhs)
  {
    using TopoView = QuadViews<poly_order>;

    static_assert (TopoView::dim == 2,"Only 2D implemented");

    for (unsigned j = 0; j < TopoView::nodes1D; ++j) {
      for (unsigned i = 0; i < TopoView::nodes1D; ++i) {
        nodal_source(j,i) *= volume_metric(j,i);
      }
    }

    // computes the contribution of a volumetric source to the right-hand side
    HighOrderOperators::volume_2D<poly_order>(mat.nodalWeights, nodal_source, rhs);
  }

} // namespace HighOrderLaplacianQuad
} // namespace naluUnit
} // namespace Sierra

#endif
