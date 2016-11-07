/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderLaplacianHex_h
#define HighOrderLaplacianHex_h

#include <element_promotion/new_assembly/HighOrderOperatorsHex.h>
#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <element_promotion/new_assembly/DirectionEnums.h>
#include <TopologyViews.h>


namespace sierra {
namespace naluUnit {
namespace TensorAssembly {
  //--------------------------------------------------------------------------
  template <unsigned nodes1D> int idx(int i, int j, int k)
  {
    return (nodes1D * (i * nodes1D + j) + k);
  };
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void add_elemental_dxx(
      const CoefficientMatrices<poly_order>& mat,
      const typename HexViews<poly_order>::scs_tensor_array& metric,
      typename HexViews<poly_order>::matrix_array& lhs,
      typename HexViews<poly_order>::nodal_scalar_array& scalar,
      typename HexViews<poly_order>::nodal_scalar_array& rhs)
  {
    using TopoView = HexViews<poly_order>;
    constexpr int n1D = TopoView::nodes1D;

    for (int n = 0; n < n1D; ++n) {
      for (int m = 0; m < n1D; ++m) {

        // x- element boundary
        constexpr int l_minus = 0;

        auto rowIndexMinus = idx<n1D>(n, m, l_minus);
        for (int k = 0; k < n1D; ++k) {
          const double w1 = mat.nodalWeights(n, k);
          for (int j = 0; j < n1D; ++j) {
            double orth = w1 * mat.nodalWeights(m, j) * metric(XH, XH, l_minus, k, j);

            for (int i = 0; i < n1D; ++i) {
              double matrix_coeff = orth * mat.scsDeriv(l_minus, i);
              lhs(rowIndexMinus, idx<n1D>(k, j, i)) += matrix_coeff;
            }
          }
        }

        // interior flux
        for (int l = 1; l < n1D - 1; ++l) {
          auto rowIndex = idx<n1D>(n, m, l);
          for (int k = 0; k < n1D; ++k) {
            const double w1 = mat.nodalWeights(n, k);
            for (int j = 0; j < n1D; ++j) {
              const double w = w1 * mat.nodalWeights(m, j);
              const double orthm1 = w * metric(XH, XH, l - 1, k, j);
              const double orthp0 = w * metric(XH, XH, l + 0, k, j);

              for (int i = 0; i < n1D; ++i) {
                double matrix_coeff = orthp0 * mat.scsDeriv(l + 0, i) - orthm1 * mat.scsDeriv(l - 1, i);
                lhs(rowIndex, idx<n1D>(k, j, i)) += matrix_coeff;
              }
            }
          }
        }

        // x+ element boundary
        constexpr int l_plus = n1D - 1;
        auto rowIndexPlus = idx<n1D>(n, m, l_plus);
        for (int k = 0; k < n1D; ++k) {
          const double w1 = mat.nodalWeights(n, k);
          for (int j = 0; j < n1D; ++j) {
            const double orth = w1 * mat.nodalWeights(m, j) * metric(XH, XH, l_plus - 1, k, j);

            for (int i = 0; i < n1D; ++i) {
              double matrix_coeff = -orth * mat.scsDeriv(l_plus - 1, i);
              lhs(rowIndexPlus, idx<n1D>(k, j, i)) += matrix_coeff;
            }
          }
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void add_elemental_dyy(
    const CoefficientMatrices<poly_order>& mat,
    const typename HexViews<poly_order>::scs_tensor_array& metric,
    typename HexViews<poly_order>::matrix_array& lhs,
    typename HexViews<poly_order>::nodal_scalar_array& scalar,
    typename HexViews<poly_order>::nodal_scalar_array& rhs)
  {
    using TopoView = HexViews<poly_order>;
    constexpr int n1D = TopoView::nodes1D;

    for (int n = 0; n < n1D; ++n) {
      for (int l = 0; l < n1D; ++l) {

        // x- element boundary
        constexpr int m_minus = 0;
        auto rowIndexMinus = idx<n1D>(n, m_minus, l);
        for (int k = 0; k < n1D; ++k) {
          const double w1 = mat.nodalWeights(n, k);
          for (int i = 0; i < n1D; ++i) {
            double orth = w1 * mat.nodalWeights(l, i) * metric(YH, YH, m_minus, k, i);

            for (int j = 0; j < n1D; ++j) {
              double matrix_coeff = orth * mat.scsDeriv(m_minus, j);
              lhs(rowIndexMinus, idx<n1D>(k, j, i)) += matrix_coeff;
            }
          }
        }

        // interior flux
        for (int m = 1; m < n1D - 1; ++m) {
          auto rowIndex = idx<n1D>(n, m, l);
          for (int k = 0; k < n1D; ++k) {
            const double w1 = mat.nodalWeights(n, k);
            for (int i = 0; i < n1D; ++i) {
              const double w = w1 * mat.nodalWeights(l, i);
              const double orthm1 = w * metric(YH, YH, m - 1, k, i);
              const double orthp0 = w * metric(YH, YH, m + 0, k, i);

              for (int j = 0; j < n1D; ++j) {
                double matrix_coeff = orthp0 * mat.scsDeriv(m + 0, j) - orthm1 * mat.scsDeriv(m - 1, j);
                lhs(rowIndex, idx<n1D>(k, j, i)) += matrix_coeff;
              }
            }
          }
        }

        // x+ element boundary
        constexpr int m_plus = n1D - 1;
        auto rowIndexPlus = idx<n1D>(n, m_plus, l);
        for (int k = 0; k < n1D; ++k) {
          const double w1 = mat.nodalWeights(n, k);
          for (int i = 0; i < n1D; ++i) {
            const double orth = w1 * mat.nodalWeights(l, i) * metric(YH, YH, m_plus - 1, k, i);

            for (int j = 0; j < n1D; ++j) {
              double matrix_coeff = -orth * mat.scsDeriv(m_plus - 1, j);
              lhs(rowIndexPlus, idx<n1D>(k, j, i)) += matrix_coeff;
            }
          }
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
   void add_elemental_dzz(
       const CoefficientMatrices<poly_order>& mat,
       const typename HexViews<poly_order>::scs_tensor_array& metric,
       typename HexViews<poly_order>::matrix_array& lhs,
       typename HexViews<poly_order>::nodal_scalar_array& scalar,
       typename HexViews<poly_order>::nodal_scalar_array& rhs)
  {
    using TopoView = HexViews<poly_order>;
    constexpr int n1D = TopoView::nodes1D;

    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {

        // x- element boundary
        constexpr int n_minus = 0;
        auto rowIndexMinus = idx<n1D>(n_minus, m, l);
        for (int j = 0; j < n1D; ++j) {
          const double w1 = mat.nodalWeights(m, j);
          for (int i = 0; i < n1D; ++i) {
            double orth = w1 * mat.nodalWeights(l, i) * metric(ZH, ZH, n_minus, j, i);

            for (int k = 0; k < n1D; ++k) {
              double matrix_coeff = orth * mat.scsDeriv(n_minus, k);
              lhs(rowIndexMinus, idx<n1D>(k, j, i)) += matrix_coeff;
            }
          }
        }

        // interior flux
        for (int n = 1; n < n1D - 1; ++n) {
          auto rowIndex = idx<n1D>(n, m, l);
          for (int j = 0; j < n1D; ++j) {
            const double w1 = mat.nodalWeights(m, j);
            for (int i = 0; i < n1D; ++i) {
              const double w = w1 * mat.nodalWeights(l, i);
              const double orthm1 = w * metric(ZH, ZH, n - 1, j, i);
              const double orthp0 = w * metric(ZH, ZH, n + 0, j, i);

              for (int k = 0; k < n1D; ++k) {
                double matrix_coeff = orthp0 * mat.scsDeriv(n + 0, k) - orthm1 * mat.scsDeriv(n - 1, k);
                lhs(rowIndex, idx<n1D>(k, j, i)) += matrix_coeff;
              }
            }
          }
        }

        // x+ element boundary
        constexpr int n_plus = n1D - 1;
        auto rowIndexPlus = idx<n1D>(n_plus, m, l);
        for (int j = 0; j < n1D; ++j) {
          const double w1 = mat.nodalWeights(m, j);
          for (int i = 0; i < n1D; ++i) {
            const double orth = w1 * mat.nodalWeights(l, i) * metric(ZH,ZH, n_plus - 1, j, i);

            for (int k = 0; k < n1D; ++k) {
              double matrix_coeff = -orth * mat.scsDeriv(n_plus - 1, k);
              lhs(rowIndexPlus, idx<n1D>(k, j, i)) += matrix_coeff;
            }
          }
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void add_elemental_laplacian_matrix(
    const CoefficientMatrices<poly_order>& mat,
    const typename HexViews<poly_order>::scs_tensor_array& metric,
    typename HexViews<poly_order>::matrix_array& lhs,
    typename HexViews<poly_order>::nodal_scalar_array& scalar,
    typename HexViews<poly_order>::nodal_scalar_array& rhs)
  {
    add_elemental_dxx(mat, metric, lhs, scalar, rhs);
    add_elemental_dyy(mat, metric, lhs, scalar, rhs);
    add_elemental_dzz(mat, metric, lhs, scalar, rhs);
  }




  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void add_volumetric_source(
    const CoefficientMatrices<poly_order>& mat,
    const typename HexViews<poly_order>::nodal_scalar_array& volume_metric,
    typename HexViews<poly_order>::nodal_scalar_array& nodal_source,
    typename HexViews<poly_order>::nodal_scalar_array& rhs)
  {
    using TopoView = HexViews<poly_order>;

    for (unsigned k = 0; k < TopoView::nodes1D; ++k) {
      for (unsigned j = 0; j < TopoView::nodes1D; ++j) {
        for (unsigned i = 0; i < TopoView::nodes1D; ++i) {
          nodal_source(k,j,i) *= volume_metric(k,j,i);
        }
      }
    }
    // computes the contribution of a volumetric source to the right-hand side
    HighOrderOperators::volume_3D<poly_order>(mat.nodalWeights, nodal_source, rhs);
  }

} // namespace TensorAssembly
} // namespace naluUnit
} // namespace Sierra

#endif
