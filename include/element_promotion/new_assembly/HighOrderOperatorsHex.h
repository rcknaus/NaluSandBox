/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperatorsHex_h
#define HighOrderOperatorsHex_h

#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <element_promotion/new_assembly/DirectionEnums.h>
#include <TopologyViews.h>

#include <Teuchos_BLAS.hpp>

namespace sierra {
namespace naluUnit {
namespace HighOrderOperators {

  namespace HexInternal {
    inline void mxm(
      Teuchos::ETransp transA,
      Teuchos::ETransp transB,
      int n,
      double alpha,
      const double* A,
      const double* B,
      double beta,
      double* C)
    {
      // matrix multiplication with the usual arguments fixed
      Teuchos::BLAS<int, double>().GEMM(
        transA, transB,
        n, n, n,
        alpha,
        A, n,
        B, n,
        beta,
        C, n
      );
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void apply_operator_x(const double* A, const double* x, double* y)
    {
      constexpr unsigned n = poly_order+1;
      constexpr unsigned n2 = n * n;
      for (unsigned j = 0; j < n; ++j) {
        mxm(Teuchos::TRANS, Teuchos::NO_TRANS, poly_order+1, 1.0, A, x + j * n2, 0.0, y + j * n2);
      }
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void apply_operator_y(const double* A, const double* x, double* y)
    {
      constexpr unsigned n = poly_order+1;
      constexpr unsigned n2 = n * n;
      for (unsigned j = 0; j < n; ++j) {
        mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, poly_order+1, 1.0, x + j * n2, A, 0.0, y + j * n2);
      }
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void apply_operator_z(const double* A, const double* x, double* y)
    {
      constexpr unsigned n = poly_order+1;
      constexpr unsigned n2 = n * n;
      Teuchos::BLAS<int, double>().GEMM(
        Teuchos::NO_TRANS,
        Teuchos::NO_TRANS,
        n2,
        n,
        n,
        1.0,
        x,
        n2,
        A,
        n,
        0.0,
        y,
        n2
      );
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dx_xhat(const double* scsDeriv, const double* in, double* out)
    {
      // computes xhat-derivative at scs of constant xhat coordinate
      apply_operator_x<poly_order>(scsDeriv, in, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dy_xhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at scs of constant xhat coordinate
      typename HexViews<poly_order>::nodal_scalar_array temp("");
      auto* p_temp = temp.ptr_on_device();

      apply_operator_x<poly_order>(scsInterp, in, p_temp);
      apply_operator_y<poly_order>(nodalDeriv, p_temp, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dz_xhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at scs of constant xhat coordinate

      typename HexViews<poly_order>::nodal_scalar_array temp("");
      auto* p_temp = temp.ptr_on_device();

      apply_operator_x<poly_order>(scsInterp, in, p_temp);
      apply_operator_z<poly_order>(nodalDeriv, p_temp, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dx_yhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      typename HexViews<poly_order>::nodal_scalar_array temp("");
      auto* p_temp = temp.ptr_on_device();

      apply_operator_y<poly_order>(scsInterp, in, p_temp);
      apply_operator_x<poly_order>(nodalDeriv, p_temp, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dy_yhat(const double* scsDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at scs of constant yhat coordinate
      apply_operator_y<poly_order>(scsDeriv, in, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dz_yhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      typename HexViews<poly_order>::nodal_scalar_array temp("");
      auto* p_temp = temp.ptr_on_device();

      apply_operator_y<poly_order>(scsInterp, in, p_temp);
      apply_operator_z<poly_order>(nodalDeriv, p_temp, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dx_zhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      typename HexViews<poly_order>::nodal_scalar_array temp("");
      auto* p_temp = temp.ptr_on_device();

      apply_operator_z<poly_order>(scsInterp, in, p_temp);
      apply_operator_x<poly_order>(nodalDeriv, p_temp, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dy_zhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      typename HexViews<poly_order>::nodal_scalar_array temp("");
      auto* p_temp = temp.ptr_on_device();

      apply_operator_z<poly_order>(scsInterp, in, p_temp);
      apply_operator_y<poly_order>(nodalDeriv, p_temp, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dz_zhat(const double* scsDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at scs of constant yhat coordinate
      apply_operator_z<poly_order>(scsDeriv, in, out);
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void nodal_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_scalar_array& f,
    const typename HexViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at nodes
    HexInternal::apply_operator_x<poly_order>(nodalDeriv.data(), &f(0,0,0), &grad(XH,0,0,0));
    HexInternal::apply_operator_y<poly_order>(nodalDeriv.data(), &f(0,0,0), &grad(YH,0,0,0));
    HexInternal::apply_operator_z<poly_order>(nodalDeriv.data(), &f(0,0,0), &grad(ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void nodal_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_vector_array& f,
    const typename HexViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at nodes
    for (unsigned d = 0; d < 3; ++d) {
      HexInternal::apply_operator_x<poly_order>(nodalDeriv.data(), &f(d,0,0,0), &grad(d,XH,0,0,0));
      HexInternal::apply_operator_y<poly_order>(nodalDeriv.data(), &f(d,0,0,0), &grad(d,YH,0,0,0));
      HexInternal::apply_operator_z<poly_order>(nodalDeriv.data(), &f(d,0,0,0), &grad(d,ZH,0,0,0));
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_xhat_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_scalar_array& f,
    const typename HexViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at nodes
    const auto* p_scsInterp = scsInterp.ptr_on_device();
    const auto* p_scsDeriv = scsDeriv.ptr_on_device();
    const auto* p_nodalDeriv = nodalDeriv.ptr_on_device();

    HexInternal::Dx_xhat<poly_order>(p_scsDeriv               , &f(0,0,0), &grad(XH,0,0,0));
    HexInternal::Dy_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(0,0,0), &grad(YH,0,0,0));
    HexInternal::Dz_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(0,0,0), &grad(ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_yhat_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_scalar_array& f,
    const typename HexViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at nodes
    const auto* p_scsInterp = scsInterp.ptr_on_device();
    const auto* p_scsDeriv = scsDeriv.ptr_on_device();
    const auto* p_nodalDeriv = nodalDeriv.ptr_on_device();

    HexInternal::Dx_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(0,0,0), &grad(XH,0,0,0));
    HexInternal::Dy_yhat<poly_order>(p_scsDeriv               , &f(0,0,0), &grad(YH,0,0,0));
    HexInternal::Dz_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(0,0,0), &grad(ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_zhat_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_scalar_array& f,
    const typename HexViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at nodes
    const auto* p_scsInterp = scsInterp.ptr_on_device();
    const auto* p_scsDeriv = scsDeriv.ptr_on_device();
    const auto* p_nodalDeriv = nodalDeriv.ptr_on_device();

    HexInternal::Dx_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(0,0,0), &grad(XH,0,0,0));
    HexInternal::Dy_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(0,0,0), &grad(YH,0,0,0));
    HexInternal::Dz_zhat<poly_order>(p_scsDeriv               , &f(0,0,0), &grad(ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_xhat_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_vector_array& f,
    const typename HexViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at nodes
    const auto* p_scsInterp = scsInterp.ptr_on_device();
    const auto* p_scsDeriv = scsDeriv.ptr_on_device();
    const auto* p_nodalDeriv = nodalDeriv.ptr_on_device();

    HexInternal::Dx_xhat<poly_order>(p_scsDeriv               , &f(XH, 0,0,0), &grad(XH,XH,0,0,0));
    HexInternal::Dy_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(XH, 0,0,0), &grad(XH,YH,0,0,0));
    HexInternal::Dz_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(XH, 0,0,0), &grad(XH,ZH,0,0,0));

    HexInternal::Dx_xhat<poly_order>(p_scsDeriv               , &f(YH, 0,0,0), &grad(YH,XH,0,0,0));
    HexInternal::Dy_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(YH, 0,0,0), &grad(YH,YH,0,0,0));
    HexInternal::Dz_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(YH, 0,0,0), &grad(YH,ZH,0,0,0));

    HexInternal::Dx_xhat<poly_order>(p_scsDeriv               , &f(ZH, 0,0,0), &grad(ZH,XH,0,0,0));
    HexInternal::Dy_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(ZH, 0,0,0), &grad(ZH,YH,0,0,0));
    HexInternal::Dz_xhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(ZH, 0,0,0), &grad(ZH,ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_yhat_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_vector_array& f,
    const typename HexViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at nodes
    const auto* p_scsInterp = scsInterp.ptr_on_device();
    const auto* p_scsDeriv = scsDeriv.ptr_on_device();
    const auto* p_nodalDeriv = nodalDeriv.ptr_on_device();

    HexInternal::Dx_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(XH, 0,0,0), &grad(XH,XH,0,0,0));
    HexInternal::Dy_yhat<poly_order>(p_scsDeriv               , &f(XH, 0,0,0), &grad(XH,YH,0,0,0));
    HexInternal::Dz_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(XH, 0,0,0), &grad(XH,ZH,0,0,0));

    HexInternal::Dx_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(YH, 0,0,0), &grad(YH,XH,0,0,0));
    HexInternal::Dy_yhat<poly_order>(p_scsDeriv               , &f(YH, 0,0,0), &grad(YH,YH,0,0,0));
    HexInternal::Dz_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(YH, 0,0,0), &grad(YH,ZH,0,0,0));

    HexInternal::Dx_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(ZH, 0,0,0), &grad(ZH,XH,0,0,0));
    HexInternal::Dy_yhat<poly_order>(p_scsDeriv               , &f(ZH, 0,0,0), &grad(ZH,YH,0,0,0));
    HexInternal::Dz_yhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(ZH, 0,0,0), &grad(ZH,ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_zhat_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename HexViews<poly_order>::nodal_vector_array& f,
    const typename HexViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at nodes
    const auto* p_scsInterp = scsInterp.ptr_on_device();
    const auto* p_scsDeriv = scsDeriv.ptr_on_device();
    const auto* p_nodalDeriv = nodalDeriv.ptr_on_device();

    HexInternal::Dx_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(XH, 0,0,0), &grad(XH,XH,0,0,0));
    HexInternal::Dy_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(XH, 0,0,0), &grad(XH,YH,0,0,0));
    HexInternal::Dz_zhat<poly_order>(p_scsDeriv               , &f(XH, 0,0,0), &grad(XH,ZH,0,0,0));

    HexInternal::Dx_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(YH, 0,0,0), &grad(YH,XH,0,0,0));
    HexInternal::Dy_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(YH, 0,0,0), &grad(YH,YH,0,0,0));
    HexInternal::Dz_zhat<poly_order>(p_scsDeriv               , &f(YH, 0,0,0), &grad(YH,ZH,0,0,0));

    HexInternal::Dx_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(ZH, 0,0,0), &grad(ZH,XH,0,0,0));
    HexInternal::Dy_zhat<poly_order>(p_scsInterp, p_nodalDeriv, &f(ZH, 0,0,0), &grad(ZH,YH,0,0,0));
    HexInternal::Dz_zhat<poly_order>(p_scsDeriv               , &f(ZH, 0,0,0), &grad(ZH,ZH,0,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order, CartesianDirection dir>
  void volume_2D(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalWeights,
    const typename HexViews<poly_order>::nodal_scalar_array& f,
    typename HexViews<poly_order>::nodal_scalar_array& fbar)
  {
    typename HexViews<poly_order>::nodal_scalar_array fbar0("temp0");

    const auto* p_nodalWeights = nodalWeights.ptr_on_device();
    auto* p_fbar0 = fbar0.ptr_on_device();
    auto* p_fbar = fbar.ptr_on_device();
    switch (dir)
    {
      case(XH):
      {
        HexInternal::apply_operator_y<poly_order>(p_nodalWeights, f      , p_fbar0);
        HexInternal::apply_operator_z<poly_order>(p_nodalWeights, p_fbar0, p_fbar );
        break;
      }
      case(YH):
      {
        HexInternal::apply_operator_x<poly_order>(p_nodalWeights, f      , p_fbar0);
        HexInternal::apply_operator_z<poly_order>(p_nodalWeights, p_fbar0, p_fbar );
        break;
      }
      case(ZH):
      {
        HexInternal::apply_operator_y<poly_order>(p_nodalWeights, f      , p_fbar0);
        HexInternal::apply_operator_x<poly_order>(p_nodalWeights, p_fbar0, p_fbar );
        break;
      }
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void volume_3D(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalWeights,
    const typename HexViews<poly_order>::nodal_scalar_array& f,
    typename HexViews<poly_order>::nodal_scalar_array& fbar2)
  {
    // computes volume integral along 3D volumes (e.g. "scv" in 3D)
    typename HexViews<poly_order>::nodal_scalar_array fbar0("temp0");

    HexInternal::apply_operator_x<poly_order>(nodalWeights.ptr_on_device(), f.ptr_on_device()    , fbar2.ptr_on_device());
    HexInternal::apply_operator_y<poly_order>(nodalWeights.ptr_on_device(), fbar2.ptr_on_device(), fbar0.ptr_on_device());
    HexInternal::apply_operator_z<poly_order>(nodalWeights.ptr_on_device(), fbar0.ptr_on_device(), fbar2.ptr_on_device());
  }
}
} // namespace naluUnit
} // namespace Sierra

#endif

