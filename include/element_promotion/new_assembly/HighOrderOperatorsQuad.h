/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperators_h
#define HighOrderOperators_h

#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <TopologyViews.h>
#include <Teuchos_BLAS.hpp>

namespace sierra {
namespace naluUnit {
namespace HighOrderOperators {
  namespace QuadInternal {
    void mxm(
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
      Teuchos::BLAS<int, double>().GEMM(transA, transB,
        n, n, n,
        alpha, A, n, B, n,
        beta, C, n);
    }

    template <unsigned poly_order>
    void Dx( const double* nodalDeriv, const double* in, double* out)
    {
      // computes xhat-derivative at nodes
      mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, poly_order+1, 1.0, in, nodalDeriv, 0.0, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dy(const double* nodalDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at nodes
      mxm(Teuchos::TRANS   , Teuchos::NO_TRANS,poly_order+1, 1.0, nodalDeriv, in, 0.0, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dx_xhat(const double* scsDeriv, const double* in, double* out)
    {
      // computes xhat-derivative at scs of constant xhat coordinate
      mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, poly_order+1, 1.0, in, scsDeriv,  0.0, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dy_xhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at scs of constant xhat coordinate
      typename QuadViews<poly_order>::nodal_scalar_array temp("temp");
      mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, poly_order+1, 1.0, in, scsInterp, 0.0, temp.data());
      mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, poly_order+1, 1.0, nodalDeriv, temp.data(), 0.0, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dx_yhat(const double* scsInterp, const double* nodalDeriv, const double* in, double* out)
    {
      // computes xhat-derivative at scs of constant yhat coordinate
      typename QuadViews<poly_order>::nodal_scalar_array temp("temp");
      mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, poly_order+1, 1.0, in, scsInterp, 0.0, temp.data());
      mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, poly_order+1, 1.0, nodalDeriv, temp.data(), 0.0, out);
    }
    //--------------------------------------------------------------------------
    template <unsigned poly_order>
    void Dy_yhat(const double* scsDeriv, const double* in, double* out)
    {
      // computes yhat-derivative at scs of constant yhat coordinate
      mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, poly_order+1, 1.0, in, scsDeriv,  0.0, out);
    }
  }

  enum Direction
  {
    XH = 0,
    YH = 1
  };

  template <unsigned poly_order>
  void nodal_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    const typename QuadViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at nodes
    QuadInternal::Dx<poly_order>(nodalDeriv.data(), &f(0,0), &grad(XH,0,0));
    QuadInternal::Dy<poly_order>(nodalDeriv.data(), &f(0,0), &grad(YH,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void nodal_grad(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename QuadViews<poly_order>::nodal_vector_array& f,
    typename QuadViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at nodes
    QuadInternal::Dx<poly_order>(nodalDeriv.data(), &f(XH, 0,0), &grad(XH,XH,0,0));
    QuadInternal::Dy<poly_order>(nodalDeriv.data(), &f(XH, 0,0), &grad(XH,YH,0,0));
    QuadInternal::Dx<poly_order>(nodalDeriv.data(), &f(YH, 0,0), &grad(YH,XH,0,0));
    QuadInternal::Dy<poly_order>(nodalDeriv.data(), &f(YH, 0,0), &grad(YH,YH,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_yhat_grad(
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element at scs of constant yhat coordinate
    QuadInternal::Dx_yhat<poly_order>(scsInterp.data(), nodalDeriv.data(), &f(0,0), &grad(XH,0,0));
    QuadInternal::Dy_yhat<poly_order>(scsDeriv.data(), &f(0,0), &grad(YH,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_yhat_grad(
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename QuadViews<poly_order>::nodal_vector_array& f,
    typename QuadViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at scs of constant yhat coordinate
    QuadInternal::Dx_yhat<poly_order>(scsInterp.data(), nodalDeriv.data(), &f(XH,0,0), &grad(XH,XH,0,0));
    QuadInternal::Dy_yhat<poly_order>(scsDeriv.data(), &f(XH,0,0), &grad(XH,YH,0,0));

    QuadInternal::Dx_yhat<poly_order>(scsInterp.data(), nodalDeriv.data(),&f(YH,0,0), &grad(YH,XH,0,0));
    QuadInternal::Dy_yhat<poly_order>(scsDeriv.data(), &f(YH,0,0), &grad(YH,YH,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_xhat_grad(
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at scs of constant xhat coordinate
    QuadInternal::Dx_xhat<poly_order>(scsDeriv.data(), &f(0,0), &grad(XH,0,0));
    QuadInternal::Dy_xhat<poly_order>(scsInterp.data(), nodalDeriv.data(), &f(0,0), &grad(YH,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scs_xhat_grad(
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsInterp,
    const typename CoefficientViews<poly_order>::scs_matrix_array& scsDeriv,
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalDeriv,
    const typename QuadViews<poly_order>::nodal_vector_array& f,
    typename QuadViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at scs of constant xhat coordinate
    QuadInternal::Dx_xhat<poly_order>(scsDeriv.data(),&f(XH,0,0), &grad(XH,XH,0,0));
    QuadInternal::Dy_xhat<poly_order>(scsInterp.data(), nodalDeriv.data(),&f(XH,0,0), &grad(XH,YH,0,0));
    QuadInternal::Dx_xhat<poly_order>(scsDeriv.data(),&f(YH,0,0), &grad(YH,XH,0,0));
    QuadInternal::Dy_xhat<poly_order>(scsInterp.data(), nodalDeriv.data(),&f(YH,0,0), &grad(YH,YH,0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void volume_1D(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalWeights,
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_scalar_array& f_bar)
  {
    // computes volume integral along 1D lines (e.g. "scs" in 2D)
    QuadInternal::mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, poly_order +1, 1.0, nodalWeights.data(), &f(0,0), 0.0, &f_bar(0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void volume_2D(
    const typename CoefficientViews<poly_order>::nodal_matrix_array& nodalWeights,
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_scalar_array& f_bar)
  {
    // computes volume integral along 2D volumes (e.g. "scv" in 2D)
    typename QuadViews<poly_order>::nodal_scalar_array temp("temp");
    QuadInternal::mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, poly_order +1, 1.0,  &f(0,0), nodalWeights.data(),    0.0, temp.data());
    QuadInternal::mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, poly_order +1, 1.0, nodalWeights.data(), temp.data(), 1.0, &f_bar(0,0));
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scatter_flux_xhat(
    const typename QuadViews<poly_order>::nodal_scalar_array& flux,
    typename QuadViews<poly_order>::nodal_scalar_array& residual)
  {
    for (unsigned n = 0; n < poly_order+1; ++n) {
      residual(n,0) -= flux(0,n);
      for (unsigned p = 1; p < poly_order; ++p) {
        residual(n,p) -= flux(p,n) - flux(p-1,n);
      }
      residual(n,poly_order) += flux(poly_order-1,n);
    }
  }
  //--------------------------------------------------------------------------
  template <unsigned poly_order>
  void scatter_flux_yhat(
    const typename QuadViews<poly_order>::nodal_scalar_array& flux,
    typename QuadViews<poly_order>::nodal_scalar_array& residual)
  {
    // Scattering of the fluxes to nodes
    for (unsigned m = 0; m < poly_order+1; ++m) {
      residual(0, m) -= flux(0, m);
      for (unsigned p = 1; p < poly_order; ++p) {
        residual(p, m) -= flux(p, m) - flux(p - 1, m);
      }
      residual(poly_order, m) += flux(poly_order-1, m);
    }
  }

}
} // namespace naluUnit
} // namespace Sierra

#endif

