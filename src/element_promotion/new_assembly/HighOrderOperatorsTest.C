/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/new_assembly/HighOrderOperatorsTest.h>
#include <element_promotion/new_assembly/HighOrderOperatorsHex.h>
#include <element_promotion/new_assembly/HighOrderGeometryHex.h>
#include <element_promotion/new_assembly/DirectionEnums.h>

#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <element_promotion/ElementDescription.h>
#include <TopologyViews.h>

#include <NaluEnv.h>
#include <nalu_make_unique.h>
#include <TestHelper.h>

#include <Teuchos_BLAS.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <stddef.h>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

//--------------------------------------------------------------------------
double poly_val(const std::vector<double>& coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j] * std::pow(x, j);
  }
  return val;
}
//--------------------------------------------------------------------------
double poly_der(const std::vector<double>& coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 1; j < coeffs.size(); ++j) {
    val += coeffs[j] * std::pow(x, j - 1) * j;
  }
  return val;
}
//--------------------------------------------------------------------------
double poly_int(const std::vector<double>& coeffs, double xl, double xr)
{
  double upper = 0.0;
  double lower = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    upper += coeffs[j] * std::pow(xr, j + 1) / (j + 1.0);
    lower += coeffs[j] * std::pow(xl, j + 1) / (j + 1.0);
  }
  return (upper - lower);
}

namespace sierra{
namespace naluUnit{
namespace internal = HighOrderOperators::HexInternal;
  //--------------------------------------------------------------------------
  template <unsigned p> bool
  derivative_element_tests(unsigned numTrials, double tol)
  {
    // create a (-1,1) x (-1,1) element filled with polynomial values
    // and various quantities

    auto elem = ElementDescription::create(HexViews<p>::dim, p);

    // Tests an internal routine for computing spatial gradients
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    std::vector<double> coeffsX(elem->polyOrder+1);
    std::vector<double> coeffsY(elem->polyOrder+1);
    std::vector<double> coeffsZ(elem->polyOrder+1);

    typename HexViews<p>::nodal_tensor_array approxGrad("approxGrad");
    typename HexViews<p>::nodal_tensor_array exactGrad("exactGrad");
    typename HexViews<p>::nodal_vector_array nodalValues("nodalValues");
    const auto mat = CoefficientMatrices<p>();
    bool testPassed = false; // assume failure

    for (unsigned trial = 0; trial < numTrials; ++trial) {

      for (unsigned d = 0; d < HexViews<p>::dim; ++d) {
        for (unsigned k = 0; k < p + 1; ++k) {
          coeffsX[k] = coeff(rng);
          coeffsY[k] = coeff(rng);
          coeffsZ[k] = coeff(rng);
        }

        for (unsigned k = 0; k < p + 1; ++k) {
          double locz = elem->nodeLocs[k];
          for (unsigned j = 0; j < p + 1; ++j) {
            double locy = elem->nodeLocs[j];
            for (unsigned i = 0; i < p + 1; ++i) {
              double locx = elem->nodeLocs[i];

              nodalValues(d, k, j, i) =
                  poly_val(coeffsX, locx)
                * poly_val(coeffsY, locy)
                * poly_val(coeffsZ, locz);

              exactGrad(d, XH, k, j, i) =
                  poly_der(coeffsX, locx)
                * poly_val(coeffsY, locy)
                * poly_val(coeffsZ, locz);

              exactGrad(d, YH, k, j, i) =
                  poly_val(coeffsX, locx)
                * poly_der(coeffsY, locy)
                * poly_val(coeffsZ, locz);

              exactGrad(d, ZH, k, j, i) =
                  poly_val(coeffsX, locx)
                * poly_val(coeffsY, locy)
                * poly_der(coeffsZ, locz);
            }
          }
        }
      }


      HighOrderOperators::nodal_grad<p>(mat.nodalDeriv, nodalValues, approxGrad);
      bool gradTestPassed = is_near_output("gradient",
        &exactGrad(0,0,0,0,0), &approxGrad(0,0,0,0,0), tol, HexViews<p>::nodesPerElement *  HexViews<p>::dim);


      const auto* p_nodalDeriv = mat.nodalDeriv.ptr_on_device();
      for (unsigned d = 0; d < HexViews<p>::dim; ++d) {
        // internal routines
        internal::apply_operator_x<p>(p_nodalDeriv, &nodalValues(d,0,0,0), &approxGrad(d,XH,0,0,0) );
        bool dxTestPassed = is_near_output("x-derivative",
          &exactGrad(d,XH,0,0,0), &approxGrad(d,XH,0,0,0), tol, HexViews<p>::nodesPerElement);

        internal::apply_operator_y<p>(p_nodalDeriv, &nodalValues(d,0,0,0), &approxGrad(d,YH,0,0,0) );
        bool dyTestPassed = is_near_output("y-derivative",
          &exactGrad(d,YH,0,0,0), &approxGrad(d,YH,0,0,0), tol, HexViews<p>::nodesPerElement);

        internal::apply_operator_z<p>(p_nodalDeriv, &nodalValues(d,0,0,0), &approxGrad(d,ZH,0,0,0) );
        bool dzTestPassed = is_near_output("z-derivative",
          &exactGrad(d,ZH,0,0,0), &approxGrad(d,ZH,0,0,0), tol, HexViews<p>::nodesPerElement);

        testPassed  = dxTestPassed && dyTestPassed && dzTestPassed && gradTestPassed;

        if (!testPassed) {
          return false;
        }
      }
    }
    return testPassed;
  }
  //--------------------------------------------------------------------------
  template <unsigned p> bool
  integral_element_tests(unsigned numTrials, double tol)
  {
    auto elem = ElementDescription::create(3, p);
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    std::vector<double> coeffsX(elem->polyOrder+1);
    std::vector<double> coeffsY(elem->polyOrder+1);
    std::vector<double> coeffsZ(elem->polyOrder+1);

    typename HexViews<p>::nodal_scalar_array nodalValues("nodalValues");
    typename HexViews<p>::nodal_scalar_array exactVolIntegral("exactInt");
    typename HexViews<p>::nodal_scalar_array approxVolIntegral("approxInt");
    typename HexViews<p>::nodal_vector_array exactIntegrals("exactIntx");
    typename HexViews<p>::nodal_vector_array approxIntegrals("approxIntx");
    const auto mat = CoefficientMatrices<p>();
    const auto& scsEndLoc = elem->quadrature->scsEndLoc();

    bool testPassed = false; // assume failure

    for (unsigned trial = 0; trial < numTrials; ++trial) {
      for (unsigned k = 0; k < elem->polyOrder+1; ++k) {
        coeffsX[k] = coeff(rng);
        coeffsY[k] = coeff(rng);
        coeffsZ[k] = coeff(rng);
      }

      for (unsigned k = 0; k < p+1; ++k) {
        double zl = scsEndLoc[k + 0];
        double zr = scsEndLoc[k + 1];
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double yl = scsEndLoc[j + 0];
          double yr = scsEndLoc[j + 1];
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p + 1; ++i) {
            double xl = scsEndLoc[i + 0];
            double xr = scsEndLoc[i + 1];
            double locx = elem->nodeLocs[i];

            nodalValues(k,j,i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactIntegrals(0,k, j, i) =
                poly_int(coeffsX, xl, xr)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactIntegrals(1, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_int(coeffsY, yl, yr)
              * poly_val(coeffsZ, locz);

            exactIntegrals(2, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_int(coeffsZ, zl, zr);

            exactVolIntegral(k, j, i) =
                poly_int(coeffsX, xl, xr)
              * poly_int(coeffsY, yl, yr)
              * poly_int(coeffsZ, zl, zr);
          }
        }
      }

      HighOrderOperators::volume_3D<p>(mat.nodalWeights, nodalValues, approxVolIntegral);
      bool intTestPassed = is_near_output("volume integral",
        exactVolIntegral.data(), approxVolIntegral.data(), tol, HexViews<p>::nodesPerElement);

      // internal routines
      internal::apply_operator_x<p>(mat.nodalWeights.ptr_on_device(), nodalValues.ptr_on_device(), &approxIntegrals(0,0,0,0));
      bool ixTestPassed = is_near_output("x-integral",
        &exactIntegrals(0,0,0,0), &approxIntegrals(0,0,0,0), tol, HexViews<p>::nodesPerElement);

      internal::apply_operator_y<p>(mat.nodalWeights.ptr_on_device(), nodalValues.ptr_on_device(), &approxIntegrals(1,0,0,0));
      bool iyTestPassed = is_near_output("y-integral",
        &exactIntegrals(1,0,0,0), &approxIntegrals(1,0,0,0), tol, HexViews<p>::nodesPerElement);

      internal::apply_operator_z<p>(mat.nodalWeights.ptr_on_device(), nodalValues.ptr_on_device(), &approxIntegrals(2,0,0,0));
      bool izTestPassed = is_near_output("z-integral",
        &exactIntegrals(2,0,0,0), &approxIntegrals(2,0,0,0), tol, HexViews<p>::nodesPerElement);

      testPassed  = ixTestPassed && iyTestPassed && izTestPassed && intTestPassed;

      if (!testPassed) {
        return false;
      }
    }
    return testPassed;
  }
  //--------------------------------------------------------------------------
  template <unsigned p> bool
  scs_interp_element_tests(unsigned numTrials, double tol)
  {
    auto elem = ElementDescription::create(3, p);
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    std::vector<double> coeffsX(elem->polyOrder+1);
    std::vector<double> coeffsY(elem->polyOrder+1);
    std::vector<double> coeffsZ(elem->polyOrder+1);

    typename HexViews<p>::nodal_scalar_array nodalValues("nodalValues");
    typename HexViews<p>::nodal_vector_array exactScsInterp("exactScsInterp");
    typename HexViews<p>::nodal_vector_array approxScsInterp("approxScsInterp");
    const auto mat = CoefficientMatrices<p>();
    bool testPassed = false; // assume failure

    for (unsigned trial = 0; trial < numTrials; ++trial) {
      for (unsigned k = 0; k < elem->polyOrder+1; ++k) {
        coeffsX[k] = coeff(rng);
        coeffsY[k] = coeff(rng);
        coeffsZ[k] = coeff(rng);
      }

      for (unsigned k = 0; k < p +1; ++k) {
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p + 1; ++i) {
            double locx = elem->nodeLocs[i];

            nodalValues(k,j,i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);
          }
        }
      }

      for (unsigned k = 0; k < p + 1; ++k) {
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p; ++i) {
            double locx = elem->scsLoc[i];

            exactScsInterp(0, k,j,i) =
                poly_val(coeffsX, locx)
                * poly_val(coeffsY, locy)
                * poly_val(coeffsZ, locz);
          }
        }
      }

      for (unsigned k = 0; k < p + 1; ++k) {
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p; ++j) {
          double locy = elem->scsLoc[j];
          for (unsigned i = 0; i < p+1; ++i) {
            double locx = elem->nodeLocs[i];

            exactScsInterp(1, k,j,i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);
          }
        }
      }

      for (unsigned k = 0; k < p ; ++k) {
        double locz = elem->scsLoc[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p+1; ++i) {
            double locx = elem->nodeLocs[i];

            exactScsInterp(2, k,j,i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);
          }
        }
      }

      // internal routines
      internal::apply_operator_x<p>(mat.scsInterp.ptr_on_device(), nodalValues.ptr_on_device(), &approxScsInterp(0,0,0,0));
      internal::apply_operator_y<p>(mat.scsInterp.ptr_on_device(), nodalValues.ptr_on_device(), &approxScsInterp(1,0,0,0));
      internal::apply_operator_z<p>(mat.scsInterp.ptr_on_device(), nodalValues.ptr_on_device(), &approxScsInterp(2,0,0,0));

      bool ixTestPassed = is_near_output("x-interp",
        &exactScsInterp(0,0,0,0), &approxScsInterp(0,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool iyTestPassed = is_near_output("y-interp",
        &exactScsInterp(1,0,0,0), &approxScsInterp(1,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool izTestPassed = is_near_output("z-interp",
        &exactScsInterp(2,0,0,0), &approxScsInterp(2,0,0,0), tol, HexViews<p>::nodesPerElement);

      testPassed  = ixTestPassed && iyTestPassed && izTestPassed;

      if (!testPassed) {
        return false;
      }
    }
    return testPassed;
  }
  //--------------------------------------------------------------------------
  template <unsigned p> bool
  scs_deriv_element_tests(unsigned numTrials, double tol)
  {
    auto elem = ElementDescription::create(3, p);
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    std::vector<double> coeffsX(elem->polyOrder+1);
    std::vector<double> coeffsY(elem->polyOrder+1);
    std::vector<double> coeffsZ(elem->polyOrder+1);

    typename HexViews<p>::nodal_scalar_array nodalValues("nodalValues");
    typename HexViews<p>::nodal_tensor_array exactScsDeriv("exactScsInterp");
    typename HexViews<p>::nodal_tensor_array approxScsDeriv("approxScsInterp");
    const auto mat = CoefficientMatrices<p>();
    bool testPassed = false; // assume failure

    for (unsigned trial = 0; trial < numTrials; ++trial) {
      for (unsigned k = 0; k < elem->polyOrder+1; ++k) {
        coeffsX[k] = coeff(rng);
        coeffsY[k] = coeff(rng);
        coeffsZ[k] = coeff(rng);
      }

      for (unsigned k = 0; k < p + 1; ++k) {
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p + 1; ++i) {
            double locx = elem->nodeLocs[i];

            nodalValues(k, j, i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);
          }
        }
      }

      for (unsigned k = 0; k < p + 1; ++k) {
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p; ++i) {
            double locx = elem->scsLoc[i];

            exactScsDeriv(XH,XH, k, j, i) =
                poly_der(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactScsDeriv(XH,YH, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_der(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactScsDeriv(XH,ZH, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_der(coeffsZ, locz);
          }
        }
      }

      for (unsigned k = 0; k < p + 1; ++k) {
        double locz = elem->nodeLocs[k];
        for (unsigned j = 0; j < p; ++j) {
          double locy = elem->scsLoc[j];
          for (unsigned i = 0; i < p + 1; ++i) {
            double locx = elem->nodeLocs[i];

            exactScsDeriv(YH,XH, k, j, i) =
                poly_der(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactScsDeriv(YH,YH, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_der(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactScsDeriv(YH,ZH, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_der(coeffsZ, locz);
          }
        }
      }

      for (unsigned k = 0; k < p; ++k) {
        double locz = elem->scsLoc[k];
        for (unsigned j = 0; j < p + 1; ++j) {
          double locy = elem->nodeLocs[j];
          for (unsigned i = 0; i < p + 1; ++i) {
            double locx = elem->nodeLocs[i];

            exactScsDeriv(ZH,XH, k, j, i) =
                poly_der(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactScsDeriv(ZH,YH, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_der(coeffsY, locy)
              * poly_val(coeffsZ, locz);

            exactScsDeriv(ZH,ZH, k, j, i) =
                poly_val(coeffsX, locx)
              * poly_val(coeffsY, locy)
              * poly_der(coeffsZ, locz);
          }
        }
      }

      // internal routines
      internal::Dx_xhat<p>(mat.scsDeriv.ptr_on_device(), nodalValues.ptr_on_device(), &approxScsDeriv(XH,XH,0,0,0));
      internal::Dy_yhat<p>(mat.scsDeriv.ptr_on_device(), nodalValues.ptr_on_device(), &approxScsDeriv(YH,YH,0,0,0));
      internal::Dz_zhat<p>(mat.scsDeriv.ptr_on_device(), nodalValues.ptr_on_device(), &approxScsDeriv(ZH,ZH,0,0,0));

      internal::Dy_xhat<p>(
        mat.scsInterp.ptr_on_device(),
        mat.nodalDeriv.ptr_on_device(),
        nodalValues.ptr_on_device(),
        &approxScsDeriv(XH, YH, 0, 0, 0)
      );

      internal::Dz_xhat<p>(
        mat.scsInterp.ptr_on_device(),
        mat.nodalDeriv.ptr_on_device(),
        nodalValues.ptr_on_device(),
        &approxScsDeriv(XH, ZH, 0, 0, 0)
      );

      internal::Dx_yhat<p>(
        mat.scsInterp.ptr_on_device(),
        mat.nodalDeriv.ptr_on_device(),
        nodalValues.ptr_on_device(),
        &approxScsDeriv(YH, XH, 0, 0, 0)
      );

      internal::Dz_yhat<p>(
        mat.scsInterp.ptr_on_device(),
        mat.nodalDeriv.ptr_on_device(),
        nodalValues.ptr_on_device(),
        &approxScsDeriv(YH, ZH, 0, 0, 0)
      );

      internal::Dx_zhat<p>(
        mat.scsInterp.ptr_on_device(),
        mat.nodalDeriv.ptr_on_device(),
        nodalValues.ptr_on_device(),
        &approxScsDeriv(ZH, XH, 0, 0, 0)
      );

      internal::Dy_zhat<p>(
        mat.scsInterp.ptr_on_device(),
        mat.nodalDeriv.ptr_on_device(),
        nodalValues.ptr_on_device(),
        &approxScsDeriv(ZH, YH, 0, 0, 0)
      );

      bool dxxTestPassed = is_near_output("dx_xhat",
        &exactScsDeriv(XH,XH,0,0,0), &approxScsDeriv(XH,XH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dyyTestPassed = is_near_output("dy_yhat",
        &exactScsDeriv(YH,YH,0,0,0), &approxScsDeriv(YH,YH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dzzTestPassed = is_near_output("dz_zhat",
        &exactScsDeriv(ZH,ZH,0,0,0), &approxScsDeriv(ZH,ZH,0,0,0), tol, HexViews<p>::nodesPerElement);

      // cross terms
      bool dxyTestPassed = is_near_output("dy_xhat",
        &exactScsDeriv(XH,YH,0,0,0), &approxScsDeriv(XH,YH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dxzTestPassed = is_near_output("dy_zhat",
        &exactScsDeriv(XH,ZH,0,0,0), &approxScsDeriv(XH,ZH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dyxTestPassed = is_near_output("dx_yhat",
        &exactScsDeriv(YH,XH,0,0,0), &approxScsDeriv(YH,XH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dyzTestPassed = is_near_output("dz_yhat",
        &exactScsDeriv(YH,ZH,0,0,0), &approxScsDeriv(YH,ZH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dzxTestPassed = is_near_output("dx_zhat",
        &exactScsDeriv(ZH,XH,0,0,0), &approxScsDeriv(ZH,XH,0,0,0), tol, HexViews<p>::nodesPerElement);

      bool dzyTestPassed = is_near_output("dy_zhat",
        &exactScsDeriv(ZH,YH,0,0,0), &approxScsDeriv(ZH,YH,0,0,0), tol, HexViews<p>::nodesPerElement);

      testPassed  =
          dxxTestPassed && dyyTestPassed && dzzTestPassed &&
          dxyTestPassed && dxzTestPassed &&
          dyxTestPassed && dyzTestPassed &&
          dzxTestPassed && dzyTestPassed;

      if (!testPassed) {
        return false;
      }
    }
    return testPassed;
  }

}
}
//--------------------------------------------------------------------------
namespace sierra{
namespace naluUnit{
void
HighOrderOperatorsHexTest::execute()
{
  constexpr unsigned polyOrder = 10;

  NaluEnv::self().naluOutputP0() << "High Order Operator Unit Tests for order '" << polyOrder << "'"<< std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;

  unsigned numTrials = 1; // number of random polynomials for each test
  double tol = 1.0e-10;    // floating point tolerance (polynomial coeffs ~ order 1)

  output_result("Reference element derivatives     ", derivative_element_tests<polyOrder>(numTrials, tol));
  output_result("Reference element integrals       ", integral_element_tests<polyOrder>(numTrials, tol));
  output_result("Reference element scs-interp terms", scs_interp_element_tests<polyOrder>(numTrials, tol));
  output_result("Reference element scs-deriv terms ", scs_deriv_element_tests<polyOrder>(numTrials, tol));
  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;
}
} // namespace naluUnit
}  // namespace sierra
