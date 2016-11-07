/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TestHelper_h
#define TestHelper_h

#include <NaluEnv.h>

#include <stk_util/environment/ReportHandler.hpp>

#include <stddef.h>
#include <cmath>
#include <ostream>
#include <string>
#include <vector>
#include <limits>

namespace sierra {
namespace naluUnit {

  inline bool
  is_near(double approx, double exact, double tol)
  {
    return (std::abs(approx-exact) < tol);
  }

  template<typename Container> double
  max_error(const Container& approx, const Container& exact)
  {
    if (approx.size() != exact.size() || approx.empty()) {
      return std::numeric_limits<double>::max();
    }

    double err = -1.0;
    for (unsigned j = 0; j < approx.size(); ++j) {
      if (!std::isfinite(approx[j])) return std::numeric_limits<double>::max();
      err = std::max(err, std::abs(approx[j]-exact[j]));
    }
    return err;
  }

  inline double
  max_error(const double* approx, const double* exact, int N)
  {
    double err = -1.0;
    for (int j = 0; j < N; ++j) {
      if (!std::isfinite(approx[j])) return std::numeric_limits<double>::max();
      err = std::max(err, std::abs(approx[j]-exact[j]));
    }
    return err;
  }

  template<typename Container> bool
  is_near(
    const Container& approx, const Container& exact, double tol)
  {
    if (max_error(approx,exact) < tol) {
      return true;
    }
    return false;
  }

  inline bool is_near(const double* approx, const double* exact, double tol, int N)
  {
    if (max_error(approx, exact, N) < tol) {
      return true;
    }
    return false;
  }

  inline void
  output_result(std::string test_name, bool status)
  {
    if (status) {
      NaluEnv::self().naluOutputP0() << test_name << " TEST: PASSED " << std::endl;
    }
    else {
      NaluEnv::self().naluOutputP0() << test_name << " TEST: FAILED " << std::endl;
    }
  }

  //--------------------------------------------------------------------------
  inline bool
  is_near_output(std::string testName, const double* exact, const double* approx, double tol, int length)
  {
    if (is_near(exact, approx, tol, length)) {
      return true;
    }
    else {
      NaluEnv::self().naluOutputP0() << testName + " Test failed with max error: "
                                     << max_error(exact, approx, length)
                                     << std::endl;
      return false;
    }
  }


} // namespace naluUnit
} // namespace Sierra

#endif
