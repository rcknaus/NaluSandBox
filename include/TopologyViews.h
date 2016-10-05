/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TopologyViews_h
#define TopologyViews_h

#include <Kokkos_Core.hpp>

namespace stk {namespace mesh {struct Entity; }}

namespace sierra {
namespace naluUnit {

// commonly used (statically-sized) arrays for different element topologies
template <unsigned p>
struct CoefficientViews
{
  constexpr static int poly_order = p;
  constexpr static int nodes1D = p + 1;
  constexpr static int nodesPerElement = nodes1D;
  using real_type = double;

  // scs matrices are padded to be square at the moment, so nodal/scs matrices are the same type
  using scs_matrix_array = Kokkos::View<real_type[nodes1D][nodes1D]>;
  using nodal_matrix_array = Kokkos::View<real_type[nodes1D][nodes1D]>;
  using linear_nodal_matrix_array = Kokkos::View<real_type[2][p+1]>;
  using linear_scs_matrix_array = Kokkos::View<real_type[2][p]>;
};


template <unsigned p>
struct LineViews  {
  constexpr static int poly_order = p;
  constexpr static int nodes1D = p + 1;
  constexpr static int dim = 2;
  using real_type = double;
  using connectivity_type = int;

  using connectivity_array = Kokkos::View<connectivity_type[nodes1D]>;
  using nodal_scalar_array = Kokkos::View<real_type[nodes1D]>;

};

template <unsigned p>
struct QuadViews
{
  constexpr static int poly_order = p;
  constexpr static int nodes1D = p + 1;
  constexpr static int dim = 2;
  constexpr static int nodesPerElement = nodes1D * nodes1D;
  using real_type = double;
  using connectivity_type = int;

  using matrix_array = Kokkos::View<real_type[nodesPerElement][nodesPerElement]>;

  // arrays for nodal variables
  using connectivity_array = Kokkos::View<connectivity_type[nodes1D][nodes1D]>;
  using nodal_scalar_array = Kokkos::View<real_type[nodes1D][nodes1D]>;
  using nodal_vector_array = Kokkos::View<real_type[dim][nodes1D][nodes1D]>;
  using nodal_tensor_array = Kokkos::View<real_type[dim][dim][nodes1D][nodes1D]>;

  // arrays for variables evaluated at subcontrol surfaces in one  direction, e.g.
  // at a constant xhat line
  using scs_array = Kokkos::View<real_type[p][nodes1D]>;
  using scs_vector_array = Kokkos::View<real_type[dim][p][nodes1D]>;
  using scs_tensor_array = Kokkos::View<real_type[dim][dim][p][nodes1D]>;
};

template <unsigned p>
struct HexViews
{
  constexpr static int poly_order = p;
  constexpr static int nodes1D = p + 1;
  constexpr static int dim = 3;
  constexpr static int nodesPerElement = nodes1D * nodes1D * nodes1D;
  using real_type = double;
  using connectivity_type = int;

  using matrix_array = Kokkos::View<real_type[nodesPerElement][nodesPerElement]>;

  // arrays for nodal variables
  using connectivity_array = Kokkos::View<connectivity_type[nodes1D][nodes1D][nodes1D]>;
  using nodal_scalar_array = Kokkos::View<real_type[nodes1D][nodes1D][nodes1D]>;
  using nodal_vector_array = Kokkos::View<real_type[dim][nodes1D][nodes1D][nodes1D]>;
  using nodal_tensor_array = Kokkos::View<real_type[dim][dim][nodes1D][nodes1D][nodes1D]>;

  // arrays for variables evaluated at subcontrol surfaces in one direction, e.g.
  // at a constant xhat surface
  using scs_scalar_array = Kokkos::View<real_type[p][nodes1D]>;
  using scs_vector_array = Kokkos::View<real_type[dim][p][nodes1D][nodes1D]>;
  using scs_tensor_array = Kokkos::View<real_type[dim][dim][p][nodes1D][nodes1D]>;
};



} // namespace naluUnit
} // namespace Sierra

#endif
