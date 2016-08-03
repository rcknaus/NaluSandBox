/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef KokkosInterface_h
#define KokkosInterface_h

#include <Kokkos_Core.hpp>

namespace stk {namespace mesh {struct Entity; }}

namespace sierra {
namespace naluUnit {

// commonly used (statically-sized) arrays for different element topologies
template <unsigned poly_order, typename Scalar = double>
struct CoefficientViews
{
  constexpr static int nodes1D = poly_order + 1;
  using data_type = Scalar;

  // scs matrices are padded to be square at the moment, so nodal/scs matrices are the same type
  using scs_matrix_array = Kokkos::View<data_type[nodes1D][nodes1D]>;
  using nodal_matrix_array = Kokkos::View<data_type[nodes1D][nodes1D]>;

  using linear_nodal_matrix_array = Kokkos::View<double[2][poly_order+1]>;
  using linear_scs_matrix_array = Kokkos::View<double[2][poly_order]>;
};


template <unsigned poly_order, typename Scalar = double>
struct LineViews  {
  constexpr static int nodes1D = poly_order + 1;
  constexpr static int dim = 2;
  using data_type = Scalar;

  using connectivity_array = Kokkos::View<stk::mesh::Entity[nodes1D][nodes1D][nodes1D]>;
  using nodal_scalar_array = Kokkos::View<data_type[nodes1D]>;
};

template <unsigned poly_order, typename Scalar = double>
struct QuadViews
{
  constexpr static int nodes1D = poly_order + 1;
  constexpr static int dim = 2;
  using data_type = Scalar;

  using matrix_array = Kokkos::View<data_type[(nodes1D)*(nodes1D)][(nodes1D)*(nodes1D)]>;

  // arrays for nodal variables
  using connectivity_array = Kokkos::View<stk::mesh::Entity[nodes1D][nodes1D]>;
  using nodal_scalar_array = Kokkos::View<data_type[nodes1D][nodes1D]>;
  using nodal_vector_array = Kokkos::View<data_type[dim][nodes1D][nodes1D]>;
  using nodal_tensor_array = Kokkos::View<data_type[dim][dim][nodes1D][nodes1D]>;

  // arrays for variables evaluated at subcontrol surfaces in 1 direction, e.g.
  // at a constant xhat line
  using scs_array = Kokkos::View<data_type[poly_order][nodes1D]>;
  using scs_vector_array = Kokkos::View<data_type[dim][poly_order][nodes1D]>;
  using scs_tensor_array = Kokkos::View<data_type[dim][dim][poly_order][nodes1D]>;
};

template <unsigned poly_order, typename Scalar = double>
struct HexViews
{
  constexpr static int nodes1D = poly_order + 1;
  constexpr static int dim = 3;
  using data_type = Scalar;

  using matrix_array = Kokkos::View<data_type[nodes1D*nodes1D*nodes1D][nodes1D*nodes1D*nodes1D]>;

  // arrays for nodal variables
  using connectivity_array = Kokkos::View<stk::mesh::Entity[nodes1D][nodes1D][nodes1D]>;
  using nodal_scalar_array = Kokkos::View<data_type[nodes1D][nodes1D][nodes1D]>;
  using nodal_vector_array = Kokkos::View<data_type[dim][nodes1D][nodes1D][nodes1D]>;
  using nodal_tensor_array = Kokkos::View<data_type[dim][dim][nodes1D][nodes1D][nodes1D]>;

  // arrays for variables evaluated at subcontrol surfaces in 1 direction, e.g.
  // at a constant xhat line
  using scs_scalar_array = Kokkos::View<data_type[poly_order][nodes1D]>;
  using scs_vector_array = Kokkos::View<data_type[dim][poly_order][nodes1D][nodes1D]>;
  using scs_tensor_array = Kokkos::View<data_type[dim][dim][poly_order][nodes1D][nodes1D]>;
};



} // namespace naluUnit
} // namespace Sierra

#endif
