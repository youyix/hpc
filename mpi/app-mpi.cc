/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth and Ralf Hartmann, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include <typeinfo>
#include <fstream>
#include <iostream>
 #include <time.h>

namespace AppMpi
{
  using namespace dealii;

// Solutions

  template <int dim>
  class SolutionBase
  {
  protected:
    static const unsigned int n_source_centers = 3;
    static const Point<dim>   source_centers[n_source_centers];
    static const double       width;
  };


  template <>
  const Point<1>
  SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers]
    = { Point<1>(-1.0 / 3.0),
        Point<1>(0.0),
        Point<1>(+1.0 / 3.0)
      };

  template <>
  const Point<2>
  SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers]
    = { Point<2>(-0.5, +0.5),
        Point<2>(-0.5, -0.5),
        Point<2>(+0.5, -0.5)
      };

  template <int dim>
  const double SolutionBase<dim>::width = 1./8.;



  template <int dim>
  class Solution : public Function<dim>,
    protected SolutionBase<dim>
  {
  public:
    Solution () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
  };


  template <int dim>
  double Solution<dim>::value (const Point<dim>   &p,
                               const unsigned int) const
  {
    double return_value = 0;
    for (unsigned int i=0; i<this->n_source_centers; ++i)
      {
        const Point<dim> x_minus_xi = p - this->source_centers[i];
        return_value += std::exp(-x_minus_xi.square() /
                                 (this->width * this->width));
      }

    return return_value;
  }


  template <int dim>
  Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                         const unsigned int) const
  {
    Tensor<1,dim> return_value;

    for (unsigned int i=0; i<this->n_source_centers; ++i)
      {
        const Point<dim> x_minus_xi = p - this->source_centers[i];

        return_value += (-2 / (this->width * this->width) *
                         std::exp(-x_minus_xi.square() /
                                  (this->width * this->width)) *
                         x_minus_xi);
      }

    return return_value;
  }



  template <int dim>
  class HelmholtzProblem
  {
  public:
    // enum RefinementMode
    // {
    //   global_refinement, adaptive_refinement
    // };

    HelmholtzProblem ();
    ~HelmholtzProblem ();

    void run ();

  private:
    void setup_system ();
    void assemble_system ();
    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    MPI_Comm                                  mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    FE_Q<dim>                                 fe;
    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix                          constraints;

    LA::MPI::SparseMatrix                     system_matrix;
    LA::MPI::Vector                           locally_relevant_solution;
    LA::MPI::Vector                           system_rhs;


    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;

    // const RefinementMode                      refinement_mode = RefinementMode::global_refinement;

  };


  template <int dim>
  HelmholtzProblem<dim>::HelmholtzProblem () 
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    fe (2),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
    
  {}



  template <int dim>
  HelmholtzProblem<dim>::~HelmholtzProblem ()
  {
    dof_handler.clear ();
  }



  template <int dim>
  void HelmholtzProblem<dim>::setup_system ()
  {

    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs (fe);

    DoFRenumbering::subdomain_wise (dof_handler);

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                         locally_relevant_dofs);


    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);

    // TODO with BC
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close ();


    CompressedSimpleSparsityPattern csp (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, csp,
                                     constraints, false);

    SparsityTools::distribute_sparsity_pattern (csp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);
    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          csp,
                          mpi_communicator);
    // dof_handler.distribute_dofs (fe);
    // DoFRenumbering::Cuthill_McKee (dof_handler);

    // constraints.clear ();
    // DoFTools::make_constraints (dof_handler, constraints);
    // constraints.close ();

    // sparsity_pattern.reinit (dof_handler.n_dofs(),
    //                          dof_handler.n_dofs(),
    //                          dof_handler.max_couplings_between_dofs());
    // DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    // constraints.condense (sparsity_pattern);
    // sparsity_pattern.compress();

    // system_matrix.reinit (sparsity_pattern);

    // solution.reinit (dof_handler.n_dofs());
    // system_rhs.reinit (dof_handler.n_dofs());
  }



  template <int dim>
  void HelmholtzProblem<dim>::assemble_system ()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);

    const Solution<dim> exact_solution;
    const unsigned int   dofs_per_cell    = fe.dofs_per_cell;
    const unsigned int   n_q_points       = quadrature_formula.size();
    const unsigned int   n_face_q_points  = face_quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    //
    const unsigned int n_source_centers = 3;
    const Point<dim>   source_centers[n_source_centers] 
      = { Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5) };
    const double       width = 1./8.;
    //
  
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit (cell);
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point) 
              {

                const Point<dim> p = fe_values.quadrature_point(q_point);
                double rhs_value = 0;
                for (unsigned int ni=0; ni<n_source_centers; ++ni)
                  {
                    const Point<dim> x_minus_xi = p - source_centers[ni];
                    rhs_value += ((2*dim - 4*x_minus_xi.square()/
                                      (width * width)) /
                                     (width * width) *
                                     std::exp(-x_minus_xi.square() /
                                              (width * width)));
                    rhs_value += std::exp(-x_minus_xi.square() /
                                             (width * width));
                  }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  { 
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      {
                        cell_matrix(i,j) += ((fe_values.shape_grad(i,q_point) *
                                            fe_values.shape_grad(j,q_point)
                                            +
                                            fe_values.shape_value(i,q_point) *
                                            fe_values.shape_value(j,q_point)) *
                                           fe_values.JxW(q_point));
                      }
                      
                      cell_rhs(i) += (fe_values.shape_value(i,q_point) *
                                rhs_value *
                                fe_values.JxW(q_point));
                  }
              }

            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
              if (cell->face(face_number)->at_boundary()
                  &&
                  (cell->face(face_number)->boundary_indicator() == 1))
                {
                  fe_face_values.reinit (cell, face_number);

                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      const double neumann_value
                        = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                           fe_face_values.normal_vector(q_point));

                      for (unsigned int i=0; i<dofs_per_cell; ++i)
                        cell_rhs(i) += (neumann_value *
                                        fe_face_values.shape_value(i,q_point) *
                                        fe_face_values.JxW(q_point));
                    }
                }
              }

              cell->get_dof_indices (local_dof_indices);
              constraints.distribute_local_to_global (cell_matrix,
                                                      cell_rhs,
                                                      local_dof_indices,
                                                      system_matrix,
                                                      system_rhs);
      }
    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);

  }
      // constraints.condense (system_matrix);
    // constraints.condense (system_rhs);


  template <int dim>
  void HelmholtzProblem<dim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector
    completely_distributed_solution (locally_owned_dofs, mpi_communicator);

    SolverControl solver_control (dof_handler.n_dofs(), 1e-12);

    LA::SolverCG solver(solver_control, mpi_communicator);
    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif
    preconditioner.initialize(system_matrix, data);

    solver.solve (system_matrix, completely_distributed_solution, system_rhs,
                  preconditioner);

    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;

    constraints.distribute (completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }


  template <int dim>
  void HelmholtzProblem<dim>::refine_grid ()
  {
    triangulation.refine_global (1);
    // switch (refinement_mode)
    //   {
    //   case global_refinement:
    //   {
    //     triangulation.refine_global (1);
    //     break;
    //   }

    //   case adaptive_refinement:
    //   {
    //     Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    //     KellyErrorEstimator<dim>::estimate (dof_handler,
    //                                         QGauss<dim-1>(3),
    //                                         typename FunctionMap<dim>::type(),
    //                                         locally_relevant_solution,
    //                                         estimated_error_per_cell);

    //     GridRefinement::refine_and_coarsen_fixed_number (triangulation,
    //                                                      estimated_error_per_cell,
    //                                                      0.3, 0.03);

    //     triangulation.execute_coarsening_and_refinement ();

    //     break;
    //   }

    //   default:
    //   {
    //     Assert (false, ExcNotImplemented());
    //   }
    //   }
  }

  template <int dim>
  void HelmholtzProblem<dim>::output_results (const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "u");

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();

    // The next step is to write this data to disk. We choose file names of
    // the form <code>solution-XX-PPPP.vtu</code> where <code>XX</code>
    // indicates the refinement cycle, <code>PPPP</code> refers to the
    // processor number (enough for up to 10,000 processors, though we hope
    // that nobody ever tries to generate this much data -- you would likely
    // overflow all file system quotas), and <code>.vtu</code> indicates the
    // XML-based Visualization Toolkit (VTK) file format.
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    // The last step is to write a "master record" that lists for the
    // visualization program the names of the various files that combined
    // represents the graphical data for the entire domain. The
    // DataOutBase::write_pvtu_record does this, and it needs a list of
    // filenames that we create first. Note that only one processor needs to
    // generate this file; we arbitrarily choose processor zero to take over
    // this job.
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 2) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");

        std::ofstream master_output ((filename + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
  }

  template <int dim>
  void HelmholtzProblem<dim>::run ()
  {

    const unsigned int n_cycles = 1;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (5);

            // typename Triangulation<dim>::cell_iterator
            // cell = triangulation.begin (),
            // endc = triangulation.end();
            // for (; cell!=endc; ++cell)
            //   for (unsigned int face_number=0;
            //        face_number<GeometryInfo<dim>::faces_per_cell;
            //        ++face_number)
            //     if (( std::fabs(  cell->face(face_number)->center()(0) - (-1)  ) < 1e-12 )
            //         ||
            //         (std::fabs(cell->face(face_number)->center()(1) - (-1)) < 1e-12))
            //       cell->face(face_number)->set_boundary_indicator (1);
          }
        else
          refine_grid ();

        setup_system ();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

        assemble_system ();
        solve ();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results (cycle);
          }

        computing_timer.print_summary ();
        computing_timer.reset ();

        pcout << std::endl;
      }

    // Mark Nueman Boundaries to `1`, which will not be changed by D Condition
    // {
    //     GridGenerator::hyper_cube (triangulation, -1, 1);
    //     triangulation.refine_global (1);

    //     typename Triangulation<dim>::cell_iterator
    //     cell = triangulation.begin (),
    //     endc = triangulation.end();
    //     for (; cell!=endc; ++cell)
    //       for (unsigned int face_number=0;
    //            face_number<GeometryInfo<dim>::faces_per_cell;
    //            ++face_number)
    //         if (( std::fabs(  cell->face(face_number)->center()(0) - (-1)  ) < 1e-12 )
    //             ||
    //             (std::fabs(cell->face(face_number)->center()(1) - (-1)) < 1e-12))
    //           cell->face(face_number)->set_boundary_indicator (1);
    // }
    // for ( int i=0; i<5; i++ ) {
    //     refine_grid ();
    // }
    // setup_system ();
    // assemble_system ();
    // solve (steps);
    // process_solution (steps);


  }

}
namespace AppMpi
{
  template const double SolutionBase<2>::width;
}

int main (int argc, char *argv[])
{
  const unsigned int dim = 2;


  try
    {
      using namespace dealii;
      using namespace AppMpi;

      deallog.depth_console (0);
      {
        HelmholtzProblem<dim> helmholtz_problem_2d;
        // helmholtz_problem_2d.run ();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}



