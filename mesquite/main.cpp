/*
 * Standalone Mesquite 2.3.0 example. No Kratos or Python bindings.
 *
 * Build (from RL-AMR/mesquite):
 *   clang++ -std=c++98 main.cpp -I install/include install/lib/libmesquite.a -o smooth
 * Run the built-in square with one displaced interior vertex:
 *   ./smooth
 * Run an exported notebook mesh (ASCII legacy VTK, linear triangles, z=0):
 *   ./smooth square.vtk both 10 10 square
 * Arguments: input.vtk smart|prcg|both sweeps cg_inner [output_prefix]
 * Only an explicit output_prefix writes files. Each method starts from the
 * SAME initial coordinates. Outer and hole boundaries are inferred from edges
 * with exactly one incident triangle, and all those vertices are fixed.
 *
 * MATHEMATICS (valid counterclockwise triangles):
 *   q(T) = 4 sqrt(3) area(T) / sum(edge_length^2),  0 < q <= 1.
 *   Report q_min, q_mean and mean(1/q) for BOTH methods.
 *
 * SMART LAPLACIAN -- original SmartLaplacianSmoother, not LaplaceWrapper:
 *   trial x_i = sum(x_j for adjacent vertices j) / degree(i).
 *   Mesquite projects onto the domain and reverts the trial ONLY if the
 *   number of inverted incident elements increases. It does NOT test whether
 *   q_min or q_mean improves. This is exactly the 2.3.0 implementation.
 *
 * LOCAL CONJUGATE GRADIENT -- original ConjugateGradient:
 *   IdealWeightInverseMeanRatio gives m(T)=1/q(T) on these triangles.
 *   LPtoPTemplate(p=1, dividing_by_n=true) minimizes
 *       F_i(x_i) = mean(m(T) for T incident on i),
 *   with all other vertices held fixed in this one-vertex patch.
 *   Mesquite computes the analytic metric gradient and uses its own PR rule:
 *       d_0=-g_0,
 *       beta_k = g_{k+1}.(g_{k+1}-g_k) / (g_k.g_k),
 *       d_{k+1}=-g_{k+1}+beta_k*d_k.
 *   Its original get_step searches feasible, decreasing steps by factors of
 *   two and restarts with -g if no step is found. No custom line search or
 *   clipping of beta is added here. Inner termination: iteration cap OR
 *   absolute L2 gradient <= 1e-8. Outer termination: requested mesh sweeps.
 *   This objective is NOT the notebook's regularized soft-min objective.
 *
 * Both methods use Mesquite's sequential patch traversal (default non-Jacobi),
 * not a promised notebook ascending-node order. The wrapper below only does
 * mesh I/O, boundary marking, calls, timing and final validity checks.
 */
#include "ArrayMesh.hpp"
#include "MeshImpl.hpp"
#include "MsqVertex.hpp"
#include "MsqError.hpp"
#include "PlanarDomain.hpp"
#include "SmartLaplacianSmoother.hpp"
#include "ConjugateGradient.hpp"
#include "IdealWeightInverseMeanRatio.hpp"
#include "LPtoPTemplate.hpp"
#include "InstructionQueue.hpp"
#include "TerminationCriterion.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Mesquite;

struct MeshData {
    std::vector<double> xyz;
    std::vector<unsigned long> triangles;
    std::vector<int> fixed;
};

void check(MsqError& err)
{
    if (err) {
        std::ostringstream message;
        message << err;
        throw std::runtime_error(message.str());
    }
}

MeshData read_mesh(const char* filename)
{
    MeshImpl input;
    MsqError err;
    input.read_vtk(filename, err); check(err);
    std::vector<Mesh::VertexHandle> vertices, connected;
    std::vector<Mesh::ElementHandle> elements;
    input.get_all_vertices(vertices, err); check(err);
    input.get_all_elements(elements, err); check(err);
    if (vertices.empty() || elements.empty())
        throw std::runtime_error("Empty mesh.");
    std::vector<MsqVertex> points(vertices.size());
    input.vertices_get_coordinates(&vertices[0], &points[0], vertices.size(), err);
    check(err);
    std::map<Mesh::VertexHandle, unsigned long> index;
    MeshData data;
    for (size_t i = 0; i < vertices.size(); ++i) {
        index[vertices[i]] = static_cast<unsigned long>(i);
        for (int d = 0; d < 3; ++d) data.xyz.push_back(points[i][d]);
    }
    std::vector<EntityTopology> types(elements.size());
    std::vector<size_t> offsets;
    input.elements_get_topologies(&elements[0], &types[0], elements.size(), err);
    check(err);
    input.elements_get_attached_vertices(&elements[0], elements.size(),
                                         connected, offsets, err); check(err);
    for (size_t t = 0; t < elements.size(); ++t) {
        const size_t end = t + 1 < elements.size() ? offsets[t+1] : connected.size();
        if (types[t] != TRIANGLE || end - offsets[t] != 3)
            throw std::runtime_error("Only linear triangles are supported.");
        for (size_t j = offsets[t]; j < end; ++j)
            data.triangles.push_back(index[connected[j]]);
    }
    return data;
}

MeshData demo_mesh()
{
    // Four fixed corners, one free point initially at (0.23,0.32).
    const double xyz[] = {0,0,0, 1,0,0, 1,1,0, 0,1,0, 0.23,0.32,0};
    const unsigned long triangles[] = {0,1,4, 1,2,4, 2,3,4, 3,0,4};
    MeshData data;
    data.xyz.assign(xyz, xyz + 15);
    data.triangles.assign(triangles, triangles + 12);
    return data;
}

void fix_boundaries(MeshData& data)
{
    typedef std::pair<unsigned long, unsigned long> Edge;
    std::map<Edge, int> counts;
    for (size_t t = 0; t < data.triangles.size(); t += 3) {
        for (int j = 0; j < 3; ++j) {
            unsigned long a = data.triangles[t+j], b = data.triangles[t+(j+1)%3];
            if (a > b) std::swap(a,b);
            ++counts[Edge(a,b)];
        }
    }
    data.fixed.assign(data.xyz.size()/3, 0);
    for (std::map<Edge,int>::const_iterator e = counts.begin(); e != counts.end(); ++e) {
        if (e->second > 2) throw std::runtime_error("Non-manifold edge.");
        if (e->second == 1) data.fixed[e->first.first] = data.fixed[e->first.second] = 1;
    }
}

void print_quality(const MeshData& data, const std::string& label)
{
    double q_min = 1.0, q_sum = 0.0, inverse_sum = 0.0;
    for (size_t i = 0; i < data.xyz.size(); i += 3) {
        if (!std::isfinite(data.xyz[i]) || !std::isfinite(data.xyz[i+1]) ||
            data.xyz[i+2] != 0.0)
            throw std::runtime_error("Expected finite coordinates in the z=0 plane.");
    }
    for (size_t t = 0; t < data.triangles.size(); t += 3) {
        const double* a = &data.xyz[3*data.triangles[t]];
        const double* b = &data.xyz[3*data.triangles[t+1]];
        const double* c = &data.xyz[3*data.triangles[t+2]];
        const double area2 = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
        double edges2 = 0.0;
        for (int d = 0; d < 2; ++d)
            edges2 += (a[d]-b[d])*(a[d]-b[d]) + (b[d]-c[d])*(b[d]-c[d])
                    + (c[d]-a[d])*(c[d]-a[d]);
        if (!(area2 > 1e-14 * edges2))
            throw std::runtime_error("Inverted, degenerate or nearly collinear triangle; result rejected.");
        const double q = 2.0*std::sqrt(3.0)*area2/edges2;
        q_min = std::min(q_min,q); q_sum += q; inverse_sum += 1.0/q;
    }
    const double n = data.triangles.size()/3;
    std::cout << label << ": q_min=" << q_min << " q_mean=" << q_sum/n
              << " mean(1/q)=" << inverse_sum/n << " inverted=0 degenerate=0\n";
}

MeshData smooth(const MeshData& initial, const std::string& method, int sweeps, int cg_inner)
{
    MeshData result = initial; // Each method gets a fresh copy; failures never change input.
    ArrayMesh mesh(3, result.fixed.size(), &result.xyz[0], &result.fixed[0],
                   result.triangles.size()/3, TRIANGLE, &result.triangles[0]);
    PlanarDomain plane(PlanarDomain::XY);
    MeshDomainAssoc association(&mesh, &plane);
    MsqError err;
    TerminationCriterion outer, inner;
    outer.add_iteration_limit(sweeps);

    IdealWeightInverseMeanRatio metric;
    LPtoPTemplate objective(1, &metric);
    objective.set_dividing_by_n(true);
    SmartLaplacianSmoother smart;
    ConjugateGradient cg(&objective);
    VertexMover* solver = method == "smart" ? static_cast<VertexMover*>(&smart)
                                           : static_cast<VertexMover*>(&cg);
    if (method == "prcg") {
        cg.use_element_on_vertex_patch(1); // One free center; neighbors fixed in this patch.
        inner.add_iteration_limit(cg_inner);
        inner.add_absolute_gradient_L2_norm(1e-8);
    } else {
        inner.add_iteration_limit(1); // One centroid trial per vertex per mesh sweep.
    }
    solver->set_inner_termination_criterion(&inner);
    solver->set_outer_termination_criterion(&outer);
    InstructionQueue queue;
    queue.set_master_quality_improver(solver, err); check(err);
    const std::clock_t start = std::clock();
    queue.run_instructions(&association, err); check(err);
    const double seconds = double(std::clock()-start)/CLOCKS_PER_SEC;
    for (size_t i = 0; i < initial.fixed.size(); ++i)
        if (initial.fixed[i])
            for (int d = 0; d < 3; ++d)
                if (result.xyz[3*i+d] != initial.xyz[3*i+d])
                    throw std::runtime_error("A fixed boundary vertex moved; result rejected.");
    print_quality(result, method);
    std::cout << "  boundary unchanged; requested sweeps=" << sweeps
              << " cpu_seconds=" << seconds << '\n';
    return result;
}

void write_mesh(const MeshData& data, const std::string& filename)
{
    std::ofstream out(filename.c_str());
    out << std::setprecision(17);
    out << "# vtk DataFile Version 3.0\nMesquite result\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << data.xyz.size()/3 << " double\n";
    for (size_t i = 0; i < data.xyz.size(); i += 3)
        out << data.xyz[i] << ' ' << data.xyz[i+1] << ' ' << data.xyz[i+2] << '\n';
    out << "CELLS " << data.triangles.size()/3 << ' ' << 4*(data.triangles.size()/3) << '\n';
    for (size_t t = 0; t < data.triangles.size(); t += 3)
        out << "3 " << data.triangles[t] << ' ' << data.triangles[t+1] << ' ' << data.triangles[t+2] << '\n';
    out << "CELL_TYPES " << data.triangles.size()/3 << '\n';
    for (size_t t = 0; t < data.triangles.size(); t += 3) out << "5\n";
    out.close();
    if (!out) throw std::runtime_error("Failed to write " + filename);
}

int positive_integer(const char* text)
{
    std::istringstream in(text);
    int value; char extra;
    if (!(in >> value) || value < 1 || (in >> extra))
        throw std::runtime_error("Iteration counts must be positive integers.");
    return value;
}

int main(int argc, char** argv)
{
    try {
        if (argc != 1 && (argc < 3 || argc > 6)) {
            std::cerr << "Usage: " << argv[0]
                      << " [input.vtk smart|prcg|both [sweeps=10 [cg_inner=10 [output_prefix]]]]\n";
            return 1;
        }
        const std::string method = argc == 1 ? "both" : argv[2];
        if (method != "smart" && method != "prcg" && method != "both")
            throw std::runtime_error("Method must be smart, prcg or both.");
        const int sweeps = argc > 3 ? positive_integer(argv[3]) : 10;
        const int inner = argc > 4 ? positive_integer(argv[4]) : 10;
        MeshData initial = argc == 1 ? demo_mesh() : read_mesh(argv[1]);
        fix_boundaries(initial);
        std::cout << std::setprecision(10) << "vertices=" << initial.fixed.size()
                  << " triangles=" << initial.triangles.size()/3 << " fixed="
                  << std::count(initial.fixed.begin(), initial.fixed.end(), 1) << '\n';
        print_quality(initial, "initial");
        const char* methods[] = {"smart", "prcg"};
        for (int j = 0; j < 2; ++j) {
            if (method != "both" && method != methods[j]) continue;
            const MeshData result = smooth(initial, methods[j], sweeps, inner);
            if (argc == 1)
                std::cout << "  demo center=(" << result.xyz[12] << ", " << result.xyz[13] << ")\n";
            if (argc == 6) write_mesh(result, std::string(argv[5])+"_"+methods[j]+".vtk");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
