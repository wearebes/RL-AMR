// C ABI only: no Kratos C++ objects or compiler ABI cross this boundary.
#include "Mesquite_all_headers.hpp"
#include "ElementQM.hpp"
#include "ObjectiveFunctionTemplate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <vector>

using namespace Mesquite;

extern "C"
{
    struct SmoothingResult
    {
        int32_t status; // 0 skipped, 1 accepted, 2 rejected, 3 invalid input, 4 exception
        int32_t called;
        int32_t selected;
        int32_t moved;
        double seconds;
    };
    int mesquite_smoothing_abi() { return 5; }
    const char *mesquite_smoothing_version() { return Mesquite::version_string(false); }
}

namespace
{

    class RegularizedMeanRatio : public ElementQM
    {
        double delta;

        bool element_value(PatchData &patch, size_t handle, double &quality,
                           Vector3D *gradient, MsqError &err)
        {
            const MsqVertex *vertices = patch.get_vertex_array(err);
            if (err)
                return false;
            const size_t *index = patch.element_by_index(handle).get_vertex_index_array();
            const Vector3D u = vertices[index[1]] - vertices[index[0]];
            const Vector3D v = (2.0 * (vertices[index[2]] - vertices[index[0]]) - u) / std::sqrt(3.0);
            const double norm2 = (u % u) + (v % v);
            const double determinant = u[0] * v[1] - u[1] * v[0];
            const double radius = std::hypot(determinant, 2.0 * delta);
            if (!(norm2 > 0) || !std::isfinite(norm2) || !std::isfinite(radius))
            {
                MSQ_SETERR(err)("Undefined regularized mean ratio", MsqError::INVALID_MESH);
                return false;
            }
            // Rationalized negative branch avoids cancellation in d + sqrt(d*d+4*delta*delta).
            const double h = determinant >= 0 ? 0.5 * (determinant + radius)
                                              : 2.0 * delta * delta / (radius - determinant);
            quality = 2.0 * h / norm2;
            if (!std::isfinite(quality))
            {
                MSQ_SETERR(err)("Nonfinite regularized mean ratio", MsqError::INVALID_MESH);
                return false;
            }
            if (gradient)
            {
                // d(q)/dS = q * (cofactor(S)/radius - 2*S/||S||_F^2).
                const Vector3D du = quality * (Vector3D(v[1], -v[0], 0) / radius - 2.0 * u / norm2);
                const Vector3D dv = quality * (Vector3D(-u[1], u[0], 0) / radius - 2.0 * v / norm2);
                gradient[0] = -du - dv / std::sqrt(3.0);
                gradient[1] = du - dv / std::sqrt(3.0);
                gradient[2] = 2.0 * dv / std::sqrt(3.0);
            }
            return true;
        }

    public:
        explicit RegularizedMeanRatio(double regularization_delta) : delta(regularization_delta) {}
        std::string get_name() const override { return "Regularized Mean Ratio"; }
        int get_negate_flag() const override { return -1; }

        bool evaluate(PatchData &patch, size_t handle, double &quality, MsqError &err) override
        {
            return element_value(patch, handle, quality, nullptr, err);
        }

        bool evaluate_with_gradient(PatchData &patch, size_t handle, double &quality,
                                    std::vector<size_t> &indices, std::vector<Vector3D> &gradient,
                                    MsqError &err) override
        {
            Vector3D derivatives[3];
            if (!element_value(patch, handle, quality, derivatives, err))
                return false;
            const size_t *index = patch.element_by_index(handle).get_vertex_index_array();
            indices.clear();
            gradient.clear();
            for (size_t j = 0; j < 3; ++j)
                if (index[j] < patch.num_free_vertices())
                {
                    indices.push_back(index[j]);
                    gradient.push_back(derivatives[j]);
                }
            return true;
        }
    };

    // Minimize -softmin(q): F = log(sum(exp(-alpha*q)))/alpha.
    // Mesquite's PR-CG uses CALCULATE evaluations for the existing local/Nash patches.
    class SoftMinObjective : public ObjectiveFunctionTemplate
    {
        double alpha;
        struct Evaluation
        {
            double quality;
            std::vector<size_t> indices;
            std::vector<Vector3D> gradient;
        };

        bool calculate(EvalType type, PatchData &patch, double &value, bool free_only,
                       std::vector<Vector3D> *gradient, MsqError &err)
        {
            if (type != CALCULATE)
            {
                MSQ_SETERR(err)("SoftMinObjective supports local patch evaluation only", MsqError::INVALID_STATE);
                return false;
            }
            QualityMetric *metric = get_quality_metric();
            std::vector<size_t> handles;
            metric->get_evaluations(patch, handles, free_only, err);
            if (err)
                return false;
            if (handles.empty())
            {
                MSQ_SETERR(err)("SoftMinObjective has no element evaluations", MsqError::INVALID_STATE);
                return false;
            }
            std::vector<Evaluation> samples(handles.size());
            double minimum = std::numeric_limits<double>::infinity();
            for (size_t e = 0; e < handles.size(); ++e)
            {
                Evaluation &sample = samples[e];
                const bool valid = gradient
                                       ? metric->evaluate_with_gradient(patch, handles[e], sample.quality, sample.indices, sample.gradient, err)
                                       : metric->evaluate(patch, handles[e], sample.quality, err);
                if (err || !valid)
                    return false;
                minimum = std::min(minimum, sample.quality);
            }
            // Shift by the minimum quality; every exponential is in [0,1].
            double sum = 0;
            for (const Evaluation &sample : samples)
                sum += std::exp(-alpha * (sample.quality - minimum));
            value = -minimum + std::log(sum) / alpha;
            if (gradient)
            {
                gradient->assign(patch.num_free_vertices(), Vector3D(0, 0, 0));
                for (const Evaluation &sample : samples)
                {
                    const double weight = std::exp(-alpha * (sample.quality - minimum)) / sum;
                    for (size_t j = 0; j < sample.indices.size(); ++j)
                        (*gradient)[sample.indices[j]] -= weight * sample.gradient[j];
                }
            }
            return true;
        }

    public:
        SoftMinObjective(QualityMetric *metric, double softmin_alpha)
            : ObjectiveFunctionTemplate(metric), alpha(softmin_alpha) {}
        ObjectiveFunction *clone() const override { return new SoftMinObjective(get_quality_metric(), alpha); }
        void clear() override {}
        bool evaluate(EvalType type, PatchData &patch, double &value, bool free_only, MsqError &err) override
        {
            return calculate(type, patch, value, free_only, nullptr, err);
        }
        bool evaluate_with_gradient(EvalType type, PatchData &patch, double &value,
                                    std::vector<Vector3D> &gradient, MsqError &err) override
        {
            return calculate(type, patch, value, true, &gradient, err);
        }
    };

    std::string error_text(const MsqError &err)
    {
        std::ostringstream text;
        text << err;
        return text.str();
    }

    // Buffer checks only. Element quality and orientation belong to Mesquite.
    bool valid_buffers(int64_t nv, const double *x, int64_t ne, const int64_t *tri)
    {
        if (!x || !tri || nv <= 0 || ne <= 0)
            return false;
        for (int64_t i = 0; i < 3 * nv; ++i)
            if (!std::isfinite(x[i]))
                return false;
        for (int64_t i = 0; i < nv; ++i)
            if (x[3 * i + 2] != x[2])
                return false;
        for (int64_t i = 0; i < 3 * ne; ++i)
            if (tri[i] < 0 || tri[i] >= nv)
                return false;
        return true;
    }

    bool assess(ArrayMesh &mesh, MeshDomainAssoc &assoc, double &mean, double &minimum,
                std::string &message, double *values = nullptr)
    {
        IdealWeightMeanRatio metric;
        // Include ALL elements, including those whose vertices are all fixed.
        const char *tag_name = values ? "fsi2_mean_ratio" : nullptr;
        QualityAssessor assessor(&metric, 0, 0.0, false, tag_name, false);
        MsqError err;
        assessor.loop_over_mesh(&assoc, nullptr, err);
        if (err)
        {
            message = error_text(err);
            return false;
        }
        int inverted = 0, samples = 0;
        const bool counted = assessor.get_inverted_element_count(inverted, samples, err);
        const QualityAssessor::Assessor *stats = assessor.get_results(&metric);
        if (err)
        {
            message = error_text(err);
            return false;
        }
        if (!counted || !stats || inverted || stats->get_invalid_element_count() || assessor.invalid_elements())
        {
            message = "Mesquite QualityAssessor: inverted/degenerate elements or invalid mean-ratio values";
            return false;
        }
        mean = stats->get_average();
        minimum = stats->get_minimum();
        if (!std::isfinite(mean) || !std::isfinite(minimum))
        {
            message = "Mesquite returned nonfinite quality statistics";
            return false;
        }
        if (values)
        {
            std::vector<Mesh::ElementHandle> elements;
            mesh.get_all_elements(elements, err);
            TagHandle tag = mesh.tag_get(tag_name, err);
            // ArrayMesh native element handles are the input connectivity row indices.
            if (!err)
                mesh.tag_get_element_data(tag, elements.size(), elements.data(), values, err);
            if (err)
            {
                message = error_text(err);
                return false;
            }
        }
        return true;
    }
}

extern "C" int mesquite_measure(int64_t nv, double *x, int64_t ne, const int64_t *tri,
                                double *values, double *mean, double *minimum, double *min_j, char *error, uint64_t error_size)
{
    if (!error || !error_size)
        return 3;
    error[0] = 0;
    auto fail = [&](const std::string &message)
    {
        std::snprintf(error, error_size, "%s", message.c_str());
        return 3;
    };
    if (!values || !mean || !minimum || !valid_buffers(nv, x, ne, tri))
        return fail("Invalid/nonfinite/nonplanar buffers or out-of-range connectivity");
    try
    {
        std::vector<unsigned long> connectivity(tri, tri + 3 * ne);
        std::vector<int> fixed(nv, 1);
        ArrayMesh mesh(3, nv, x, fixed.data(), ne, TRIANGLE, connectivity.data());
        PlanarDomain plane(PlanarDomain::XY, x[2]);
        MeshDomainAssoc assoc(&mesh, &plane);
        std::string message;
        if (!assess(mesh, assoc, *mean, *minimum, message, values))
            return fail(message);
        if (min_j)
        {
            MsqError err;
            PatchData patch;
            patch.set_mesh(&mesh);
            patch.set_domain(&plane);
            patch.fill_global_patch(err);
            if (err)
                return fail(error_text(err));
            *min_j = std::numeric_limits<double>::infinity();
            for (size_t e = 0; e < patch.num_elements(); ++e)
            {
                // Linear XY triangle: det(J) = twice Mesquite's signed area.
                const double jac = 2 * patch.element_by_index(e).compute_signed_area(patch, err);
                if (err)
                    return fail(error_text(err));
                if (!std::isfinite(jac))
                    return fail("Nonfinite signed Jacobian");
                *min_j = std::min(*min_j, jac);
            }
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        return fail(e.what());
    }
    catch (...)
    {
        return fail("Unknown Mesquite measurement exception");
    }
}

extern "C" int mesquite_smooth(
    int64_t nv, double *x, int64_t ne, const int64_t *tri, const int32_t *boundary,
    double threshold, int64_t budget, int32_t inner_limit, int32_t outer_limit,
    double regularization_delta, double softmin_alpha,
    SmoothingResult *out, char *error, uint64_t error_size)
{
    if (!out || !error || !error_size)
        return 3;
    std::memset(out, 0, sizeof(*out));
    error[0] = 0;
    auto fail = [&](int status, const std::string &msg)
    {
        out->status = status;
        std::snprintf(error, error_size, "%s", msg.c_str());
        return status;
    };
    if (!valid_buffers(nv, x, ne, tri) || !boundary || budget < 0 || budget > std::numeric_limits<int32_t>::max() || !std::isfinite(threshold) || threshold < 0 || threshold > 1 || inner_limit < 1 || outer_limit < 1 || !std::isfinite(regularization_delta) || regularization_delta <= 0 || !std::isfinite(softmin_alpha) || softmin_alpha <= 0)
        return fail(3, "Invalid bridge arguments");
    std::vector<double> original;
    try
    {
        std::vector<double> q(ne);
        const double plane_z = x[2];
        std::vector<unsigned long> connectivity(tri, tri + 3 * ne);
        std::vector<int> fixed(nv, 1);
        ArrayMesh mesh(3, nv, x, fixed.data(), ne, TRIANGLE, connectivity.data());
        PlanarDomain plane(PlanarDomain::XY, plane_z);
        MeshDomainAssoc assoc(&mesh, &plane);
        std::string message;
        double mean = 0, minimum = 0;
        if (!assess(mesh, assoc, mean, minimum, message, q.data()))
            return fail(3, "Input mesh: " + message);
        // Union of vertices from bad elements; repeated incidences set the same flag.
        std::vector<unsigned char> candidate(nv, 0);
        for (int64_t e = 0; e < ne; ++e)
            if (q[e] < threshold)
                for (int j = 0; j < 3; ++j)
                    candidate[tri[3 * e + j]] = 1;
        // Python supplies vertices in ascending real Node ID. Take the first
        // budget free candidates in that order; every other vertex stays fixed.
        for (int64_t i = 0; i < nv && out->selected < budget; ++i)
            if (candidate[i] && !boundary[i])
            {
                fixed[i] = 0;
                ++out->selected;
            }
        if (!out->selected)
            return 0;
        original.assign(x, x + 3 * nv);
        MsqError err;
        RegularizedMeanRatio metric(regularization_delta);
        SoftMinObjective objective(&metric, softmin_alpha);
        ConjugateGradient optimizer(&objective);
        optimizer.use_element_on_vertex_patch();
        TerminationCriterion inner, outer;
        inner.add_iteration_limit(inner_limit);
        outer.add_iteration_limit(outer_limit);
        optimizer.set_inner_termination_criterion(&inner);
        optimizer.set_outer_termination_criterion(&outer);
        InstructionQueue queue;
        queue.set_master_quality_improver(&optimizer, err);
        if (err)
            return fail(2, error_text(err));
        const auto start = std::chrono::steady_clock::now();
        out->called = 1;
        try
        {
            queue.run_instructions(&assoc, err);
        }
        catch (...)
        {
            out->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            throw;
        }
        out->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::string rejection;
        if (err)
            rejection = error_text(err);
        if (rejection.empty() && !std::all_of(x, x + 3 * nv, [](double v)
                                              { return std::isfinite(v); }))
            rejection = "Optimized coordinates are nonfinite";
        for (int64_t i = 0; i < nv && rejection.empty(); ++i)
        {
            if (fixed[i] && std::memcmp(x + 3 * i, original.data() + 3 * i, 3 * sizeof(double)))
                rejection = "Mesquite moved a fixed/nonselected vertex";
            if (x[3 * i + 2] != plane_z)
                rejection = "Mesquite moved a vertex out of the XY plane";
        }
        if (rejection.empty())
            assess(mesh, assoc, mean, minimum, rejection);
        if (!rejection.empty())
        {
            std::copy(original.begin(), original.end(), x);
            return fail(2, rejection);
        }
        for (int64_t i = 0; i < nv; ++i)
            if (std::memcmp(x + 3 * i, original.data() + 3 * i, 3 * sizeof(double)))
                ++out->moved;
        out->status = 1;
        return 1;
    }
    catch (const std::exception &e)
    {
        if (!original.empty())
            std::copy(original.begin(), original.end(), x);
        return fail(4, e.what());
    }
    catch (...)
    {
        if (!original.empty())
            std::copy(original.begin(), original.end(), x);
        return fail(4, "Unknown native exception");
    }
}
