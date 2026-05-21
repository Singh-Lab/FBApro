# import needed libraries
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from troppo.methods.reconstruction import imat, gimme, fastcore
import pandas as pd
import cobra

class FBApro(nn.Module):
    def __init__(self, stoichiometric_matrix, measured_indices=None, unknown_indices=None,
                 steady_state_basis_matrix=None, acond=0, rcond=1e-3, device=torch.device('cuda'), driver=None,
                 dtype=torch.float64, **kwargs):

        if driver is None:
            if device.type == 'cuda':
                # print("Warning: on CUDA can only use gels driver, which assumes matrices are full-rank")
                driver = 'gels'
            else:
                driver = 'gelsd'

        super().__init__()



        self.have_warned_intersection = False
        self.have_warned_projection = False

        # The projection returns, for an input, a closest steady-state vector that agrees with it on measured_indices,
        # where distances are computed only on indices not in unknown_indices.
        # When computing some auxiliary matrices, acond is absolute conditioning, such that values with smaller
        # absolute values are zerod.
        if measured_indices is None:
            measured_indices = []
        if unknown_indices is None:
            unknown_indices = []

        # assert no intersection between measured and unknown indices
        assert set(measured_indices).isdisjoint(set(unknown_indices))

        # check if any of the index lists is out of order, and order it if so
        if measured_indices != sorted(measured_indices):
            print("Warning: measured indices not in order, sorting them.")
            measured_indices = sorted(measured_indices)
        if unknown_indices != sorted(unknown_indices):
            print("Warning: unknown indices not in order, sorting them.")
            unknown_indices = sorted(unknown_indices)

        if type(stoichiometric_matrix) != torch.Tensor:
            stoichiometric_matrix = torch.tensor(stoichiometric_matrix, dtype=dtype, device=device)

        # Get orthonormal basis for steady-state space
        if steady_state_basis_matrix is None:
            steady_state_basis_matrix = scipy.linalg.null_space(stoichiometric_matrix.cpu().numpy(), rcond=rcond)
        if type(steady_state_basis_matrix) != torch.Tensor:
            steady_state_basis_matrix = torch.tensor(steady_state_basis_matrix, dtype=dtype,
                                                     device=device)
        else:
            steady_state_basis_matrix = steady_state_basis_matrix.to(device).to(dtype)
        assert stoichiometric_matrix.shape[1] == steady_state_basis_matrix.shape[0]
        self.register_buffer('stoic', torch.tensor(stoichiometric_matrix, device=device, dtype=dtype),
                             persistent=True)
        A = steady_state_basis_matrix

        if len(measured_indices) == 0 and len(unknown_indices) == 0:
            # Simply a projection to the steady-state space, since it's span(A) and A is orthogonal,
            # the projection is AA^T
            self.register_buffer('projection_matrix',
                                 A @ torch.transpose(A, 0, 1), persistent=True)
            self.name = "FBAproBasic"
            return

        if len(unknown_indices) > 0:
            # define a restriction to the indices complement set:
            restriction_matrix = torch.zeros(stoichiometric_matrix.shape[1] - len(unknown_indices) - len(measured_indices),
                                             stoichiometric_matrix.shape[1],
                                             dtype=dtype, device=device)
            j = 0
            for i in range(stoichiometric_matrix.shape[1] - len(unknown_indices) - len(measured_indices)):
                while j in unknown_indices or j in measured_indices:
                    j += 1
                restriction_matrix[i, j] = 1
                j += 1
            P = restriction_matrix

        if len(measured_indices) == 0:
            # This is a steady-state projection ignoring some indices using a restriction.
            # Compute A @ (PA)^+ @ P

            # only compute A @ A0^T, where A0 only has the rows of A corresponding to non-unknown indices
            # PA_pinv = torch.linalg.pinv(P @ self.A, rtol=rcond)
            # projection_matrix = self.A @ PA_pinv @ P
            PA_pinv_P = torch.linalg.lstsq(P @ A, P, rcond=rcond)[0]
            projection_matrix = A @ PA_pinv_P
            self.register_buffer('projection_matrix', projection_matrix, persistent=True)

            self.name = "FBAproPartial"
            return

        # Fixing a subset of reactions, with or without ignored indices, is first expressed as a projection
        # to the intersection of two affine spaces, the steady-state and unfixed-degrees-of-freedom space.

        # degrees of freedom - span{e_i, i not in measured_indices}
        B = torch.zeros(A.shape[0], A.shape[0] - len(measured_indices), dtype=dtype,
                        device=device)
        unmeasured_indices = [i for i in range(A.shape[0]) if i not in measured_indices]
        for i, j in enumerate(unmeasured_indices):
            B[j, i] = 1

        # Now construct D = AA^T + BB^T
        # BB^T is the dot products of rows of B. Each row can have at most 1 and the rest 0s, and two different rows
        # can't have 1 in the same index. BB^T is then a diagonal matrix with 1s at non-high confidence indices.
        BBT = torch.diag(
            torch.tensor([int(i not in measured_indices) for i in range(A.shape[0])], dtype=dtype,
                         device=device))
        # D = torch.matmul(self.A, torch.transpose(self.A, 1, 0)) + torch.matmul(self.B, torch.transpose(self.B, 1, 0))
        D = torch.matmul(A, torch.transpose(A,1, 0)) + BBT
        del BBT
        # self.D = D
        # Compute pseudo-inverse for D
        # print fraction of nan/inf values in D
        d_inf_frac = np.count_nonzero(~np.isfinite(D.cpu().numpy())) / (D.shape[0] * D.shape[1])
        assert d_inf_frac == 0
        # print("D inf/nan fration: {}".format(d_inf_frac))
        D[torch.abs(D) < acond] = 0

        # I was getting too many "svd did not converge" errors, not sure what conditions lead to this, it's not very predictable. Seems like using lstsq instead of pinv I'm not getting these errors.
        # print("D mean|.|:", abs(D.cpu()).mean().mean())
        # print("D min|.|, max|.|:", abs(D.cpu()).min().min(), abs(D.cpu()).max().max())
        # D_pseudoinv = torch.linalg.lstsq(D, torch.eye(*D.shape, dtype=dtype, device=device), rcond=rcond)[0].to(
        #     device).to(dtype)
        D_pseudoinv = torch.linalg.pinv(D, rtol=rcond, hermitian=True).to(device).to(dtype)
        del D

        # D_pseudoinv = torch.tensor(scipy.linalg.pinv(D.cpu(), atol=acond, rtol=rcond), device=device, dtype=dtype)
        # self.D_pseudoinv = D_pseudoinv

        # Projection is then to the space c + col(C), where c = AA^TD+b and C=[BB^TD+A:AA^TD+B]
        C = torch.cat((B @ torch.transpose(B, 1, 0) @ D_pseudoinv @ A,
                       A @ torch.transpose(A, 1, 0) @ D_pseudoinv @ B), dim=1)
        C_shape = C.shape
        C[torch.abs(C) < acond] = 0

        AAtDpinv = A @ torch.transpose(A, 0, 1) @ D_pseudoinv

        del D_pseudoinv

        if len(unknown_indices) > 0:
            restriction_matrix = torch.zeros((stoichiometric_matrix.shape[1] - len(unknown_indices),
                                                 stoichiometric_matrix.shape[1]), dtype=dtype, device=device)
            j = 0
            for i in range(stoichiometric_matrix.shape[1] - len(unknown_indices)):
                while j in unknown_indices:
                    j += 1
                restriction_matrix[i, j] = 1
                j += 1
            P = restriction_matrix

            # The first projection - to the restricted intersection space
            # v -> P(AA^TD^+ + PC(PC)^+[P - PAA^TD^+])v.
            # Define T = PAA^TD^+ + PC(PC)^+[I - PAA^TD^+])
            # PAAtDpinv = P @ AAtDpinv

            # PCPCpinv_minus = P @ self.C @ torch.linalg.lstsq(
            #     P @ self.C, P - PAAtDpinv).solution
            # self.T = PAAtDpinv + PCPCpinv_minus
            #
            # projection_matrix = AAtDpinv + self.C @ torch.linalg.lstsq(
            #     P @ self.C, self.T - PAAtDpinv).solution


            # E = self.C @ torch.linalg.pinv(P @ self.C, atol=acond, rtol=rcond)
            # EP = E @ P
            #
            # projection_matrix = EP + (torch.eye(self.C.shape[0], dtype=dtype, device=device) - EP) @ AAtDpinv

            # E = self.C @ torch.linalg.pinv(P @ self.C, atol=acond, rtol=rcond)
            E = C @ torch.linalg.lstsq(P @ C, P, rcond=rcond)[0]
            del C
            projection_matrix = E + (torch.eye(C_shape[0], dtype=dtype, device=device) - E) @ AAtDpinv

            # restricted_projection_matrix = EP + (torch.eye(self.C.shape[0], dtype=dtype, device=device) - EP) @ AAtDpinv
            # # We computed L such that L (Pv) = Py, for the y we're seeking. Then y is (P^+ * L * P)v, but
            # # P is orthogonal, so P^+ can be replaced with P^T.
            # projection_matrix = restricted_projection_matrix @ P
            #

            self.register_buffer('projection_matrix', projection_matrix, persistent=True)

            self.name = "FBAproFull"
            return

        # projection_matrix = self.C @ torch.linalg.lstsq(C, (torch.eye(
        #     self.C.shape[0], dtype=dtype, device=device) - AAtDpinv))[0] + AAtDpinv
        # projection_matrix = AAtDpinv + C @ torch.linalg.pinv(C, rtol=rcond) @ (torch.eye(
        #     C.shape[0], dtype=dtype, device=device) - AAtDpinv)

                
        E = torch.linalg.lstsq(C, (torch.eye(C.shape[0], dtype=dtype, device=device) - AAtDpinv), rcond=rcond)[0].to(device=device)
        Q = C @ E
        del C
        del E
        projection_matrix = AAtDpinv + Q
        self.register_buffer('projection_matrix', projection_matrix, persistent=True)
        self.name = "FBAproFixed"

    def forward(self, x, l_bounds=None, u_bounds=None):
        # x is samples X reactions, transpose for operations here
        x = torch.transpose(x, 0, 1)
        # x here is b in the projection to (a+col(A)) \cap (b + col(B))
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=self.projection_matrix.dtype)
        x = x.to(self.stoic.device)
        nan_frac = x.isnan().float().mean().mean()
        if nan_frac > 0:
            print("(FbaPro) nan frac in x:{:.2f}".format(nan_frac))
            if nan_frac == 1:
                raise ValueError("All nans in x")
            print("Converting nans to zeros")
            x = x.nan_to_num(0.0)

        # Check that intersection exists first (x = DDpinvx)
        if not self.have_warned_intersection and hasattr(self, 'DDpinv') and not torch.allclose(x, self.DDpinv @ x):
            print("Warning: Not sure that intersection exists")
            # print(x, "")
            # print absolute values, min, max, and mean
            print("input x - abs min: {:.2e}, abs max: {:.2e}, abs mean: {:.2e}".format(x.abs().min(), x.abs().max(),
                                                                                        x.abs().mean()))
            mul = self.DDpinv @ x
            # print(mul)
            print("DDpinv * x - abs min: {:.2e}, abs max: {:.2e}, abs mean: {:.2e}".format(mul.abs().min(),
                                                                                           mul.abs().max(),
                                                                                           mul.abs().mean()))
            print("Suppressing further intersection warnings.")
            self.have_warned_intersection = True

        res = self.projection_matrix @ x

        if not self.have_warned_projection and not torch.allclose(self.stoic @ res,
                                                                  torch.zeros((self.stoic.shape[0], 1),
                                                                              dtype=self.projection_matrix.dtype,
                                                                              device=self.stoic.device)):
            print("Warning: projection failed to project to steady state. Accepting, but here are magnitudes:")
            # print(res, "")
            # print absolute values, min, max, and mean

            print("Input - abs min: {:.2e}, abs max: {:.2e}, abs mean: {:.2e}".format(x.abs().min(), x.abs().max(),
                                                                                      x.abs().mean()))

            print("output - abs min: {:.2e}, abs max: {:.2e}, abs mean: {:.2e}".format(res.abs().min(), res.abs().max(),
                                                                                       res.abs().mean()))

            mul = self.stoic @ res
            # print(mul)
            print("S * output - abs min: {:.2e}, abs max: {:.2e}, abs mean: {:.2e}".format(mul.abs().min(),
                                                                                           mul.abs().max(),
                                                                                           mul.abs().mean()))
            print("Suppressing further warnings.")
            self.have_warned_projection = True
        # transpose back
        res = torch.transpose(res, 0, 1)
        return res

    def __repr__(self):
        if self is not None:
            return self.name
        else:
            return "FBApro(uninitialized)"

    def __str__(self):
        return self.__repr__()

class SoftClip(nn.Module):
    # see https://ccrma.stanford.edu/~jos/pasp/Soft_Clipping.html
    def __init__(self, l_bounds, u_bounds, device=torch.device('cuda'), dtype=torch.float64, slope=0.01,
                 input_mix_fraction=0.8):
        super().__init__()
        if type(l_bounds) != torch.Tensor:
            l_bounds = torch.tensor(l_bounds, dtype=dtype, device=device)
        if type(u_bounds) != torch.Tensor:
            u_bounds = torch.tensor(u_bounds, dtype=dtype, device=device)
        self.register_buffer('l_bounds', l_bounds, persistent=True)
        self.register_buffer('u_bounds', u_bounds, persistent=True)
        if len(l_bounds.shape) == 1:
            # assert (l_bounds != u_bounds).all()
            assert (l_bounds <= u_bounds).all()
        else:
            assert (l_bounds <= u_bounds).all().all()
        self.register_buffer('slope', torch.tensor(slope, dtype=dtype, device=device), persistent=True)
        self.input_mix_fraction = input_mix_fraction

    def forward(self, x):
        orig_nan_frac = x.isnan().float().mean().mean()
        if orig_nan_frac > 0:
            print("(SoftClip) nan frac in input:{:.2f}".format(orig_nan_frac))
            if orig_nan_frac == 1:
                raise ValueError("All nans in input")

        assert (x.shape == self.l_bounds.shape) or (x[0].shape == self.l_bounds.shape)
        if x[0].shape == self.l_bounds.shape:
            # l_bounds and u_bounds correspond to reactions. Duplicate them for each sample to get the same shape as x
            l_bounds = torch.tile(self.l_bounds, (x.shape[0], 1))
            u_bounds = torch.tile(self.u_bounds, (x.shape[0], 1))
        else:
            l_bounds = self.l_bounds
            u_bounds = self.u_bounds
        assert (x.shape == l_bounds.shape)

        # shift and scale inputs so that the clipping is to -1, 1
        denom = (u_bounds - l_bounds)
        # where denom != 0, do the shift.
        new_x = torch.where(denom != 0, 2 * (x - l_bounds) / denom - 1, x)
        # where denom == 0, x = l_bounds
        new_x = torch.where(denom == 0, l_bounds, new_x)

        # now consider three cases: x < -1, -1 <= x <= 1, x > 1
        # for x < -1, almost clip to -1 (with slope)
        # for 0 <= x <= 1, use (x - x^3/3) * 3/2
        # for x > 1, almost clip to 1 (with slope)
        new_x = torch.where(new_x < -1, -1 + self.slope * (new_x + 1), new_x)
        new_x = torch.where(new_x > 1, 1 + self.slope * (new_x - 1), new_x)
        new_x = torch.where((-1 <= new_x) & (new_x <= 1), (new_x - new_x ** 3 / 3) * 3 / 2, new_x)
        # now scale back
        new_x = (new_x + 1) / 2 * (u_bounds - l_bounds) + l_bounds

        output = self.input_mix_fraction * x + (1 - self.input_mix_fraction) * new_x

        nan_frac = output.isnan().float().mean().mean()
        if nan_frac > 0:
            print("(SoftClip) nan frac in result:{:.2f}".format(nan_frac))
            if nan_frac == 1:
                raise ValueError("All nans in result")
            print("Converting nans to zeros")
            output = output.nan_to_num(0.0)

        return output

    def __repr__(self):
        return "SoftClip"

    def __str__(self):
        return self.__repr__()

class BoundScale(nn.Module):
    # see https://ccrma.stanford.edu/~jos/pasp/Soft_Clipping.html
    def __init__(self, l_bounds, u_bounds, device=torch.device('cuda'), dtype=torch.float64, zero_sign_mismatches=True,
                 input_mix_fraction=0.8):
        super().__init__()
        if type(l_bounds) != torch.Tensor:
            l_bounds = torch.tensor(l_bounds, dtype=dtype, device=device)
        if type(u_bounds) != torch.Tensor:
            u_bounds = torch.tensor(u_bounds, dtype=dtype, device=device)
        self.register_buffer('l_bounds', l_bounds, persistent=True)
        self.register_buffer('u_bounds', u_bounds, persistent=True)
        if len(l_bounds.shape) == 1:
            # assert (l_bounds != u_bounds).all()
            assert (l_bounds <= u_bounds).all()
        else:
            assert (l_bounds <= u_bounds).all().all()
        self.zero_sign_mismatches = zero_sign_mismatches
        self.input_mix_power = input_mix_fraction

    def forward(self, x):
        assert (x.shape == self.l_bounds.shape) or (x[0].shape == self.l_bounds.shape)
        if x[0].shape == self.l_bounds.shape:
            # l_bounds and u_bounds correspond to reactions. Duplicate them for each sample to get the same shape as x
            l_bounds = torch.tile(self.l_bounds, (x.shape[0], 1))
            u_bounds = torch.tile(self.u_bounds, (x.shape[0], 1))
        else:
            l_bounds = self.l_bounds
            u_bounds = self.u_bounds
        assert (x.shape == l_bounds.shape)

        # Get deviation from bounds. Here we only consider the upper bound if the value is positive, and lower
        # if it's negative.
        # first find sign mismatches where this won't be applied: (x > 0 and u_bounds <= 0) or (x < 0 and l_bounds >= 0)
        sign_mismatch = ((x > 0) & (u_bounds <= 0)) | ((x < 0) & (l_bounds >= 0))
        relevant_bounds = torch.abs(torch.where(x >= 0, u_bounds, l_bounds))
        abs_x = torch.abs(x)
        scaler = relevant_bounds / abs_x
        # take the minimum of scaler, but only where abs_x !=0, relevant_bounds != 0 and not sign_mismatch
        scaler = torch.where(abs_x != 0, scaler, torch.ones_like(scaler))
        scaler = torch.where(relevant_bounds != 0, scaler, torch.ones_like(scaler))
        scaler = torch.where(~sign_mismatch, scaler, torch.ones_like(scaler))
        # now take the minimum per sample
        scaler = torch.min(scaler, dim=1)
        # take an additional minimum with 1 (don't scale up samples)
        scaler = torch.min(scaler.values, torch.ones_like(scaler.values))

        # now multiply each sample by the corresponding scaler
        scaled_x = x * scaler[:, None]

        if self.zero_sign_mismatches:
            # wherever there's a sign mismatch, set to the corresponding bounds (l_bound if x < 0, u_bound if x >= 0)
            scaled_x = torch.where(sign_mismatch,
                                   torch.where(x >= 0, u_bounds, l_bounds),
                                   scaled_x)

        output = self.input_mix_power * x + (1 - self.input_mix_power) * scaled_x

        nan_frac = output.isnan().float().mean().mean()
        if nan_frac > 0:
            print("(Boundscale) nan frac in output:{:.2f}".format(nan_frac))
            if nan_frac == 1:
                raise ValueError("All nans in x")
            print("Converting nans to zeros")
            output = output.nan_to_num(0.0)

        return output

    def __repr__(self):
        return "BoundScale"

    def __str__(self):
        return self.__repr__()

class IterativeProjectionClip(nn.Module):
    def __init__(self, projection_class, n_iters, l_bounds, u_bounds, dtype=torch.float64, device=torch.device('cuda'),
                 slope=0.01, input_mix_fraction=0.5, projection_as_last=True, *params, **kwargs):
        super().__init__()
        assert l_bounds is not None
        assert u_bounds is not None
        # assumes the projection isn't learnable, so no need to create independent instances (very memory heavy).
        projection_instance = projection_class(l_bounds=l_bounds, u_bounds=u_bounds, device=device, dtype=dtype,
                                               *params, **kwargs)
        clip = SoftClip(l_bounds, u_bounds, device=device, dtype=dtype,
                        slope=slope, input_mix_fraction=input_mix_fraction)
        if projection_as_last:
            sequence = [projection_instance] + [clip, projection_instance] * n_iters
        else:
            sequence = [projection_instance] + [clip, projection_instance] * n_iters + [clip]
        self.steps = nn.Sequential(*sequence)
        self.projection_class = projection_class

    def forward(self, x, *args, **kwargs):
        return self.steps(x)

    def __repr__(self):
        return "{}Clip".format(repr(self.steps[0]))

    def __str__(self):
        return self.__repr__()

class IterativeProjectionScale(nn.Module):
    def __init__(self, projection_class, n_iters, l_bounds, u_bounds, dtype=torch.float64, device=torch.device('cuda'),
                 zero_sign_mismatches=True, input_mix_fraction=0.8, projection_as_last=True, *params, **kwargs):
        super().__init__()
        sequence = []
        assert l_bounds is not None
        assert u_bounds is not None
        # assumes the projection isn't learnable, so no need to create independent instances (very memory heavy).
        projection_instance = projection_class(l_bounds=l_bounds, u_bounds=u_bounds, device=device, dtype=dtype,
                                               *params, **kwargs)
        scale = BoundScale(l_bounds, u_bounds, device=device, dtype=dtype,
                           zero_sign_mismatches=zero_sign_mismatches, input_mix_fraction=input_mix_fraction)
        if projection_as_last:
            sequence = [projection_instance] + [scale, projection_instance] * n_iters
        else:
            sequence = [projection_instance] + [scale, projection_instance] * n_iters + [scale]
        self.steps = nn.Sequential(*sequence)
        self.projection_class = projection_class

    def forward(self, x, *args, **kwargs):
        return self.steps(x)

    def __repr__(self):
        return "{}Scale".format(repr(self.steps[0]))

    def __str__(self):
        return self.__repr__()


class FBAWrapper(object):
    def __init__(self, model, dtype=torch.float64, **kwargs):
        self.model = model
        self.dtype = dtype

    def forward(self, x=None, l_bounds=None, u_bounds=None):
        res = torch.zeros(size=x.shape, dtype=self.dtype, device=x.device)
        if l_bounds is not None:
            assert len(l_bounds.shape) == 2
        for sample in range(x.shape[0]):
            with self.model:
                if l_bounds is not None:
                    for i, r in enumerate(self.model.reactions):
                        r.bounds = (l_bounds[sample, i], u_bounds[sample, i])
                sol = self.model.optimize(objective_sense='maximize')
                if sol.status != 'optimal':
                    print("Warning: FBA failed to converge. Solver status: {}".format(sol.status))
                res[sample, :] = torch.tensor([sol.fluxes[r.id] for r in self.model.reactions], dtype=self.dtype)
        return res

    def __repr__(self):
        return "FBA"

    def __str__(self):
        return self.__repr__()

class RawInputWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, l_bounds=None, u_bounds=None):
        return x

    def __repr__(self):
        return "Reference Values"

    def __str__(self):
        return self.__repr__()

class MoMAWrapper(object):
    def __init__(self, model, objective_id, dtype=torch.float64, device=torch.device("cpu"), **kwargs):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.objective_id = objective_id

    def forward(self, x=None, l_bounds=None, u_bounds=None):
        res = torch.zeros(size=x.shape, dtype=self.dtype, device=x.device)
        for sample in range(x.shape[0]):
            with self.model:
                # throws gurobi error about retrieving X if I change model bounds,
                # I assume it's not just using the solution values given
                # as input but also/instead querying the solver directly?
                # I tried debugging and reading docs on this but no success yet.

                # if l_bounds is not None:
                #     for i, r in enumerate(self.model.reactions):
                #         r.bounds = (l_bounds[sample, i], u_bounds[sample, i])
                sample_series = pd.Series(x.cpu()[sample])
                sample_series.index = [r.id for r in self.model.reactions]

                ref_solution = cobra.Solution(
                    objective_value=sample_series.loc[self.objective_id],
                    status=None,
                    fluxes=sample_series,
                    reduced_costs=None,
                    shadow_prices=None
                )
                output_solution = cobra.flux_analysis.moma(model=self.model, solution=ref_solution, linear=False)
                output_solution = torch.tensor([output_solution.fluxes.loc[r.id] for r in self.model.reactions],
                                               dtype=self.dtype, device=self.device)
                res[sample, :] = output_solution

        return res

    def __repr__(self):
        return "MoMA"

    def __str__(self):
        return self.__repr__()

class IMATWrapper(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.S = kwargs['stoichiometric_matrix']

    def forward(self, x=None, l_bounds=None, u_bounds=None):
        device = x.device
        dtype = x.dtype
        # rank-normalize every row of x's absolute values (to create thresholds of activity)
        x = scipy.stats.rankdata(abs(x).cpu().numpy(), axis=1)
        # transform to [0, 1] ranks
        x = (x - 1) / (x.shape[1] - 1)
        # assert rows are indeed ranked
        for i in range(x.shape[0]):
            if len(np.unique(x)) < x.shape[1] / 3:
                print("Warning: for sample {}, less than 1/3 of values are unique ({:.2f})".format(i,
                                                                                                   len(np.unique(x)) /
                                                                                                   x.shape[1]))
            # shouldn't actually be all ranks, because of possible repeated values.
            # assert torch.all(x[i, torch.argsort(x[i])] ==
            #                  torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device))

        res = torch.zeros(size=x.shape, dtype=dtype, device=device)
        if l_bounds is not None:
            assert len(l_bounds.shape) == 2
            if isinstance(l_bounds, torch.Tensor):
                l_bounds = l_bounds.to('cpu').numpy()
                u_bounds = u_bounds.to('cpu').numpy()
        with self.model:
            for sample in range(x.shape[0]):
                print("Sample {}".format(sample))
                properties = imat.IMATProperties(exp_vector=x[sample], exp_thresholds=[0.3, 0.7])
                instance = imat.IMAT(S=self.S, lb=l_bounds[sample], ub=u_bounds[sample], properties=properties)
                # that's troppo's run_imat() code, reproduced here so that I can control the underlying optimization problem and set a timeout (I failed with event handlers)
                exp_vector = instance.properties['exp_vector']
                exp_lb, exp_ub = instance.properties['exp_thresholds']
                core = instance.properties['core']
                epsilon = instance.properties['epsilon']

                high_idx = (np.where(exp_vector >= exp_ub)[0]).astype(int)
                low_idx = (np.where((exp_vector >= 0) & (exp_vector < exp_lb))[0]).astype(int)

                if core:
                    high_idx = np.union1d(high_idx, np.array(core))

                lso, lsystem = instance.generate_imat_problem(instance.S, instance.lb, instance.ub, high_idx, low_idx,
                                                              epsilon)
                backend_model = lso.model
                # print(backend_model.__dict__['configuration'])
                # print()
                # print()
                # print(backend_model.__dict__['problem'])
                backend_model.problem.setParam('TimeLimit', 3 * 60)

                sol = lso.optimize()

                res[sample, :] = torch.tensor(
                    [sol._Solution__value_map["V{}".format(i)] for i in range(len(self.model.reactions))],
                    dtype=dtype, device=device)
        return res

    def __repr__(self):
        return "iMAT"

    def __str__(self):
        return self.__repr__()

class GIMMEWrapper(object):
    def __init__(self, model, objective_id, **kwargs):
        self.model = model
        self.S = kwargs['stoichiometric_matrix']
        self.objective_id = objective_id

    def forward(self, x=None, l_bounds=None, u_bounds=None):
        device = x.device
        dtype = x.dtype
        # rank-normalize every row of x (because gimme needs a single threshold for which reactions to minimize).
        x = scipy.stats.rankdata(abs(x).cpu().numpy(), axis=1)
        # transform to [0, 1] ranks
        x = (x - 1) / (x.shape[1] - 1)
        # assert rows are indeed ranked
        for i in range(x.shape[0]):
            if len(np.unique(x)) < x.shape[1] / 3:
                print("Warning: for sample {}, less than 1/3 of values are unique ({:.2f})".format(i,
                                                                                                   len(np.unique(x)) /
                                                                                                   x.shape[1]))
            # assert torch.all(x[i, torch.argsort(x[i])] ==
            #                  torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device))

        res = torch.zeros(size=x.shape, dtype=dtype, device=device)
        if l_bounds is not None:
            assert len(l_bounds.shape) == 2
            # if l_bounds and u_bounds aren't torch, make them
            if not isinstance(l_bounds, torch.Tensor):
                l_bounds = torch.tensor(l_bounds, dtype=dtype, device=device)
                u_bounds = torch.tensor(u_bounds, dtype=dtype, device=device)
            l_bounds = l_bounds.to('cpu').numpy()
            u_bounds = u_bounds.to('cpu').numpy()
        with self.model:
            objective = {self.model.reactions.index(self.model.reactions.get_by_id(self.objective_id)): 1}
            for sample in range(x.shape[0]):
                print("Sample {}".format(sample))
                properties = gimme.GIMMEProperties(exp_vector=x[sample], objectives=[objective], obj_frac=0.9,
                                                   preprocess=False, flux_threshold=0.7)
                instance = gimme.GIMME(S=self.S, lb=l_bounds[sample], ub=u_bounds[sample], properties=properties)

                sol = instance.run()
                # print(sol)
                # print(type(sol))
                # print(sol.shape)
                # print(np.unique(sol))
                # print(np.mean(sol))
                # returns a list of active reactions, translate to binary vector
                res[sample, :] = torch.tensor([int(r.id in sol) for r in self.model.reactions], dtype=dtype,
                                              device=device)
        return res

    def __repr__(self):
        return "GIMME"

    def __str__(self):
        return self.__repr__()

class RestrictionModule(torch.nn.Module):
    def __init__(self, indices):
        super(RestrictionModule, self).__init__()
        self.indices = indices

    def forward(self, x):
        return x[:, self.indices]