# Variable-step BDF2 for the monolithic GIA solver

`SelfGravitatingGIASolver(..., time_scheme="bdf2")` selects prescribed
variable-step BDF2, with backward-Euler startup. The default remains
`time_scheme="backward_euler"`. Both the coupled DG history formulation and
linear pointwise substitution support BDF2. The global displacement, potential,
rotation, sea-level and frame constraints remain coupled at the new time.

```python
# Use a Real-space Function so assignments are checkpointed on the adjoint tape.
dt = Function(FunctionSpace(mechanics_mesh, "R", 0)).assign(first_dt)
solver = SelfGravitatingGIASolver(
    state, approximation, layout=layout, dt=dt,
    time_scheme="bdf2", **existing_gia_options,
)
for h, forcing in prescribed_schedule:
    dt.assign(h)
    load.assign(forcing)
    solver.solve()  # prepares coefficients, solves and accepts material history
```

There is no separate `set_step` call in the integrated API. The solver prepares
coefficients automatically. With $r=h/h_{previous}$,

$$
\alpha=\frac{1+2r}{1+r},\qquad
\beta=1+r,\qquad \gamma=-\frac{r^2}{1+r},
$$

$$
\alpha m^{n+1}-\beta m^n-\gamma m^{n-1}
=h\frac{e(u^{n+1})-m^{n+1}}{\tau(s^{n+1})}.
$$

Only material histories have BDF derivatives; algebraic constraints use their
existing new-time residuals. Coefficients and both accepted history levels are
Functions recorded by pyadjoint. Constant Jacobian mode is refused because
startup and subsequent steps have different tangents. Timestep increases must
satisfy $h/h_{previous}<1+\sqrt2$; all steps must be positive and finite.

At a known load discontinuity, finish at the event, update the load, then call
`solver.restart_time_integration()` before the next solve to request BE startup.
There is no automatic event detection, error controller, step rejection/rollback
or differentiation of a control-dependent timestep controller.

The BDF history Functions are additional checkpoint state. An application-level
restart must save/restore both levels, previous accepted step size and the
current mixed state; the existing BE-only restart file is not sufficient.

## Validation

`tests/unit/test_gia_bdf2.py` compares full two-mesh mixed solutions against
independently constructed BE history problems with time $h/\alpha$ and old
history $(\beta m^n+\gamma m^{n-1})/\alpha$. It also checks unequal-step
reverse adjoints and equality of startup with the existing BE solver.

Codex assisted.
