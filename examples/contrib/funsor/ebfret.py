# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This example is largely copied from ``examples/hmm.py``.
It illustrates the use of the experimental ``pyro.contrib.funsor`` Pyro backend
through the ``pyroapi`` package, demonstrating the utility of Funsor [0]
as an intermediate representation for probabilistic programs

This example combines Stochastic Variational Inference (SVI) with a
variable elimination algorithm, where we use enumeration to exactly
marginalize out some variables from the ELBO computation. We might
call the resulting algorithm collapsed SVI or collapsed SGVB (i.e
collapsed Stochastic Gradient Variational Bayes). In the case where
we exactly sum out all the latent variables (as is the case here),
this algorithm reduces to a form of gradient-based Maximum
Likelihood Estimation.

To marginalize out discrete variables ``x`` in Pyro's SVI:

1. Verify that the variable dependency structure in your model
    admits tractable inference, i.e. the dependency graph among
    enumerated variables should have narrow treewidth.
2. Annotate each target each such sample site in the model
    with ``infer={"enumerate": "parallel"}``
3. Ensure your model can handle broadcasting of the sample values
    of those variables
4. Use the ``TraceEnum_ELBO`` loss inside Pyro's ``SVI``.

Note that empirical results for the models defined here can be found in
reference [1]. This paper also includes a description of the "tensor
variable elimination" algorithm that Pyro uses under the hood to
marginalize out discrete latent variables.

References

0. "Functional Tensors for Probabilistic Programming",
Fritz Obermeyer, Eli Bingham, Martin Jankowiak,
Du Phan, Jonathan P Chen. https://arxiv.org/abs/1910.10775

1. "Tensor Variable Elimination for Plated Factor Graphs",
Fritz Obermeyer, Eli Bingham, Martin Jankowiak, Justin Chiu,
Neeraj Pradhan, Alexander Rush, Noah Goodman. https://arxiv.org/abs/1902.03210
"""
import argparse
import functools
import logging
import sys

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.contrib.examples import polyphonic_data_loader as poly
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings

try:
    import pyro.contrib.funsor
except ImportError:
    pass

from pyroapi import distributions as dist
from pyroapi import handlers, infer, optim, pyro, pyro_backend

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
def model(data, num_states, vectorized):
    pyro.param("rho0", torch.ones(num_states), constraint=constraints.positive)
    pyro.param(
        "alpha0", torch.ones(num_states, num_states), constraint=constraints.positive
    )
    pyro.param("a0", torch.ones(num_states) * 10, constraint=constraints.positive)
    pyro.param("b0", torch.ones(num_states), constraint=constraints.positive)
    pyro.param("m0", torch.arange(num_states) / num_states, constraint=constraints.real)
    pyro.param("beta0", torch.ones(num_states), constraint=constraints.positive)

    with pyro.plate("trace", data.shape[1], dim=-1):
        pi = pyro.sample("pi", dist.Dirichlet(pyro.param("rho0")))
        # with pyro.plate("state", num_states, dim=-2):
        A = pyro.sample("A", dist.Dirichlet(pyro.param("alpha0")).to_event(1))
        lamda = pyro.sample(
            "lamda", dist.Gamma(pyro.param("a0"), pyro.param("b0")).to_event(1)
        )
        mu = pyro.sample(
            "mu",
            dist.Normal(
                pyro.param("m0"), 1 / (pyro.param("beta0") * lamda).sqrt()
            ).to_event(1),
        )

        z_prev = None
        markov_loop = (
            pyro.vectorized_markov(name="time", size=data.shape[0], dim=-2)
            if vectorized
            else pyro.markov(range(data.shape[0]))
        )
        for t in markov_loop:
            z_curr = pyro.sample(
                "z_{}".format(t),
                dist.Categorical(
                    pi if isinstance(t, int) and t < 1 else Vindex(A)[..., z_prev, :]
                ),
                infer={"enumerate": "parallel"},
            )
            pyro.sample(
                "x_{}".format(t),
                dist.Normal(
                    Vindex(mu)[..., z_curr], 1 / Vindex(lamda)[..., z_curr].sqrt()
                ),
                obs=data[t],
            )
            z_prev = z_curr


def guide(data, num_states, vectorized):
    data_dim = data.shape[1]
    pyro.param("rho", torch.ones(data_dim, num_states), constraint=constraints.positive)
    pyro.param(
        "alpha",
        torch.ones(data_dim, num_states, num_states),
        constraint=constraints.positive,
    )
    pyro.param(
        "a", torch.ones(data_dim, num_states) * 10, constraint=constraints.positive
    )
    pyro.param("b", torch.ones(data_dim, num_states), constraint=constraints.positive)
    pyro.param(
        "m",
        torch.arange(num_states).repeat(data_dim, 1) / num_states,
        constraint=constraints.real,
    )
    pyro.param(
        "beta", torch.ones(data_dim, num_states), constraint=constraints.positive
    )

    with pyro.plate("trace", data.shape[1], dim=-1):
        pyro.sample("pi", dist.Dirichlet(pyro.param("rho")))
        pyro.sample("A", dist.Dirichlet(pyro.param("alpha")).to_event(1))
        lamda = pyro.sample(
            "lamda", dist.Gamma(pyro.param("a"), pyro.param("b")).to_event(1)
        )
        pyro.sample(
            "mu",
            dist.Normal(
                pyro.param("m"), 1 / (pyro.param("beta") * lamda).sqrt()
            ).to_event(1),
        )


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    global model, guide

    logging.info("Data simulation")
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    pyro.param("rho0", torch.tensor([100, 300, 200]), constraint=constraints.positive)
    pyro.param(
        "alpha0",
        torch.tensor([[20, 10, 30], [15, 25, 10], [30, 15, 20]]),
        constraint=constraints.positive,
    )
    pyro.param(
        "a0", torch.tensor([10000, 10000, 10000]), constraint=constraints.positive
    )
    pyro.param("b0", torch.tensor([1, 1, 1]), constraint=constraints.positive)
    pyro.param("m0", torch.tensor([0.1, 0.5, 0.8]), constraint=constraints.real)
    pyro.param("beta0", torch.tensor([1, 1, 1]), constraint=constraints.positive)
    #  samples = {}
    #  samples["pi"]
    predictive = Predictive(handlers.uncondition(model), num_samples=1)
    data = torch.zeros(100, 32)
    simulation = predictive(data, 3, False)
    for t in range(data.shape[0]):
        data[t, :] = simulation[f"x_{t}"][0].data

    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    # Bind non-PyTorch parameters to make these functions jittable.
    #  model = functools.partial(model, args=args)
    #  guide = functools.partial(guide, args=args)

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optimizer = optim.Adam({"lr": args.learning_rate})
    Elbo = infer.JitTraceEnum_ELBO if args.jit else infer.TraceEnum_ELBO
    max_plate_nesting = 2
    elbo = Elbo(
        max_plate_nesting=max_plate_nesting,
        strict_enumeration_warning=True,
        jit_options={"time_compilation": args.time_compilation},
    )
    svi = infer.SVI(model, guide, optimizer, elbo)

    # We'll train on small minibatches.
    logging.info("Step\tLoss")
    for step in range(args.num_steps):
        loss = svi.step(data, 3, args.vectorized)
        logging.info("{: >5d}\t{}".format(step, loss))

    if args.jit and args.time_compilation:
        logging.debug(
            "time to compile: {} s.".format(elbo._differentiable_loss.compile_time)
        )

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, data, 3, False)
    logging.info("training loss = {}".format(train_loss))


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.6.0")
    parser = argparse.ArgumentParser(
        description="MAP Baum-Welch learning Bach Chorales"
    )
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--vectorized", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--time-compilation", action="store_true")
    parser.add_argument("-rp", "--raftery-parameterization", action="store_true")
    parser.add_argument("--funsor", action="store_true")
    args = parser.parse_args()

    if args.funsor:
        import funsor

        funsor.set_backend("torch")
        PYRO_BACKEND = "contrib.funsor"
    else:
        PYRO_BACKEND = "pyro"

    with pyro_backend(PYRO_BACKEND):
        main(args)
