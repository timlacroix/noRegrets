import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from losses import *
from arms import *
from learners import *
from graphs import *


def do_run(learner, graph, losses, horizon=100, **kwargs):
    regrets = np.zeros(horizon)
    learner.start(**kwargs)
    for t in range(horizon):
        current_losses = losses.getLosses()
        arm = learner.getArm(t)
        observed = graph.getObserved(arm)
        learner.observe(observed, current_losses, t)
        regrets[t] = current_losses[arm]-losses.minExpected
    if hasattr(learner, 'Mocc'):
        dupl_r_estimate = 1/np.mean(np.array(learner.Mocc[1:]))
    else:
        dupl_r_estimate = 0
    return regrets


def applyLearner(do_run_function, learner, graph, losses, horizon=100,
                 repeat=4, n_jobs=4, **kwargs):
    regrets = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(do_run_function)(learner, graph, losses, horizon, **kwargs)
        for i in range(repeat)
    )
    # outputs = np.array(outputs)
    # regrets, dupl_r_estimate = zip(*outputs)
    return np.mean(np.cumsum(regrets, axis=1), axis=0)


def estimateR(graph, T, losses):
    k = int(np.e * np.log(T) / 2) + 1
    C = int(2 * np.log(T) / graph.arms) + 1
    j = 0
    c = 0
    M = np.zeros(k)
    regret = []
    for t in range(C):
        I = randint(graph.arms)
        current_losses = losses.getLosses()
        regret.append(current_losses[I]-losses.minExpected)
        Obs = graph.getObserved(I)
        c += np.sum(Obs) - 1
    if c/(C*(graph.arms - 1)) <= 3/(2*graph.arms):
        return 0, t+1, regret
    else:
        for t in range(C, T):
            I = randint(graph.arms)
            current_losses = losses.getLosses()
            regret.append(current_losses[I]-losses.minExpected)
            Obs = graph.getObserved(I)
            for i in range(graph.arms):
                M[j] += int(i != I)
                j += Obs[i] * int(i != I)
                if j == k:
                    return 1/(np.max(M) + 1), t+1, regret


def do_run_dupl(learner, graph, losses, horizon=100):
    r_, it, regret = estimateR(graph, horizon, losses)
    regrets = np.zeros(horizon)
    regrets[0:it] = regret
    if r_ < 1e-12:  # we use vanilla EXP3
        eta = np.sqrt(2 * np.log(graph.arms) / (horizon * graph.arms))
        learner = EXP3(gamma=0.01, eta=eta, arms=graph.arms)
        learner.start()
        regrets[it:horizon] = do_run(learner, graph, losses,
                                                      horizon=horizon-it)
    else:
        A = int(np.log(horizon-it)/(graph.arms * r_)) + 1
        learner = GeneralDuplExp3(arms=graph.arms, A=A)
        learner.start()
        regrets[it:horizon] = do_run(learner, graph, losses,
                                                      horizon=horizon-it)
    return regrets


def update_online(old_mean, new_value, n, old_M2):
    delta = new_value - old_mean
    new_mean = old_mean + delta / n
    new_M2 = old_M2 + delta**2
    return new_mean, new_M2


# n_arms = 30
# eps = 0.1
# n_iterations = 1000
# r = 0.5
# rep = 100

# losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])
# graph = BAGraph(arms=n_arms, m=3, m0=3, r=0.3)
# learner = BAEXP3(gamma=0.01, eta=0.01, arms=n_arms)
# global_reg = np.zeros(n_iterations)
# global_r_mean = 0
# global_M2 = 0

# for i in range(5):
#     reg, r_mean = applyLearner(do_run_dupl, learner, graph, losses,
#                                horizon=n_iterations, repeat=25, n_jobs=1)
#     global_reg, useless = update_online(global_reg, reg, i+1, global_reg)
#     global_r_mean, global_M2 = update_online(global_r_mean, r_mean, i+1, global_M2)
#     print(global_r_mean, np.sqrt(global_M2/((i+1)*n_iterations)))   



# possible_arms = [50, 100, 1000]
# possible_r = [0.3, 0.5, 0.7, 0.9]
# for n_arms in possible_arms:
#     for r in possible_r:
#         #n_arms = 50
#         eps = 0.1
#         n_iterations = 10000
#         #r = 0.5
#         rep = 100

#         losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

#         graph = ERGraph(arms=n_arms, r=r)
#         learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
#         rr = applyLearner(do_run_dupl, learner, graph, losses,
#                           horizon=n_iterations, repeat=rep, n_jobs=1)

#         regrets = applyLearner(
#             do_run, learner, graph, losses,
#             horizon=n_iterations, repeat=rep, n_jobs=1
#         )
#         plt.figure(2)
#         plt.plot(regrets, 'r-', label='EXP3-ER', linewidth=2)

#         plt.plot(rr, 'b-', label='DuplEXP3-ER', linewidth=2)

#         plt.legend(loc=2, fontsize=20)
#         plt.xlabel('iterations', fontsize=20)
#         plt.ylabel('cumulated regret', fontsize=20)
#         plt.savefig(str(n_arms) + 'new_dupl_big_r' + str(r) + '.pdf')
#         plt.close()

## BA !
n_arms = 50
graph = BAGraph(arms=n_arms, m=7, m0=7, r=0.3)

learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
rr = applyLearner(do_run_dupl, learner, graph, losses,
                  horizon=n_iterations, repeat=rep, n_jobs=1)

regrets = applyLearner(
    do_run, learner, graph, losses,
    horizon=n_iterations, repeat=rep, n_jobs=1
)
learner = BAEXP3(gamma=0.0, eta=0.01, arms=n_arms)
other_regret = applyLearner(
    do_run, learner, graph, losses,
    horizon=n_iterations, repeat=rep, n_jobs=1
)
plt.figure(2)
plt.plot(regrets, 'r-', label='EXP3-BA', linewidth=2)

plt.plot(rr, 'b-', label='DuplEXP3-BA', linewidth=2)
plt.plot(other_regret, '-', label='BAEXP3-BA', linewidth=2, color='purple')

plt.legend(loc=2, fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('cumulated regret', fontsize=20)
plt.savefig(str(n_arms) + 'new_BAdupl_big_r' + str(r) + '.pdf')
plt.close()

