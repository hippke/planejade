import numpy as np
from numba import jit, prange
import time as ttime
from tqdm import tqdm
import random
import re
import matplotlib.pyplot as plt

MAX_INT = 4294967295
CONST = 4 * np.exp(-0.5) / np.sqrt(2)


def int_commas(x):
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)


# Random numbers from getrandbits are 10x faster than single numpy calls
@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def rand_int(limit=None):
    r = random.getrandbits(32)
    if limit is not None:
        r = (r / MAX_INT) * limit
    return int(r)


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def rand_float():
    return rand_int() / MAX_INT


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def rand_cauchy(mu=0, sigma=1):
    return mu + sigma * np.tan(np.pi * (rand_normal() - 0.5))


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def rand_normal(mu=0, sigma=1):
    while True:
        u2 = 1 - rand_float()
        z = CONST * (rand_float() - 0.5) / u2
        if z * z / 4 <= -np.log(u2):
            break
    return mu + z * sigma


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def clip(min_, max_, val):
    return max(min_, min(max_, val))


@jit(cache=False, nopython=True, fastmath=True, parallel=True)
def eval_func(pop, fit, fitness):
    for idx in prange(pop.shape[0]):
        fit[idx] = fitness(pop[idx])
    return fit


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def generate_trial_func(pop, trial, limits, n_pop, n_dim, p, cr, f, cr_arr, f_arr, fit):
    k = int(p * n_pop)
    k_high = np.argsort(fit)[:-k-1:-1]
    pbest_i = k_high[rand_int(k)]
    for i in range(n_pop):
        r1 = rand_int(n_pop)
        r2 = rand_int(n_pop)
        while r1 == i: 
            r1 = rand_int(n_pop)
        while r2 == r1 or r2 == i: 
            r2 = rand_int(n_pop)
        always = rand_int(n_dim)
        cr_i = clip(0, 1, rand_normal(cr, 0.1))
        f_i = clip(0, 1, rand_cauchy(f, 0.1))
        cr_arr[i] = cr_i
        f_arr[i] = f_i
        for d in range(n_dim):
            if rand_normal() < cr_i or d == always:
                v = (pop[i, d] +
                       f_i*(pop[pbest_i, d] - pop[i, d]) +
                       f_i*(pop[r1, d] - pop[r2, d]))
            else:
                v = pop[i, d]
            trial[i, d] = clip(limits[0][d, 0], limits[0][d, 1], v)


class JADE:

    def __init__(self, fitness, n_dim, n_pop, limits, batch_id, n_it, c=0.01):
        self.batch_id = batch_id
        self.counter = 0
        self.n_pop = n_pop

        # JADE parameters
        self.p = 0.1
        self.c = c  # 0.01 for moons, 0.1 for planets??

        # current means (u_cr and u_f in the literature)
        self.cr = 0.5
        self.f = 0.5
        self.cr_arr = np.empty(n_pop)
        self.f_arr = np.empty(n_pop)

        self.fitness = fitness
        self.f = 0.7
        self.cr = 0.95
        limits = [limits] * n_dim
        self.limits = np.array(limits)
        n_dim = len(self.limits)
        self.pop = np.empty((n_pop, n_dim))
        self.trial = np.empty((n_pop, n_dim))
        for dim in range(n_dim):
            lower, upper = limits[0][dim]
            range_ = upper - lower
            for ind in range(n_pop):
                self.pop[ind, dim] = np.random.random_sample() * range_ + lower
        self.fit = np.zeros(n_pop)
        self.trial_fit = np.zeros(n_pop)
        eval_func(self.pop, self.fit, self.fitness)


    def selection(self):
        """Merges the best trial vectors to the main population."""
        pop = self.pop
        trial = self.trial
        fit = self.fit
        trial_fit = self.trial_fit
        cr_arr = self.cr_arr
        f_arr = self.f_arr
        num_cr = 0.0
        den_cr = 0.0
        num_f = 0.0
        den_f = 0.0
        for i in range(self.pop.shape[0]):
            if trial_fit[i] > fit[i]:
                fit[i] = trial_fit[i]
                pop[i] = trial[i]
                num_cr += cr_arr[i]
                den_cr += 1.0
                num_f += f_arr[i]**2
                den_f += f_arr[i]
        if den_cr > 0:
            self.cr = (1.0-self.c)*self.cr + self.c*(num_cr/den_cr)
        if den_f > 0:
            self.f = (1.0-self.c)*self.f + self.c*(num_f/den_f)


    def run(self, n_it=1000):
        verlauf = []
        list_logls = np.zeros((n_it, len(self.pop)))
        list_periods = np.copy(list_logls)
        #print(list_logls)
        #list_logls = []
        #list_periods = []
        self.k_high = None 
        pbar = tqdm(total=n_it)#, position=self.batch_id)
        for iteration in range(n_it):

            #t1 = ttime.perf_counter()
            generate_trial_func(self.pop, self.trial, self.limits, self.pop.shape[0], self.pop.shape[1], self.p, self.cr, self.f, self.cr_arr, self.f_arr, self.fit)
            #t2 = ttime.perf_counter()
            #print(t2-t1, "generate_trial")
            eval_func(self.trial, self.trial_fit, self.fitness)
            #t3 = ttime.perf_counter()
            #print(t3-t2, "evaluate")
            self.selection()
            #t4 = ttime.perf_counter()
            #print(t4-t3, "selection")
            #print("iteration", iteration)
            #print(min(self.fit))
            pbar.update(1)
            ncall = len(self.pop) * (1 + iteration)
            pbar.set_postfix({'ncall': int_commas(ncall), "fit": str(round(min(self.fit), 10))})
            best_in_iter = min(self.fit)
            verlauf.append(best_in_iter)
            no_change_iters = 200  # if no change after 100 iterations, then converged
            look_back = iteration - no_change_iters
            if look_back < 0:
                look_back = 0
            if best_in_iter == verlauf[look_back] and iteration > no_change_iters:
                #print("Stop: No change after 50 iters")
                break

            periods = self.trial[:,0]
            #print("periods", periods)
            #print(self.fit)
            #print("len", len(self.fit))
            list_logls[iteration][:] = self.fit
            list_periods[iteration][:] = periods
            #list_logls.append(self.fit)
            #list_periods.append(periods)
            #print(iteration, self.fit)
            #for idx in range(len(periods)):
            #    print(periods[idx], self.fit[idx])
            #fitnesses = self.fitness()
            #breal

            #besti = np.argmax(self.fit)
            #print(iteration, self.trial[besti][0], self.fitness(self.pop[besti]))
        besti = np.argmax(self.fit)#best_k_func(1, self.k_high, self.fit)
        print(self.c, self.n_pop, self.trial[besti], iteration, self.fitness(self.pop[besti]))#, end='')
        pbar.close()
        
        return self.pop[besti], verlauf, list_logls, list_periods
