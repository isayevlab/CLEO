import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


def space_prod(spaces):
    if len(spaces) == 1:
        return spaces[0]
    if len(spaces) == 2:
        results = []
        for dct1 in spaces[0]:
            for dct2 in spaces[1]:
                results.append({**dct1, **dct2})
        return results
    else:
        return space_prod([spaces[0], space_prod(spaces[1:])])


def space_concat(spaces):
    n = len(spaces[0])

    for space in spaces:
        assert len(space) == n

    results = []
    for i in range(n):
        dct = {}
        for j in range(len(spaces)):
            dct = {**dct, **spaces[j][i]}
        results.append(dct)
    return results


def weights_decomposition(K, total_sum=1., step=0.1, weights=None, shuffle=False):
    if weights is None:
        weights = []

    if K == 1:
        if total_sum > 0:
            weights.append(total_sum)
            yield weights

    if total_sum < step*K:
        return

    sum = 0

    possible_steps = []

    while sum < total_sum:
        sum += step
        if total_sum - sum >= step:
            possible_steps.append(sum)

    if shuffle:
        random.shuffle(possible_steps)

    for sum in possible_steps:
        yield from weights_decomposition(K - 1, total_sum - sum, step=step,
                                         weights=[sum] + weights, shuffle=shuffle)


class ComponentsMixture:
    def __init__(self, components_names, n_components=None,
            step=0.1, shuffle=True, total_sum=1.):
        self.total_sum = total_sum
        self.names = components_names
        self.max_num_component = n_components if n_components is not None else len(self.names)
        self.combs, self.weights = self.space_enumeration(step, shuffle)
        self.shuffle = shuffle

    def space_enumeration(self, step, shuffle):
        combs = list(itertools.combinations(self.names, self.max_num_component))
        weights = list(weights_decomposition(self.max_num_component, total_sum=self.total_sum,
                                             step=step, shuffle=shuffle))

        if shuffle:
            random.shuffle(combs)
        return combs, weights

    def sample(self, n):
        idxs = np.random.randint(0, len(self.combs), n)
        combs = [self.combs[i] for i in idxs]
        idxs = np.random.randint(0, len(self.weights), n)
        weights = [self.weights[i] for i in idxs]
        results = []
        for c, w in zip(combs, weights):
            results.append(dict(zip(c, w)))
        return results

    def get_space_size(self):
        return len(self.combs) * len(self.weights)

    def get_space(self):
        results = []

        for c in self.combs:
            for w in self.weights:
                results.append(dict(zip(c, w)))
        return results


class MultipleComponentsMixture:
    def __init__(self, components_names, min_components=None, max_components=None,
            step=0.1, shuffle=True, total_sum=1.):
        self.components = {}

        for i in range(min_components, max_components + 1):
            comp = ComponentsMixture(components_names, n_components=i,
                                     step=step, shuffle=shuffle,
                                     total_sum=total_sum)
            if comp.get_space_size() > 0:
                self.components[i] = comp

    def sample(self, n):
        idxs = list(np.random.choice(list(self.components.keys()), n))
        id2n = Counter(idxs)
        samples = []
        for i in id2n:
            s = self.components[i].sample(id2n[i])
            samples.extend(s)
        return samples

    def get_space_size(self):
        return sum([comp.get_space_size() for comp in self.components.values()])

    def get_space(self):
        results = []

        for comp in self.components.values():
            results.extend(comp.get_space())
        return results


class SuperComponentsMixture:
    def __init__(self, component_dict, n_components=None, step=0.1,
            shuffle=True):
        self.n_components = n_components if n_components is not None else len(component_dict)

        self.names = sorted(list(component_dict.keys()))
        combs = list(itertools.combinations(self.names, self.n_components))
        weights = list(weights_decomposition(self.n_components, total_sum=1.,
                                             step=step, shuffle=shuffle))
        self.components = []

        for comb in combs:
            for weight in weights:
                components = []
                for name, w in zip(comb, weight):
                    components.append(MultipleComponentsMixture(component_dict[name]["components"],
                                                                total_sum=w,
                                                                **component_dict[name]["params"]))
                empty = False
                for comp in components:
                    if comp.get_space_size() == 0:
                        empty = True
                        break
                if not empty:
                    self.components.append(components)

    def sample(self, n):
        k = len(self.components)
        idxs = np.random.choice(np.arange(k), n)

        id2n = dict(Counter(idxs))
        spaces = [[] for _ in range(len(self.names))]
        for id in id2n:
            for i, comp in enumerate(self.components[id]):
                spaces[i].extend(comp.sample(id2n[id]))
        samples = space_concat(spaces)
        return samples

    def get_space_size(self):
        space_size = 0

        for comp in self.components:
            sizes = []
            for i, c in enumerate(comp):
                sizes.append(c.get_space_size())
            space_size += np.prod(sizes)

        return space_size

    def get_space(self):
        res = []
        for comp in self.components:
            tmp_res = []
            for i, c in enumerate(comp):
                tmp_res.append(c.get_space())
            res.extend(space_prod(tmp_res))
        return res


class SpaceGenerator:
    def __init__(self, features_dict, max_space=10 ** 7, save_space=True):
        self.grid_feats, self.mix_feats = self.build_features(features_dict)

        grid_size = None
        mix_size = None
        if len(self.grid_feats) > 0:
            grid_size = np.prod([len(i) for i in self.grid_feats.values()])
        if len(self.mix_feats) > 0:
            mix_size = np.prod([v.get_space_size() for v in self.mix_feats.values()])

        if grid_size is None and mix_size is None:
            self.space_size = 0
        elif grid_size is None:
            self.space_size = mix_size
        elif mix_size is None:
            self.space_size = grid_size
        else:
            self.space_size = grid_size * mix_size

        self.max_space = max_space
        self.save_space = save_space
        if self.save_space:
            self.construct_space()

    def build_features(self, feature_dict):
        grid_features = {}
        mixture_features = {}
        for key, value in feature_dict.items():
            if isinstance(value, list):
                grid_features[key] = value
            elif isinstance(value, dict):
                if "type" in value:
                    if value["type"] == "range":
                        grid_features[key] = np.linspace(*value["params"])
                    elif value["type"] == "mixture":
                        mixture_features[key] = MultipleComponentsMixture(value["components"],
                                                                          **value["params"])
                    elif value["type"] == "super_mixture":
                        mixture_features[key] = SuperComponentsMixture(value["components"],
                                                                       **value["params"])
                else:
                    raise NotImplemented("The dictionary should contain 'type' key")

        return grid_features, mixture_features

    def sample(self, n=1):
        if self.save_space:
            return np.random.choice(self.space, n)
        else:
            grid_space = []
            for _ in range(n):
                sample = {}
                for i, k in self.grid_feats.items():
                    sample[i] = np.random.choice(k)
                grid_space.append(sample)
            spaces = [grid_space]

            for mixture in self.mix_feats.values():
                spaces.append(mixture.sample(n))
            results = space_concat(spaces)
            return results

    def construct_space(self):
        if self.max_space >= self.space_size:
            spaces = []
            if len(self.grid_feats) > 0:
                spaces.append(list(ParameterGrid(self.grid_feats)))

            if len(self.mix_feats) > 0:
                for mix in self.mix_feats.values():
                    spaces.append(mix.get_space())
            space = space_prod(spaces)
            self.space = pd.DataFrame.from_records(space).fillna(0.)

        else:
            save_space = self.save_space
            self.save_space = False
            self.space = pd.DataFrame(self.sample(self.max_space)).fillna(0.)
            self.save_space = save_space

    @classmethod
    def read_yaml(cls, path, **kwargs):
        with open(path) as f:
            my_dict = yaml.safe_load(f)
        return cls(my_dict, **kwargs)


if __name__ == "__main__":
    import yaml

    conf = yaml.safe_load(open("../tmp/space.yaml"))
    space_generator = SpaceGenerator(conf, save_space=True, max_space=1000000)
    print(space_generator.space.head())
    space_generator = SpaceGenerator(conf, save_space=False, max_space=10)
    print(space_generator.sample(10))

