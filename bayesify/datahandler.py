import os
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT_ATTRIBUTES = ["performance", "energy", "runtime", "run-time", "time"]


class ConfigSysProxy:
    """Utility for loading and querying configuration measurements."""
    def __init__(self, folder, attribute=None, val_set_size=0, val_set_rnd_seed=None):
        """Read configuration data from ``folder`` and prepare it for modeling."""
        self.fm_name = "featuremodel.xml"
        self.measurements_file_name = "measurements.xml"
        self.measurements_file_name_csv = "measurements.csv"
        self.prototype_config = None
        self.position_map = None
        self.folder = folder
        self.attribute = attribute
        self.redundant_ft_names = []
        self.alternative_ft_names = []
        self.position_map = self.parse_fm()
        self.all_configs = self.parse_configs()
        self.redundant_ft, self.redundant_ft_names = self.remove_constant_features()
        (
            self.alternative_ft,
            self.alternative_ft_names,
        ) = self.remove_alternative_features()
        self.store_csv()
        print("Finished reading measurements")
        self.global_opt = None
        self.get_global_opt()

        self.validation_set = None

        if val_set_size:
            print("generating validation set")
            self.validation_set = self.generate_val_set(val_set_size, val_set_rnd_seed)
        else:
            self.validation_set = []

    def get_all_configs(self):
        return self.all_configs

    def get_VIF_for_features(self, x_np=None):
        if x_np is None:
            x_np = np.array(list(self.all_configs.keys()))
        vifs = [variance_inflation_factor(x_np, i) for i in range(x_np.shape[1])]
        return vifs

    def get_corr_eigen_fts(self, x_np=None, eigen_thresh=0.01, return_inner=False):
        if x_np is None:
            x_np = np.array(list(self.all_configs.keys()))
        corr_mat = np.corrcoef(x_np, rowvar=0)
        w, v = np.linalg.eig(corr_mat)
        eigen_ids = np.nonzero(w < eigen_thresh)
        if len(eigen_ids) < 1:
            component_ft_of_eigencevtors, w, v = [None] * 3
        else:
            eigenvectors = v[eigen_ids]
            component_ft_of_eigencevtors = np.nonzero(
                np.abs(eigenvectors > eigen_thresh)
            )
        if return_inner:
            return component_ft_of_eigencevtors, w, v
        else:
            return component_ft_of_eigencevtors

    def may_ft_occur_together(self, ft_names):
        np_cfgs = np.array(list(self.all_configs.keys()))
        masks = []
        for ft in ft_names:
            i = self.position_map[ft]
            mask = np_cfgs[:, i] == 1.0
            masks.append(mask)
        all_occur = np.all(np.array(masks), axis=0)
        num_occur = np.count_nonzero(all_occur)
        r = num_occur > 0
        return r

    def generate_val_set(self, size, seed, exclude_samples=None):
        samples = self.get_random_points(size, seed=seed)
        ys = self.eval(samples)
        v_set = samples, ys
        return v_set

    def get_validation_set(self):
        return self.validation_set

    def get_prototype_config(self):
        return self.prototype_config

    def get_random_point(self):
        conf = self.get_random_points(1)[0]
        return conf

    def get_random_points(self, n, seed=None):
        num_configs = len(self.all_configs.keys())
        size = min(num_configs, n)
        if seed:
            rndg = np.random.RandomState(seed)
            idx = rndg.choice(num_configs, size, replace=False)
        else:
            idx = np.random.choice(num_configs, size, replace=False)
        conf_arr = np.array(list(self.all_configs.keys()))[idx]
        confs = list(tuple(tpl) for tpl in conf_arr)
        return confs

    def get_global_opt(self):
        if self.global_opt is None:
            perfs = list(self.all_configs.values())
            confs = list(self.all_configs.keys())
            best_idx = int(np.argmin(perfs))
            best_perf = perfs[best_idx]
            best_conf = confs[best_idx]
            self.global_opt = best_perf, best_conf
        return self.global_opt

    def get_n_samples(self, n):
        samples = []
        points = self.get_random_points(n)
        for x in points:
            # x = self.get_random_point()
            y = self.eval(x)
            samples.append((x, y))
        return samples

    def eval(self, X):
        if type(X) is list:
            result = [self._eval(x) for x in X]
        else:
            result = self._eval(X)
        return result

    def _eval(self, x):
        x_tuple = self.validate_conf(x)
        perf = self.all_configs[x_tuple]
        return perf

    def eval_precisely(self, x):
        perf = self._eval(x)
        return perf

    def validate_conf(self, x):
        if not len(tuple(x)) == len(self.prototype_config):
            raise ValueError(
                "Argument {} is unlike configuration shape: {} e.g. {}".format(
                    x, np.array(self.prototype_config).shape, self.prototype_config
                )
            )
        if np.isscalar(x) or tuple(x) not in self.all_configs:
            raise ValueError("{} appears not to be a valid configuration".format(x))
        x_tuple = tuple(x)

        return x_tuple

    def parse_fm(self):
        """Parse the feature model and build ``self.position_map``."""
        top_files = os.listdir(self.folder)
        root = None
        for file in top_files:
            if self.fm_name in file.lower():
                tree = ET.parse(os.path.join(self.folder, file))
                root = tree.getroot()
                break

        attribute_names = []
        for element in root.iter("configurationOption"):
            # i.attrib["target"] = "blank"
            name = element.find("name").text
            attribute_names.append(name)
        sorted_features = sorted(attribute_names)
        self.position_map = {}
        i = 0
        for ft in sorted_features:
            if ft != "root":
                self.position_map[ft] = i
                i += 1
        self.update_prototype()
        return self.position_map

    def update_prototype(self):
        self.prototype_config = list(0.0 for i in list(self.position_map.keys()))

    def get_pos_map_from_df(self, df):
        pos = {}
        for i, col in enumerate([col for col in df.columns if col != "<y>"]):
            pos[col] = i
        return pos

    def parse_configs(self):
        """Load configuration measurements from the configured folder."""
        top_files = os.listdir(self.folder)
        performance_map = None
        for file in top_files:
            file_candidate = os.path.join(self.folder, file)
            if self.measurements_file_name in file.lower():
                performance_map = self.parse_configs_xml(file_candidate)
                break
            elif self.measurements_file_name_csv in file.lower():
                performance_map = self.parse_configs_csv(file_candidate)
                break
        if performance_map is None:
            raise ValueError("No measurement files found in {}".format(self.folder))
        return performance_map

    def parse_configs_csv(self, file):
        """Parse measurements stored in CSV format."""
        df = pd.read_csv(file, sep=";")
        print(df.head())
        # print(df)
        features = list(self.position_map.keys())
        configs_pd = df[features]
        configs = [tuple(x) for x in configs_pd.values.astype(float)]
        if not self.attribute:
            nfps = df.drop(features, axis=1)
            col = list(nfps.columns.values)[0]
        else:
            col = self.attribute
        ys_pd = df[col]
        ys = np.array(ys_pd)
        performance_map = {c: y for c, y in zip(configs, ys)}
        return performance_map

    def parse_configs_xml(self, file):
        """Parse measurement files stored in the SPLC XML format."""
        root = None
        xml_attr_name = ""
        tree = ET.parse(file)
        root = tree.getroot()
        row = next(root.iter("row"))
        data = next(row.iter("data"))
        if "column" in data.attrib:
            xml_attr_name = "column"
        elif "columname" in data.attrib:
            xml_attr_name = "columname"
        else:
            print("Could not find xml attribute that specifies content of node")
            exit(7)

        performance_map = {}
        for row in root.iter("row"):
            new_config = self.parse_config_str(xml_attr_name, row)
            perf = self.get_attribute_val(xml_attr_name, row)
            performance_map[tuple(np.array(new_config).astype(float))] = perf

        return performance_map

    def parse_config_str(self, xml_attr_name, xml_row):
        """Convert a configuration row from the XML file into a feature vector."""
        config_bin_str = xml_row.find(
            f"data[@{xml_attr_name}='Configuration']"
        ).text.strip()
        features_bin = list(
            raw_feature.strip()
            for raw_feature in config_bin_str.split(",")
            if len(raw_feature.strip()) > 0
        )
        config_num_raw = xml_row.find(f"data[@{xml_attr_name}='Variable Features']")
        if config_num_raw is not None:
            config_num_str = config_num_raw.text.strip()
            config_num_dict = dict(
                raw_feature.strip().split(";")
                for raw_feature in config_num_str.split(",")
                if len(raw_feature.strip()) > 0
            )
        else:
            config_num_dict = {}

        new_config = self.get_conf_for_feature_set(features_bin, config_num_dict)
        return new_config

    def get_conf_for_feature_set(self, features_bin, config_num_dict=None):
        """Create a numeric configuration vector from a set of feature names."""
        new_config = list(self.prototype_config)
        for bin_on_feature in features_bin:
            if (
                bin_on_feature in self.redundant_ft_names
                or bin_on_feature in self.alternative_ft_names
            ):
                continue
            pos = self.position_map[bin_on_feature]
            new_config[pos] = 1.0
        if config_num_dict is not None:
            for num_feature, val in config_num_dict.items():
                if num_feature in self.redundant_ft_names:
                    continue
                pos = self.position_map[num_feature]
                new_config[pos] = int(val)
        return new_config

    def get_attribute_val(self, xml_attr_name, row):
        """Return the performance metric encoded in a measurement row."""
        perf = None
        for data in row:
            is_attribute = False
            if not self.attribute and data.attrib[xml_attr_name] in DEFAULT_ATTRIBUTES:
                is_attribute = True
            else:
                if data.attrib[xml_attr_name] == self.attribute:
                    is_attribute = True
            if is_attribute:
                measurements = list(float(m) for m in data.text.split(","))
                perf = np.median(measurements)
                break
        if perf is None:
            print(
                "Could not find performance val. Consider specifying --attribute your-attr-name"
            )
            print("Using row: {}".format(row))
            exit(5)
        return perf

    def get_loss(self):
        return None

    def get_name(self):
        basename = os.path.basename(os.path.normpath(self.folder))
        return basename

    def query(self, query):
        n_cols = query.shape[1]
        confs = np.array(self.all_configs.values())
        match_list = []
        n_constr = 0
        for col_id, feature_val in zip(range(n_cols), query):
            if feature_val is not None:
                n_constr += 1
                matches = confs[:, col_id] == feature_val
                match_list.append(matches)
        match_arr = np.vstack(match_list)
        all_matched = np.sum(match_arr, axis=0) == n_constr
        matched_conf_ids = np.nonzero(all_matched)
        if len(matched_conf_ids) == 0:
            chosen_match = None
        else:
            matched_confs = confs[matched_conf_ids]
            chosen_match = matched_confs[0]
        return chosen_match

    def get_init(self):
        pass

    def store_csv(self, folder=None, fname=None):
        if folder is None:
            folder = self.folder
        if fname is None:
            fname = "measurements-cleared.csv"
        path = os.path.join(folder, fname)
        df_configs = self.get_measurement_df()
        df_root = pd.DataFrame(data=[1] * len(df_configs), columns=["root"])
        df_configs = pd.concat([df_root, df_configs], axis=1)
        conv_dict = {c: "int32" for c in list(self.position_map.keys())}
        df_configs = df_configs.astype(conv_dict)
        df_configs = df_configs.rename(columns={"<y>": "y"})
        df_configs.to_csv(path, index=False, sep=";")
        print("Stored measurement CSV file to", path)

        const_path = os.path.join(folder, "constant-features.txt")
        alt_path = os.path.join(folder, "alternative-features.txt")
        del_path = os.path.join(folder, "all-deleted-features.txt")
        with open(const_path, "w") as the_file:
            for ft in self.redundant_ft_names:
                the_file.write("{}\n".format(ft))
        with open(alt_path, "w") as the_file:
            for ft in self.alternative_ft_names:
                the_file.write("{}\n".format(ft))
        with open(del_path, "w") as the_file:
            for ft in self.redundant_ft_names:
                the_file.write("{}\n".format(ft))
            for ft in self.alternative_ft_names:
                the_file.write("{}\n".format(ft))

    def remove_constant_features(self):
        df_configs = self.get_all_config_df()
        redundant_ft = []
        redundant_ft_names = []
        for i, col in enumerate(df_configs.columns):
            # if len(configs[col].unique()) == 1 or col == "PS1K" or col == "CS16MB":
            if len(df_configs[col].unique()) == 1:
                redundant_ft.append(i)
                redundant_ft_names.append(col)
                df_configs.drop(col, inplace=True, axis=1)
        new_pos_map = self.get_pos_map_from_df(df_configs)
        conf_dict = {
            tuple(row): y
            for row, y in zip(
                df_configs.values.tolist(), list(self.all_configs.values())
            )
        }
        self.position_map = new_pos_map
        self.update_prototype()
        self.all_configs = conf_dict
        return redundant_ft, redundant_ft_names

    def get_measurement_df(self):
        configs = self.get_all_config_df()
        config_attrs = pd.DataFrame(list(self.all_configs.values()), columns=["<y>"])
        df_configs = pd.concat([configs, config_attrs], axis=1)
        return df_configs

    def get_all_config_df(self):
        configs = pd.DataFrame(
            list(self.all_configs.keys()), columns=self.position_map.keys()
        )
        return configs

    def remove_alternative_features(self):
        df_configs = self.get_all_config_df()
        alternative_ft = []
        alternative_ft_names = []
        group_candidates = {}

        for i, col in enumerate(df_configs.columns):
            filter_on = df_configs[col] == 1
            filter_off = df_configs[col] == 0
            group_candidates[col] = []
            for other_col in df_configs.columns:
                if other_col != col:
                    values_if_col_on = df_configs[filter_on][other_col].unique()
                    if len(values_if_col_on) == 1 and values_if_col_on[0] == 0:
                        # other feature is always off if col feature is on
                        group_candidates[col].append(other_col)

        G = nx.Graph()
        for ft, alternative_candidates in group_candidates.items():
            for candidate in alternative_candidates:
                if ft in group_candidates[candidate]:
                    G.add_edge(ft, candidate)

        cliques_remaining = True
        while cliques_remaining:
            cliques_remaining = False
            cliques = nx.find_cliques(G)
            for clique in cliques:
                # check if exactly one col is 1 in each row
                sums_per_row = df_configs[clique].sum(axis=1).unique()
                if len(sums_per_row) == 1 and sums_per_row[0] == 1.0:
                    delete_ft = sorted(clique)[0]
                    alternative_ft_names.append(delete_ft)
                    df_configs.drop(delete_ft, inplace=True, axis=1)
                    for c in clique:
                        G.remove_node(c)
                    cliques_remaining = True
                    break

        alternative_ft = [
            self.position_map[ft_name] for ft_name in alternative_ft_names
        ]

        new_pos_map = self.get_pos_map_from_df(df_configs)
        conf_dict = {
            tuple(row): y
            for row, y in zip(
                df_configs.values.tolist(), list(self.all_configs.values())
            )
        }
        self.position_map = new_pos_map
        self.update_prototype()
        self.all_configs = conf_dict
        return alternative_ft, alternative_ft_names


class DistBasedRepo(ConfigSysProxy):
    PARENT_BASE_FOLDER = "SupplementaryWebsite"
    PERF_PRED_BASE_FOLDER = "PerformancePredictions"
    SUMMARY_BASE_FOLDER = "Summary"
    MEASUREMENTS_BASE_FOLDER = "MeasuredPerformanceValues"
    T_FILE_NAMES = {
        1: "twise_t1.txt",
        2: "twise_t2.txt",
        3: "twise_t3.txt",
    }
    SPLC_COLUMNS = [
        "Round",
        "Model",
        "LearningError",
        "LearningErrorRel",
        "ValidationError",
        "ValidationErrorRel",
        "ElapsedSeconds",
        "ModelComplexity",
        "BestCandidate",
        "BestCandidateSize",
        "BestCandidateScore",
        "TestError",
    ]

    def __init__(
        self, root, sys_name, attribute=None, val_set_size=0, val_set_rnd_seed=None
    ):
        self.root = self.get_common_root(root)
        self.sys_name = sys_name

        self.summary_folder = self.get_summary_folder()
        self.measurements_folder = self.get_measurements_folder()

        super().__init__(self.measurements_folder, attribute)
        self.sample_sets = self.parse_sample_sets()

    def get_train_eval_split(self, t):
        x_train = list(self.sample_sets[t].keys())
        y_train = list(self.sample_sets[t].values())
        x_eval = list(self.all_configs.keys())
        y_eval = list(self.all_configs.values())
        return x_train, y_train, x_eval, y_eval

    def get_summary_folder(self):
        f_sum = os.path.join(
            self.root,
            DistBasedRepo.PERF_PRED_BASE_FOLDER,
            DistBasedRepo.SUMMARY_BASE_FOLDER,
            self.sys_name,
        )
        return f_sum

    def get_measurements_folder(self):
        f_meas = os.path.join(
            self.root, DistBasedRepo.MEASUREMENTS_BASE_FOLDER, self.sys_name
        )
        return f_meas

    @staticmethod
    def get_common_root(root):
        new_root = str(root)
        if DistBasedRepo.PARENT_BASE_FOLDER in list(os.listdir(root)):
            new_root = os.path.join(new_root, DistBasedRepo.PARENT_BASE_FOLDER)
        return new_root

    def parse_sample_sets(self):
        splc_lines = {}
        for filename in os.listdir(self.summary_folder):
            for t, t_name in DistBasedRepo.T_FILE_NAMES.items():
                if t_name.lower() in filename.lower():
                    abs_path = os.path.join(self.summary_folder, filename)
                    with open(abs_path) as f:
                        content = f.readlines()
                        splc_lines[t] = content
        splc_conf_sets = {}
        for t, lines in splc_lines.items():
            confs = [self.parse_line(line) for line in lines]
            splc_conf_sets[t] = {conf: y for conf, y in confs}
        return splc_conf_sets

    def parse_logs_for_coeffs(self):
        log_lines = {}
        for filename in os.listdir(self.summary_folder):
            for t, t_name in DistBasedRepo.T_FILE_NAMES.items():
                if t_name.lower() in filename.lower():
                    abs_path = os.path.join(self.summary_folder, filename)
                    with open(abs_path) as f:
                        content = f.readlines()
                        log_lines[t] = content[-1]
        splc_coeff_sets = {}
        for t, lines in log_lines.items():
            final_coeff_line = self.find_final_coeff_line(lines)
            coeff_map = self.parse_coeffs(final_coeff_line)
            splc_coeff_sets[t] = coeff_map
        return splc_coeff_sets

    def parse_line(self, line):
        conf_decoded = line.split(" ")[1][1:-1]
        feature_set = [
            f for f in conf_decoded.split("%;%") if len(f) > 0 and f != "root"
        ]
        conf = tuple(self.get_conf_for_feature_set(feature_set))
        y = self.eval(conf)
        return conf, y

    def find_final_coeff_line(self, lines):
        rev_lines = lines[::-1]
        final_line = None
        for i, line in enumerate(rev_lines):
            if "Analyze finished" in line:
                final_line = rev_lines[i + 1]
                break
        return final_line

    def parse_coeffs(self, final_coeff_line):
        df = pd.DataFrame(final_coeff_line, columns=DistBasedRepo.SPLC_COLUMNS)
        model_str = df["Model"]
        terms = model_str.split(" + ")
        coeff_map = {}
        for term in terms:
            term_comps = term.split(" * ")
            coeff, ft = term_comps[0], tuple(term_comps[1:])
            coeff_map[ft] = float(coeff)
        return coeff_map
