import os
import json
import numpy as np
import matplotlib.pyplot as plt


class Experiment(object):

    def __init__(self, fname, node_name, has_init_metrics=True):
        with open(os.path.abspath(fname)) as f:
            self.results = json.load(f)
        self.node_name = node_name
        self.steady_metrics = self.extract_metrics(self.results["client_metrics"])
        self.has_init_metrics = has_init_metrics
        if self.has_init_metrics:
            self.init_metrics = self.extract_metrics(self.results["init_client_metrics"])

    def plot_all(self, plot_init=True):
        self.plot_processing_latency(plot_init)
        self.plot_queue_sizes(plot_init)
        self.plot_send_times(plot_init)
        self.plot_batch_sizes(plot_init)
        self.plot_thruputs(plot_init)
        self.plot_container_handle_metrics()
        self.plot_container_arrival_metrics()

    def extract_metrics(self, m):
        all_metrics = m[0]["all_metrics"]
        aggregated_metrics = {}
        for idx, trial in enumerate(all_metrics):
            data_lists = trial["data_lists"]
            for data in data_lists:
                for entry in data:
                    if idx == 0:
                        aggregated_metrics[entry] = {"times": [], "vals": []}
                    for item in data[entry]["items"]:
                        if len(item) == 1:
                            aggregated_metrics[entry]["times"].append(int(list(item.keys())[0]))
                            aggregated_metrics[entry]["vals"].append(float(list(item.values())[0]))
                        else:
                            for i in item:
                                aggregated_metrics[entry]["times"].append(int(list(i.keys())[0]))
                                aggregated_metrics[entry]["vals"].append(float(list(i.values())[0]))
        return aggregated_metrics

    def plot_processing_latency(self, plot_init=True):
        metric_name = "{}:processing_latency".format(self.node_name)
        fig, ax = plt.subplots(figsize=(16, 8))

        def plot(m, metrics, color, label):
            init_time = metrics[m]["times"][0]
            # times = np.array(metrics[m]["times"])
            offset_times = np.array(metrics[m]["times"]) - init_time
            vals = np.array(metrics[m]["vals"][10:])
            ax.scatter(offset_times[10:], vals, color=color, label=label)
            print("metric: {m}, stage: {label}, mean: {mean:.4f}, std: {std:.4f}".format(
                m=m,
                label=label,
                mean=np.mean(vals),
                std=np.std(vals),
            ))

        plot(metric_name, self.steady_metrics, "blue", "steady")
        if self.has_init_metrics and plot_init:
            plot(metric_name, self.init_metrics, "orange", "init")
        print("")
        cur_ymax = ax.get_ylim()[1]
        ax.set_ylim(top=min(cur_ymax, 2000))
        ax.set_xlabel("time offset (us)")
        ax.legend()
        ax.set_title(metric_name)
        plt.show()

    def plot_queue_sizes(self, plot_init=True):
        metric_name = "{}:1:queue_sizes".format(self.node_name)
        fig, ax = plt.subplots(figsize=(16, 8))

        def plot(m, metrics, color, label):
            init_time = metrics[m]["times"][0]
            # times = np.array(metrics[m]["times"])
            offset_times = np.array(metrics[m]["times"]) - init_time
            vals = np.array(metrics[m]["vals"][10:])
            ax.scatter(offset_times[10:], vals, color=color, label=label)
            print("metric: {m}, stage: {label}, mean: {mean:.4f}, std: {std:.4f}".format(
                m=m,
                label=label,
                mean=np.mean(vals),
                std=np.std(vals),
            ))

        plot(metric_name, self.steady_metrics, "blue", "steady")
        if self.has_init_metrics and plot_init:
            plot(metric_name, self.init_metrics, "orange", "init")
        print("")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        # cur_ymax = ax.get_ylim()[1]
        # ax.set_ylim(top=min(cur_ymax, 2000))
        ax.set_xlabel("time offset (us)")
        ax.set_ylabel("queue size")
        ax.legend()
        ax.set_title(metric_name)
        plt.show()

    def plot_send_times(self, plot_init=True):
        metric_name = "send_times"
        if metric_name in self.steady_metrics:
            fig, ax = plt.subplots(figsize=(16, 8))

            def plot(m, metrics, color, label):
                diffs = np.diff(np.array(metrics[m]["vals"]) / 1000.0)
                ax.scatter(np.arange(len(diffs)), diffs, color=color, label=label)
                print("metric: {m}, stage: {label}, mean: {mean:.4f}, std: {std:.4f}".format(
                    m=m,
                    label=label,
                    mean=np.mean(diffs),
                    std=np.std(diffs),
                ))

            plot(metric_name, self.steady_metrics, "blue", "steady")
            if self.has_init_metrics and plot_init:
                plot(metric_name, self.init_metrics, "orange", "init")
            print("")
            # cur_ymax = ax.get_ylim()[1]
            # ax.set_ylim(top=min(cur_ymax, 2000))
            ax.set_xlabel("message number")
            ax.set_ylabel("inter-message time (ms)")
            ax.legend()
            ax.set_title(metric_name)
            plt.show()

    def extract_batch_sizes(self, m):
        all_metrics = m[0]["all_metrics"]
        batch_sizes = {"min": [], "mean": [], "p50": []}
        for idx, trial in enumerate(all_metrics):
            hists = trial["histograms"]
            for h in hists:
                name = list(h.keys())[0]
                vals = list(h.values())[0]
                if "batch_size" in name:
                    for v in vals:
                        if v in batch_sizes:
                            batch_sizes[v].append(float(vals[v]))
        return batch_sizes

    def plot_batch_sizes(self, plot_init=True):
        steady_batches = self.extract_batch_sizes(self.results["client_metrics"])
        if self.has_init_metrics and plot_init:
            init_batches = self.extract_batch_sizes(self.results["init_client_metrics"])
        for m in steady_batches:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(np.arange(len(steady_batches[m])) + 0.25,
                       steady_batches[m],
                       color="blue",
                       label="steady")
            if self.has_init_metrics and plot_init:
                ax.scatter(np.arange(len(init_batches[m])) - 0.25,
                           init_batches[m],
                           color="orange",
                           label="init")
            ax.set_title("{} batch size".format(m))
            ax.legend()
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.4)
        plt.show()

    def plot_thruputs(self, plot_init=True):
        steady_thrus = self.results["client_metrics"][0]["thrus"]
        if self.has_init_metrics and plot_init:
            init_thrus = self.results["init_client_metrics"][0]["thrus"]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(steady_thrus)) + 0.25, steady_thrus, color="blue", label="steady")
        if self.has_init_metrics and plot_init:
            ax.plot(np.arange(len(init_thrus)) - 0.25, init_thrus, color="orange", label="init")
        ax.set_title("Mean Throughput")
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.show()
        if self.has_init_metrics and plot_init:
            print("init mean throughput: {}, std: {}".format(np.mean(init_thrus),
                                                             np.std(init_thrus)))
        print("steady mean throughput: {}, std: {}".format(np.mean(steady_thrus),
                                                           np.std(steady_thrus)))
        
    def extract_request_enqueue_rate(self, m):
        all_metrics = m[0]["all_metrics"]
        rates = []
        for idx, trial in enumerate(all_metrics):
            meters = trial["meters"]
            for m in meters:
                name = list(m.keys())[0]
                if name == "frontend_rpc:request_enqueue":
                    rates.append(float(m[name]["rate"]))
        return rates

    
    def plot_request_enqueue_rate(self, plot_init=True):
        steady_rates = self.extract_request_enqueue_rate(self.results["client_metrics"])
        if self.has_init_metrics and plot_init:
            init_rates = self.extract_request_enqueue_rate(self.results["init_client_metrics"])
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(steady_rates)), steady_rates, color="blue", label="steady")
        if self.has_init_metrics and plot_init:
            ax.plot(np.arange(len(init_rates)), init_rates, color="orange", label="init")
        ax.set_title("Request enqueue rate")
        ax.set_ylabel("Request enqueue rate (qps)")
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.show()
        if self.has_init_metrics and plot_init:
            print("init mean rate: {}, std: {}".format(np.mean(init_rates),
                                                             np.std(init_rates)))
        print("steady mean rate: {}, std: {}".format(np.mean(steady_rates),
                                                           np.std(steady_rates)))

    def plot_container_handle_metrics(self):
        if "container_metrics" in self.results:
            loop_durs = np.array(self.results["container_metrics"]["loop_durs"][10:]) / 1000.0
            handle_durs = np.array(self.results["container_metrics"]["handle_durs"][10:]) / 1000.0
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(np.arange(len(loop_durs)), loop_durs, color="red", label="loop durs")
            ax.scatter(np.arange(len(handle_durs)), handle_durs, color="green", label="handle durs")

            ax.set_title("Container metrics")
            ax.set_ylabel("duration (ms)")
            cur_ymax = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=min(cur_ymax, 2000))
            ax.legend()

            plt.show()

    def plot_container_arrival_metrics(self):
        if "container_metrics" in self.results:
            recv_times = np.array(self.results["container_metrics"]["recv_times"][10:]) * 1000.0
            fig, ax_scatter = plt.subplots(ncols=1, figsize=(12, 8))
            diffs = np.diff(recv_times)
            ax_scatter.scatter(np.arange(len(diffs)),
                               diffs,
                               color="red",
                               label="batch arrival times")
            ax_scatter.set_title("Container batch arrival times")
            ax_scatter.set_ylabel("inter-arrival time (ms)")
            ax_scatter.set_xlabel("message number")
            ax_scatter.legend()
            cur_ymax = ax_scatter.get_ylim()[1]
            ax_scatter.set_ylim(bottom=0, top=min(cur_ymax, 2000))
            plt.show()
