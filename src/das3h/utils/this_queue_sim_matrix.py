import numpy as np


class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.
    """
    def __init__(self, only_forever=False, custom_windows=None, max_history_len=None):
        self.now = None
        self.max_history_len = max_history_len

        # self.total_queue = []
        self.correct_queue = []
        self.skill_queue = []
        # self.window_lengths = [] if only_forever else [3600 * 24 * 7, 3600 * 24]
        if custom_windows is not None:
            self.window_lengths = custom_windows
        else:
            self.window_lengths = [] if only_forever else [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
        self.cursors = [0] * len(self.window_lengths)
        if len(self.window_lengths):
            self.queue = []

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t, skill, sim_matrix):
        self.update_cursors(t)

        return_vals_total = []
        running_sum_total = 0
        return_vals_correct = []
        running_sum_correct = 0
        prior_cursor = None
        if len(self.correct_queue) > 0:
            skill_sims = np.array(list(map(lambda x: sim_matrix[skill, x], self.skill_queue)))
            correct_queue = np.array(self.correct_queue)
            if self.max_history_len is not None:
                lower_bound = len(self.correct_queue) - self.max_history_len
                if lower_bound < 0:
                    lower_bound = 0
            else:
                lower_bound = 0
            for cursor in sorted(self.cursors, reverse=True):
                if cursor < lower_bound:
                    cursor = lower_bound
                skill_sims_subset = skill_sims[cursor:prior_cursor]
                running_sum_total += np.sum(skill_sims_subset)
                running_sum_correct += np.sum(correct_queue[cursor:prior_cursor] * skill_sims_subset)
                prior_cursor = cursor
                return_vals_total.insert(0, running_sum_total)
                return_vals_correct.insert(0, running_sum_correct)
            skill_sims_subset = skill_sims[lower_bound:prior_cursor]
            return_vals_total.insert(0, running_sum_total + np.sum(skill_sims_subset))
            return_vals_correct.insert(0, running_sum_correct + np.sum(correct_queue[lower_bound:prior_cursor] * skill_sims_subset))
        else:
            return_vals_total = [0 for i in range(len(self.cursors) + 1)]
            return_vals_correct = [0 for i in range(len(self.cursors) + 1)]

        return return_vals_total, return_vals_correct  # [sum(self.value_queue)] + [sum(self.value_queue[cursor:]) for cursor in self.cursors]

        # return [len(self.queue)] + [len(self.queue) - cursor
        #                             for cursor in self.cursors]

    def push(self, time, skill, correct=1.0):
        if len(self.window_lengths):
            self.queue.append(time)
        # self.total_queue.append(total)
        self.correct_queue.append(correct)
        self.skill_queue.append(skill)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1
        # print(t, self.queue[:self.cursors[0]],  # For debug purposes
        #       [self.queue[self.cursors[i]:self.cursors[i + 1]]
        #        for i in range(len(self.cursors) - 1)],
        #       self.queue[self.cursors[-1]:])
