# from utils.np_arraylist import NP_Arraylist
import numpy as np


class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.
    """
    def __init__(self, only_forever=False, custom_windows=None, max_history_len=None):
        self.now = None
        self.queue = []
        self.total_queue = []
        self.correct_queue = []
        # self.window_lengths = [] if only_forever else [3600 * 24 * 7, 3600 * 24]
        if custom_windows is not None:
            self.window_lengths = custom_windows
        else:
            self.window_lengths = [] if only_forever else [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
        self.cursors = [0] * len(self.window_lengths)
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)

        return_vals_total = []
        running_sum_total = 0
        return_vals_correct = []
        running_sum_correct = 0
        prior_cursor = None
        if len(self.queue) > 0:
            # _, value_queue = zip(*self.queue)
            for cursor in sorted(self.cursors, reverse=True):
                running_sum_total += sum(self.total_queue[cursor:prior_cursor])
                running_sum_correct += sum(self.correct_queue[cursor:prior_cursor])
                prior_cursor = cursor
                return_vals_total.insert(0, running_sum_total)
                return_vals_correct.insert(0, running_sum_correct)
            return_vals_total.insert(0, running_sum_total + sum(self.total_queue[0:prior_cursor]))
            return_vals_correct.insert(0, running_sum_correct + sum(self.correct_queue[0:prior_cursor]))
        else:
            return_vals_total = [0 for i in range(len(self.cursors) + 1)]
            return_vals_correct = [0 for i in range(len(self.cursors) + 1)]

        return return_vals_total, return_vals_correct  # [sum(self.value_queue)] + [sum(self.value_queue[cursor:]) for cursor in self.cursors]

        # return [len(self.queue)] + [len(self.queue) - cursor
        #                             for cursor in self.cursors]

    def push(self, time, total=1.0, correct=1.0):
        if len(self.queue) and self.queue[-1] == time:
            self.total_queue[-1] += total
            self.correct_queue[-1] += correct
        else:
            self.queue.append(time)
            self.total_queue.append(total)
            self.correct_queue.append(correct)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1
        # print(t, self.queue[:self.cursors[0]],  # For debug purposes
        #       [self.queue[self.cursors[i]:self.cursors[i + 1]]
        #        for i in range(len(self.cursors) - 1)],
        #       self.queue[self.cursors[-1]:])
