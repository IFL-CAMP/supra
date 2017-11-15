import sys
import os
import fnmatch
import numpy as np

if len(sys.argv) < 2:
    # find the newest "supra.log"
    logfile_name = "supra.log"
    start_dir = os.path.abspath(os.path.join(os.curdir, ".."))
    newestTime = 0
    logfile = ""
    for paths, dirs, filenames in os.walk(os.path.abspath(start_dir)):
        for filename_filtered in fnmatch.filter(filenames, "*"+logfile_name):
            filename_complete = os.path.join(paths, filename_filtered)
            file_time = os.stat(filename_complete).st_mtime
            if file_time > newestTime:
                newestTime = file_time
                logfile = filename_complete
else:
    logfile = sys.argv[1]

if len(sys.argv) < 3:
    ignore_nodes = ['MHD']
else:
    ignore_nodes = sys.argv[2].split(',')

print("Evaluating profiling from logfile '" + logfile + "'")
lines_profiling = [line.rstrip('\n') for line in open(logfile) if line.startswith("Profiling")]
print("Found ", len(lines_profiling), " profiling lines. processing...")
node_profile_raw = {}
for line in lines_profiling:
    tokens = line.split(" ")
    assert tokens[0] == "Profiling"
    if tokens[1] not in ignore_nodes:
        if tokens[1] not in node_profile_raw:
            node_profile_raw[tokens[1]] = []
        node_profile_raw[tokens[1]].append([tokens[2], tokens[3], tokens[4]])

node_profile = {}
first_sample_overall = 0
last_sample_overall = sys.maxsize
for node in node_profile_raw.keys():
    starts = np.asarray([float(l[0]) for l in node_profile_raw[node] if l[1] == "S"])
    ends = np.asarray([float(l[0]) for l in node_profile_raw[node] if l[1] == "E"])

    if not (len(starts) == len(ends) and
            node_profile_raw[node][0][1] == "S" and node_profile_raw[node][-1][1] == "E" and
            node_profile_raw[node][0][2] == node_profile_raw[node][1][2] and
            int(node_profile_raw[node][0][2]) == 0 and
            node_profile_raw[node][-1][2] == node_profile_raw[node][-1][2]):
        print("Profile entries of node '" + node + "' are scrambled. Skipping!")
        continue

    diff_ts = (ends - starts)*1000
    diff_ts = diff_ts[round(len(diff_ts)/10):]
    num_samples = len(diff_ts)

    first_sample_overall = max(first_sample_overall, int(node_profile_raw[node][0][2]))
    last_sample_overall = min(last_sample_overall, int(node_profile_raw[node][-1][2]))

    node_profile[node] = {
        'mean' : np.mean(diff_ts),
        'std' : np.std(diff_ts),
        'median' : np.median(diff_ts),
        'min' : np.min(diff_ts),
        'max' : np.max(diff_ts)
    }

    print("Node '{0}' ({6} samples)\n    mean {1} ms\n    std {2} ms\n    median {3} ms\n    min {4} ms\n    max {5} ms".format(
        node,
        node_profile[node]['mean'],
        node_profile[node]['std'],
        node_profile[node]['median'],
        node_profile[node]['min'],
        node_profile[node]['max'],
        num_samples))

num_samples_overall = last_sample_overall - first_sample_overall
diff_t_overall = np.zeros((num_samples_overall, 1))
for node in node_profile_raw.keys():
    starts = np.asarray([float(l[0]) for l in node_profile_raw[node] if l[1] == "S" and int(l[2]) >= first_sample_overall and int(l[2]) <= last_sample_overall])
    ends = np.asarray([float(l[0]) for l in node_profile_raw[node] if l[1] == "E" and int(l[2]) >= first_sample_overall and int(l[2]) <= last_sample_overall])

    if not (len(starts) == len(ends) and
            node_profile_raw[node][0][1] == "S" and node_profile_raw[node][-1][1] == "E" and
            node_profile_raw[node][0][2] == node_profile_raw[node][1][2] and
            int(node_profile_raw[node][0][2]) == 0 and
            node_profile_raw[node][-1][2] == node_profile_raw[node][-1][2]):
        continue

    diff_ts = (ends - starts) * 1000
    diff_t_overall = diff_t_overall + diff_ts

diff_t_overall = diff_t_overall[round(len(diff_t_overall) / 10):]
num_samples_overall = len(diff_t_overall)
node = 'overall'
node_profile[node] = {
    'mean' : np.mean(diff_t_overall),
    'std' : np.std(diff_t_overall),
    'median' : np.median(diff_t_overall),
    'min' : np.min(diff_t_overall),
    'max' : np.max(diff_t_overall)
}

print("{0} ({6} samples)\n    mean {1} ms\n    std {2} ms\n    median {3} ms\n    min {4} ms\n    max {5} ms".format(
    'overall',
    node_profile[node]['mean'],
    node_profile[node]['std'],
    node_profile[node]['median'],
    node_profile[node]['min'],
    node_profile[node]['max'],
    num_samples_overall))
