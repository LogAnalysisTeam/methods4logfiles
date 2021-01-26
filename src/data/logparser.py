import pandas as pd
from drain3 import TemplateMiner
from typing import List, Dict


def get_log_templates(clusters: List) -> Dict:
    ret = {cl.cluster_id: ' '.join(cl.log_template_tokens) for cl in clusters}
    return ret


def get_log_structure(log_lines: List, cluster_ids: List, clusters: List) -> List:
    templates = get_log_templates(clusters)

    ret_log_structure = []
    for curr_id, log in zip(cluster_ids, log_lines):
        ret_log_structure.append([curr_id, log, templates[curr_id]])
    return ret_log_structure


def parse_file_drain3(file_path: str):
    template_miner = TemplateMiner()

    cluster_ids = []
    log_lines = []
    with open(file_path) as f:
        for line in f:
            line = line.rstrip().partition(': ')[2]  # produces tuple (pre, delimiter, post)
            result = template_miner.add_log_message(line)
            cluster_ids.append(result['cluster_id'])
            log_lines.append(line)

    ret_log_structure = get_log_structure(log_lines, cluster_ids, template_miner.drain.clusters)
    return ret_log_structure


if __name__ == '__main__':
    out = parse_file_drain3('../../data/raw/HDFS1/HDFS.log')
    pd.DataFrame(out, columns=["EventId", "Content", "EventTemplate"]).to_csv(
        '~/bdip25/src/data' + "/HDFS_Drain3.log_structured-new.csv", index=False)
