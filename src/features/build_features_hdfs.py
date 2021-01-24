import argparse

from src.features.hdfs import create_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HDFS1 logs to numerical representation using fastText.')
    parser.add_argument('model', action='store', type=str, help='a path to the first trained cross-validated fastText '
                                                                'model on HDFS1')
    parser.add_argument('-in', type=str, metavar='PATH/TO/FOLDER', dest='input', default='../../data/interim/HDFS1',
                        help='a location with HDFS1 training data (default: ../../data/interim/HDFS1)')
    parser.add_argument('-out', type=str, metavar='PATH/TO/FOLDER', dest='output', default='../../data/processed/HDFS1',
                        help='a location where all processed data will be saved (default: ../../data/processed/HDFS1)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--per-block', dest='per_block', action='store_true', help='generate embeddings and labels per '
                                                                                  'a block of logs')
    group.add_argument('--per-log', dest='per_log', action='store_true', help='generate embeddings and labels per '
                                                                              'a log line')

    args = parser.parse_args()

    create_embeddings(args.input, args.output, args.model, args.per_block)
