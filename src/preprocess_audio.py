import os
import subprocess
import multiprocessing as mp
from argparse import ArgumentParser

from torchtree import Directory_Tree
from tqdm import tqdm


def apply_single(video_path: str, dst_path: str, input_options: list, output_options: list, ext: None,verbose=True):
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output


    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path)
    assert os.path.isdir(os.path.dirname(dst_path))
    if ext is not None:
        dst_path = os.path.splitext(dst_path)[0] + ext
    result = subprocess.Popen(["ffmpeg", *input_options, '-i', video_path, *output_options, dst_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read().decode("utf-8")
    stderr = result.stderr
    if verbose:
        if stdout != '':
            print(stdout)
        if stderr is not None:
            print(stderr.read().decode("utf-8"))


def apply_tree(root, dst, input_options=list(), output_options=list(), multiprocessing=0, fn=apply_single, ignore=[],
               ext=None):
    """
    Applies ffmpeg processing for a given directory tree for whatever which fits this format:
        ffmpeg [input options] -i input [output options] output
    System automatically checks if files in directory are ffmpeg compatible
    Results will be stored in a replicated tree with same structure and filenames


    :param root: Root directory in which files are stored
    :param dst: Destiny directory in which to store results
    :param input_options: list[str] ffmpeg input options in a subprocess format
    :param output_options: list[str] ffmpeg output options in a subprocess format
    :param multiprocessing: int if 0 disables multiprocessin, else enables multiprocessing with that amount of cores.
    :param fn: funcion to be used. By default requires I/O options.
    :return: None
    """
    formats = ['wav','flac']
    tree = Directory_Tree(root, ignore=ignore)  # Directory tree
    if not os.path.exists(dst):
        os.mkdir(dst)
    tree.clone_tree(dst)  # Generates new directory structure (folders)

    # Python Multiproceesing mode
    if multiprocessing > 0:
        pool = mp.Pool(multiprocessing)
        results = [pool.apply(fn,
                              args=(i_path, o_path),
                              kwds={input_options: input_options, output_options: output_options, ext: ext})
                   for i_path, o_path in zip(tree.paths(root), tree.paths(dst)) if
                   os.path.splitext(i_path)[1][1:] in formats]
        pool.close()
    else:
        for i_path, o_path in tqdm(list(zip(tree.paths(root), tree.paths(dst)))):
            if os.path.splitext(i_path)[1][1:] in formats:
                fn(i_path, o_path, input_options=input_options, output_options=output_options, ext=ext, verbose=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--n_threads', type=int, default=0)  # 0 disables multiprocessing
    args = parser.parse_args()

    apply_tree(args.src, args.dst,
               output_options=['-ar', '16000', '-ac', '1'],
               multiprocessing=args.n_threads, ext='.wav')
