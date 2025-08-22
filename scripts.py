import os
import sys
import shutil
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from pathlib import Path
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import *
from reforge.plotting import *

sysdir = 'mdruns'
tops = io.pull_files(sysdir, 'mdc.pdb')
trajs = io.pull_files(sysdir, 'mdc.xtc')

def do_pca():
    from sklearn.decomposition import PCA
    from sklearn.cluster import BisectingKMeans, KMeans
    from sklearn.mixture import GaussianMixture
    us = [mda.Universe(top, traj, in_memory_step=1, in_memory=True) for top, traj in zip(tops, trajs)]
    ref = mda.Universe(tops[0])
    all_positions = []
    all_labels = []
    for idx, u in enumerate(us):
        selection = "name CA"
        ag = u.atoms.select_atoms(selection)
        # Tranform traj
        ref_sel = ref.select_atoms(selection) 
        u.trajectory.add_transformations(
            # trans.unwrap(u.atoms),                         # optional: fix PBC breaks first
            # trans.center_in_box(ag, wrap=False),            # optional: remove translation drift
            trans.fit.fit_rot_trans(ag, ref_sel)            # remove rotation & translation
        )
        positions = io.read_positions(u, ag, sample_rate=1, b=0, e=1e9)
        labels = np.full(positions.shape[0], idx)
        all_positions.append(positions)
        all_labels.append(labels)
    positions = np.concatenate(all_positions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    # PCA
    pca = PCA(n_components=10)
    x_r = pca.fit_transform(positions) # (n_samples, n_features)
    # x_r = np.stack((x_r[:, 0], x_r[:, 1]), axis=1)
    # # Clustering
    # n_clusters = 3
    # algo = GaussianMixture(n_components=n_clusters, random_state=0, n_init=3)
    # # algo = KMeans(n_clusters=n_clusters, random_state=150, n_init=10)
    # pred = algo.fit_predict(x_r)
    # labels = pred 
    # PLotting
    plt.scatter(x_r[:, 0], x_r[:, 1], alpha=0.5, c=labels)
    # plt.scatter(centers[:, 0], centers[:, 1], c="r", s=20)
    plt.savefig("png/pca_01.png")
    plt.close()
    plt.scatter(x_r[:, -2], x_r[:, -1], alpha=0.5, c=labels)
    # plt.scatter(centers[:, 0], centers[:, 1], c="r", s=20)
    plt.savefig("png/pca_12.png")
    plt.close()
    # exit()
    # for idx in range(n_clusters):
    #     # Topology
    #     u.trajectory[0]
    #     ag.atoms.write(str(mdsys.datdir / f"topology_{idx}.pdb"))
    #     # Saving samples
    #     mask = pred == idx
    #     subset = u.trajectory[mask]
    #     traj_path = str(mdsys.datdir / f"cluster_{idx}.xtc")
    #     with mda.Writer(traj_path, ag.n_atoms) as W:
    #         for ts in subset:   
    #             W.write(ag) 


def do_dfi_dci():
    ref = mda.Universe(tops[0])
    for top, traj in zip(tops, trajs):
        mdrun = GmxRun(sysdir, top.split("/")[1], "bioemu")
        mdrun.prepare_files()
        u = mda.Universe(top, traj, in_memory_step=1, in_memory=True)
        selection = "name CA"
        ag = u.atoms.select_atoms(selection)
        # Tranform traj
        ref_sel = ref.select_atoms(selection) 
        u.trajectory.add_transformations(
            # trans.unwrap(u.atoms),                         # optional: fix PBC breaks first
            # trans.center_in_box(ag, wrap=False),            # optional: remove translation drift
            trans.fit.fit_rot_trans(ag, ref_sel)            # remove rotation & translation
        )
        clean_dir(mdrun.covdir, '*npy')
        mdrun.get_covmats(u, ag, sample_rate=1, b=0, e=1e10, n=1, outtag='covmat') 
        mdrun.get_pertmats()
        mdrun.get_dfi(outtag='dfi')
        mdrun.get_dci(outtag='dci', asym=False)
        mdrun.get_dci(outtag='asym', asym=True)


def get_averages():
    sysnames = os.listdir(sysdir)
    for sysname in sysnames:
        system = GmxSystem(sysdir, sysname)
        os.makedirs(system.datdir, exist_ok=True)   
        system.get_mean_sem(pattern='dfi*.npy')
        system.get_mean_sem(pattern='dci*.npy')
        system.get_mean_sem(pattern='asym*.npy')


def set_ax_parameters(ax, xlabel=None, ylabel=None, axtitle=None, loc='lower right'):
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5, width=1.5)
    # Increase spine width for a bolder look
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Add a legend with custom font size and no frame
    legend = ax.legend(fontsize=14, frameon=False, loc=loc)
    # Optionally, add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_dfi(): 
    files = io.pull_files(sysdir, "dfi*")
    datas = [np.load(file) for file in files if '_av' in file]
    errs = [np.load(file) for file in files if '_err' in file]
    labels = [f.split("/")[1][5:] for f in files if '_av' in f]
    xs = [np.arange(len(data))+26 for data in datas]
    params = [{'lw':2, 'label':label} for label in labels]
    data_ref = datas[-1]
    err_ref = errs[-1]
    datas = [data - data_ref for data in datas]
    errs = [np.sqrt(err**2 + err_ref**2) for err in errs]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI', loc='lower right')
    plot_figure(fig, ax, figname="Delta DFI", figpath='png/ddfi.png',)


def plot_pdfi():
    files = io.pull_files(sysdir, "dfi*")
    datas = [np.load(file) for file in files if '_av' in file]
    errs = [np.load(file) for file in files if '_err' in file]
    labels = [f.split("/")[1][6:] for f in files if '_av' in f]
    xs = [np.arange(len(data))+26 for data in datas]
    datas = [mdm.percentile(data) for data in datas]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI', loc='lower right')
    plot_figure(fig, ax, figname="%DFI", figpath='png/pdfi.png',)


def plot_dci():
    files = io.pull_files(sysdir, "dci*")
    datas = [np.load(file) for file in files if '_av' in file]
    errs = [np.load(file) for file in files if '_err' in file]
    labels = [f.split("/")[1][5:] for f in files if '_av' in f]
    data_ref = datas[-1]
    param = {'lw':2}
    # Plotting
    for data, label in zip(datas, labels):
        fig, ax = init_figure(grid=(1, 1), axsize=(10, 10))
        make_heatmap(ax, data-data_ref, cmap='bwr', interpolation=None, vmin=-1, vmax=1)
        set_ax_parameters(ax, xlabel='Pert Res', ylabel='Resp Res')
        plot_figure(fig, ax, figname=f'{label}', figpath=f'png/dci_{label}.png',)


if __name__ == '__main__':
    do_pca()
    # do_dfi_dci()
    # get_averages()
    # plot_dfi()
    # plot_pdfi()
    # plot_dci()