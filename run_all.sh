#!/bin/bash
# End-to-end reproduction of all experiments (CPU). Expects the four public data files in ./data
set -e
mkdir -p data runs figs
python data_real.py
for s in 0 1 2 3 4; do python sim_tokens.py A $s; python sim_tokens.py B $s; done
M="lr;rf;gbdt;nsa;nsaga;mlp;cnn;lstm;sage;rgcn;bwgnn;rgt;nag;tokce;dimgt"
for ds in weibo reddit amazon yelpchi; do
  python run_real.py $ds "$M" 0,1,2,3,4
  python run_real.py $ds "dimgt-noproto;dimgt-nons;dimgt-nobsc;dimgt-nola;dimgt-norel;dimgt-self" 0,1,2,3,4
done
for ds in weibo amazon yelpchi; do python run_open.py $ds 0,1,2; done
for ds in amazon weibo; do
  for ed in 0.3 0.6; do python run_real.py $ds "gbdt;rgcn;tokce;dimgt" 0,1,2 --edge_drop $ed --tag _ed$ed; done
  for fn in 0.5 1.0; do python run_real.py $ds "gbdt;rgcn;tokce;dimgt" 0,1,2 --feat_noise $fn --tag _fn$fn; done
  python run_real.py $ds "dimgt:m_self=1;dimgt:m_self=3;dimgt:m_self=10;dimgt:lam_ns=0.3;dimgt:lam_ns=3.0;dimgt:m_tol=0.6;dimgt:m_tol=0.9;dimgt:lam_c=0.1;dimgt:lam_c=1.0;dimgt:demb=16;dimgt:demb=64" 0,1,2
done
python run_stream.py "dimgt;er;lwf;ewc;ft" 0,1,2,3,4
python run_stream.py "tokce;rgcn;gbdt;lstm;nsaga" 0,1,2
python run_stream.py "dimgt+nons;dimgt+nodrift;dimgt+notolind;dimgt+encfull;dimgt+nocensor;dimgt+nokd;dimgt+noreplay;dimgt+random;dimgt+notemp;dimgt+self" 0,1,2
python run_xplat.py "dimgt;tokce;lstm;gbdt;nsaga" 0,1,2,3,4
python bench.py yelpchi
python analyze.py
python fig_arch.py
python fig_results.py stream_fig family_heatmap calib_fig open_fig sens_fig robust_fig
python fig_embed_compute.py && python fig_embed_plot.py
