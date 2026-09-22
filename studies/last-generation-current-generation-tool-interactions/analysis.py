from __future__ import annotations

import csv
import hashlib
import html as html_lib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
BENCHMARKS = OUT.parent
REPO = BENCHMARKS.parent
SOURCES = [
    {
        'id': 'prior_relace',
        'label': 'Previous · both models via Relace',
        'dir': BENCHMARKS / 'deepseek-v4-v41-relace-20260920-185646',
        'edge': '#35445c',
    },
    {
        'id': 'current_mixed_provider',
        'label': 'Current · V4 via Relace; V4.1 via DeepSeek',
        'dir': BENCHMARKS / 'deepseek-v4-v41-relace-deepseek-seed-260924',
        'edge': '#b45309',
    },
]
ARMS = ['baseline', 'caveman', 'ponytail', 'caveman_ponytail']
ARM_LABEL = {
    'baseline': 'Baseline',
    'caveman': 'Caveman',
    'ponytail': 'Ponytail',
    'caveman_ponytail': 'Caveman + Ponytail',
}
ARM_SHORT = {'baseline': 'Base', 'caveman': 'Caveman', 'ponytail': 'Ponytail', 'caveman_ponytail': 'Both'}
ARM_COLOR = {
    'baseline': '#667085',
    'caveman': '#2478a8',
    'ponytail': '#399456',
    'caveman_ponytail': '#d26a2e',
}
MODEL_MARKER = {
    'deepseek/deepseek-v4-flash-0731': 'o',
    'deepseek/deepseek-v4.1-flash-20260910': 's',
}
MODEL_SHORT = {
    'deepseek/deepseek-v4-flash-0731': 'V4',
    'deepseek/deepseek-v4.1-flash-20260910': 'V4.1',
}
MODEL_ORDER = list(MODEL_MARKER)

TRACE_SOURCES = {
    'api-events.jsonl': 'api_events.jsonl',
    'tool-events.jsonl': 'tool_events.jsonl',
    'hermes-events.jsonl': 'hermes_events.jsonl',
    'lifecycle-events.jsonl': 'lifecycle_events.jsonl',
    'hermes-session.trace.jsonl': 'hermes_session_trace.jsonl',
    'hermes-session.jsonl': 'hermes_session_records.jsonl',
    'hermes-session.json': 'hermes_session_records.jsonl',
    'usage.json': 'usage_records.jsonl',
    'timing-summary.json': 'timing_summaries.jsonl',
    'treatment-evidence.json': 'treatment_evidence.jsonl',
    'model.patch': 'model_patches.jsonl',
}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    out.append(value)
            except json.JSONDecodeError:
                continue
    return out


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    combined: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for source in SOURCES:
        src = source['dir']
        meta = json.loads((src / 'metadata.json').read_text(encoding='utf-8'))
        rows = json.loads((src / 'runs.json').read_text(encoding='utf-8'))
        descriptor = {k: v for k, v in source.items() if k != 'dir'}
        descriptor.update({
            'source_path': relpath(src),
            'generated_at': meta.get('generated_at'),
            'dataset': meta.get('dataset'),
            'dataset_revision': meta.get('dataset_revision'),
            'selected_instance_ids': meta.get('selected_instance_ids'),
            'runtime_official_evaluation_setting': meta.get('official_evaluation'),
            'models': meta.get('models'),
        })
        metadata.append(descriptor)
        for row in rows:
            original_dir = Path(row['run_dir'])
            record = dict(row)
            record.update({
                'study_id': source['id'],
                'study_label': source['label'],
                'study_date': str(meta.get('generated_at', ''))[:10],
                'study_source_path': relpath(src),
                'source_run_path': relpath(original_dir),
                'source_run_id': original_dir.name,
                'model_short': MODEL_SHORT.get(row.get('model'), row.get('model', 'unknown')),
                'treatment_label': ARM_LABEL.get(row.get('arm'), row.get('arm', 'unknown')),
                'upstream_provider': row.get('upstream_provider') or meta.get('upstream_provider'),
            })
            # Keep the combined export portable; the original absolute path is retained in source_run_path.
            record['run_dir'] = relpath(original_dir)
            combined.append(record)
            context = {
                'study_id': source['id'],
                'study_label': source['label'],
                'source_run_id': original_dir.name,
                'source_run_path': relpath(original_dir),
                'run_index': row.get('run_index'),
                'model': row.get('model'),
                'upstream_provider': row.get('upstream_provider'),
                'treatment': row.get('arm'),
                'task': row.get('task'),
            }
            for filename, merged_name in TRACE_SOURCES.items():
                file_path = original_dir / filename
                if not file_path.is_file():
                    manifests.append({**context, 'source_file': filename, 'merged_file': f'traces/{merged_name}', 'exists': False})
                    continue
                data = file_path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                text = data.decode('utf-8', errors='replace')
                manifests.append({
                    **context,
                    'source_file': filename,
                    'merged_file': f'traces/{merged_name}',
                    'exists': True,
                    'source_bytes': len(data),
                    'source_sha256': digest,
                    'source_records_or_lines': len([line for line in text.splitlines() if line.strip()]) if filename.endswith('.jsonl') else 1,
                })
                dest = OUT / 'traces' / merged_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if filename == 'model.patch':
                    item = {**context, 'patch': text}
                    with dest.open('a', encoding='utf-8') as h:
                        h.write(json.dumps(item, ensure_ascii=False) + '\n')
                elif filename.endswith('.jsonl'):
                    with dest.open('a', encoding='utf-8') as h:
                        for line_no, line in enumerate(text.splitlines(), start=1):
                            if not line.strip():
                                continue
                            try:
                                payload = json.loads(line)
                            except json.JSONDecodeError:
                                payload = {'unparsed_line': line}
                            h.write(json.dumps({**context, 'source_file': filename, 'source_line': line_no, 'record': payload}, ensure_ascii=False) + '\n')
                else:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {'text': text}
                    with dest.open('a', encoding='utf-8') as h:
                        h.write(json.dumps({**context, 'record': payload}, ensure_ascii=False) + '\n')
    return combined, manifests, metadata


def load_published_bundle() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the compact release bundle without requiring its source workspaces."""
    combined = json.loads((OUT / 'combined_runs.json').read_text(encoding='utf-8'))
    metadata = json.loads((OUT / 'source_studies.json').read_text(encoding='utf-8'))
    manifest: list[dict[str, Any]] = []
    manifest_path = OUT / 'source_trace_manifest.csv'
    if manifest_path.is_file():
        with manifest_path.open(newline='', encoding='utf-8') as stream:
            manifest = list(csv.DictReader(stream))
    return combined, manifest, metadata


def theme() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.edgecolor': '#475467',
        'axes.labelcolor': '#1d2939',
        'xtick.color': '#344054',
        'ytick.color': '#344054',
        'text.color': '#101828',
        'grid.color': '#d0d5dd',
        'grid.alpha': 0.55,
    })


def savefig(fig: Any, stem: str) -> None:
    fig.savefig(OUT / f'{stem}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def metric_figures(rows: list[dict[str, Any]]) -> None:
    metrics = [
        ('total_tokens', 'Total tokens', True),
        ('cost_usd', 'Cost (USD)', True),
        ('wall_seconds', 'Agent wall time (s)', True),
        ('api_calls', 'API calls', False),
        ('tool_calls', 'Tool calls', False),
    ]
    fig, axes = plt.subplots(2, len(metrics), figsize=(21, 9), sharex='col')
    study_ids = [s['id'] for s in SOURCES]
    tasks = sorted({r['task'] for r in rows})
    task_jitter = {task: (i - (len(tasks)-1)/2) * 0.025 for i, task in enumerate(tasks)}
    model_offset = {MODEL_ORDER[0]: -0.12, MODEL_ORDER[1]: 0.12}
    for ri, source in enumerate(SOURCES):
        subset = [r for r in rows if r['study_id'] == source['id'] and r.get('completed')]
        for ci, (metric, title, logscale) in enumerate(metrics):
            ax = axes[ri, ci]
            ax.grid(axis='y', linewidth=0.7)
            for xi, arm in enumerate(ARMS):
                for model in MODEL_ORDER:
                    group = [r for r in subset if r['arm'] == arm and r['model'] == model and r.get(metric) is not None]
                    if not group:
                        continue
                    xs, ys = [], []
                    for r in group:
                        x = xi + model_offset[model] + task_jitter[r['task']]
                        y = float(r[metric])
                        xs.append(x); ys.append(y)
                        ax.scatter(x, y, s=56, marker=MODEL_MARKER[model], c=ARM_COLOR[arm],
                                   edgecolors='white', linewidths=0.9, alpha=0.9, zorder=3)
                        if not r.get('valid', True):
                            ax.scatter(x, y, s=90, marker='x', c='#111827', linewidths=1.8, zorder=5)
                        if r.get('evaluation_completed') and not r.get('resolved'):
                            ax.scatter(x, y, s=105, marker='x', c='#d92d20', linewidths=2.1, zorder=6)
                    mean = float(np.mean(ys))
                    ax.plot([xi+model_offset[model]-0.11, xi+model_offset[model]+0.11], [mean, mean],
                            color=ARM_COLOR[arm], linewidth=3.0, solid_capstyle='round', zorder=4)
            if logscale and any(float(r.get(metric) or 0) > 0 for r in subset):
                ax.set_yscale('log')
            ax.set_xticks(range(len(ARMS)), [ARM_SHORT[a] for a in ARMS], rotation=25, ha='right')
            ax.set_title(title)
            if ci == 0:
                ax.set_ylabel(source['label'] + '\nValue (log scale)' if logscale else source['label'] + '\nValue')
            if ri == 0:
                ax.tick_params(labelbottom=False)
            ax.spines[['top', 'right']].set_visible(False)
    treatment_handles = [Line2D([], [], marker='o', linestyle='None', color=ARM_COLOR[a], markersize=8, label=ARM_LABEL[a]) for a in ARMS]
    model_handles = [Line2D([], [], marker=MODEL_MARKER[m], linestyle='None', color='#344054', markerfacecolor='#98a2b3', markersize=8, label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    status_handles = [Line2D([], [], marker='x', linestyle='None', color='#d92d20', markersize=8, label='Unresolved / empty patch'), Line2D([], [], marker='x', linestyle='None', color='#111827', markersize=7, label='Protocol-invalid run')]
    fig.legend(handles=treatment_handles+model_handles+status_handles, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=7, frameon=False)
    fig.suptitle('Combined run outcomes and resource use', y=1.07, fontsize=17, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=0.9, h_pad=0.8)
    savefig(fig, 'combined_outcomes')


def mixed_effect_summary(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit crossed random-intercept models with study and task blocking.

    The response is log2(Ponytail / baseline). Variance components are estimated
    by a small REML grid search using only NumPy, then the intercept and its
    95% Wald interval are obtained by GLS.
    """
    out=[]
    for model in MODEL_ORDER:
        for metric,_ in [('total_tokens','Total tokens'),('tool_calls','Tool calls')]:
            d=[r for r in paired if r['model']==model and r['metric']==metric]
            if not d: continue
            y=np.array([r['log2_ratio'] for r in d],float); n=len(y)
            studies=sorted({r['study_id'] for r in d}); tasks=sorted({r['task'] for r in d})
            Zs=np.array([[float(r['study_id']==x) for x in studies] for r in d])
            Zt=np.array([[float(r['task']==x) for x in tasks] for r in d])
            best=None
            # REML profile over random-effect and residual variance components.
            for a in np.logspace(-4,2,25):
                for b in np.logspace(-4,2,25):
                    for e in np.logspace(-4,2,25):
                        V=e*np.eye(n)+a*(Zs@Zs.T)+b*(Zt@Zt.T)
                        sign,ld=np.linalg.slogdet(V)
                        if sign<=0: continue
                        try: Vi=np.linalg.inv(V)
                        except np.linalg.LinAlgError: continue
                        X=np.ones((n,1)); XtViX=float((X.T@Vi@X)[0,0]); beta=float((X.T@Vi@y)[0]/XtViX)
                        resid=y-beta; q=float(resid.T@Vi@resid)
                        # Restricted likelihood up to constants.
                        val=.5*(ld+math.log(XtViX)+q+(n-1)*math.log(2*math.pi))
                        if best is None or val<best[0]: best=(val,beta,1/XtViX,a,b,e)
            _,beta,var,a,b,e=best
            se=math.sqrt(var); lo=beta-1.96*se; hi=beta+1.96*se
            out.append({'model':model,'model_short':MODEL_SHORT[model],'metric':metric,
                'n_pairs':n,'study_blocks':len(studies),'task_blocks':len(tasks),
                'estimate_log2_ratio':beta,'geometric_mean_ratio':2**beta,
                'mean_percent_change':(2**beta-1)*100,
                'ci95_low_percent_change':(2**lo-1)*100,'ci95_high_percent_change':(2**hi-1)*100,
                'study_variance':a,'task_variance':b,'residual_variance':e})
    return out

def mixed_level_estimates(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Estimate log1p metric means with study/task random-intercept blocking."""
    out=[]
    for model in MODEL_ORDER:
        for arm in ARMS:
            d=[r for r in rows if r.get('model')==model and r.get('arm')==arm and r.get('completed') and r.get('valid') and r.get(metric) is not None]
            if not d: continue
            y=np.log1p(np.array([float(r[metric]) for r in d],float)); n=len(y)
            studies=sorted({r['study_id'] for r in d}); tasks=sorted({r['task'] for r in d})
            Zs=np.array([[float(r['study_id']==x) for x in studies] for r in d]); Zt=np.array([[float(r['task']==x) for x in tasks] for r in d])
            best=None
            for a in np.logspace(-4,2,25):
                for b in np.logspace(-4,2,25):
                    for e in np.logspace(-4,2,25):
                        V=e*np.eye(n)+a*(Zs@Zs.T)+b*(Zt@Zt.T); sign,ld=np.linalg.slogdet(V)
                        if sign<=0: continue
                        try: Vi=np.linalg.inv(V)
                        except np.linalg.LinAlgError: continue
                        X=np.ones((n,1)); XtViX=float((X.T@Vi@X)[0,0]); beta=float((X.T@Vi@y)[0]/XtViX)
                        resid=y-beta; q=float(resid.T@Vi@resid); val=.5*(ld+math.log(XtViX)+q+(n-1)*math.log(2*math.pi))
                        if best is None or val<best[0]: best=(val,beta,1/XtViX,a,b,e)
            _,beta,var,a,b,e=best; se=math.sqrt(var)
            out.append({'model':model,'model_short':MODEL_SHORT[model],'arm':arm,'metric':metric,'n':n,
                'estimate_log1p':beta,'estimate':math.expm1(beta),'ci_low':math.expm1(beta-1.96*se),'ci_high':math.expm1(beta+1.96*se),
                'study_variance':a,'task_variance':b,'residual_variance':e})
    return out

def focused_comparison_figure(rows: list[dict[str, Any]]) -> None:
    """Compare the three headline configurations across pooled runs."""
    configs=[
        ('v41_base','V4.1 · Baseline','deepseek/deepseek-v4.1-flash-20260910','baseline'),
        ('v4_base','V4 · Baseline','deepseek/deepseek-v4-flash-0731','baseline'),
        ('v4_pony','V4 · Ponytail','deepseek/deepseek-v4-flash-0731','ponytail'),
    ]
    metrics=['input_tokens','output_tokens','cache_read_tokens','cost_usd','wall_seconds']
    estimates=[]
    for key,label,model,arm in configs:
        subset=[r for r in rows if r.get('model')==model and r.get('arm')==arm]
        for metric in metrics:
            e=mixed_level_estimates(subset,metric)
            if e:
                v=e[0]; v.update({'config':key,'config_label':label}); estimates.append(v)
    write_csv(OUT/'focused_configuration_estimates.csv',estimates)
    labels=[x[1] for x in configs]; x=np.arange(len(configs))
    fig,axes=plt.subplots(1,3,figsize=(18,7))
    colors={'input_tokens':'#667085','output_tokens':'#f79009','cache_read_tokens':'#0891b2'}
    token_labels={'input_tokens':'Input','output_tokens':'Output','cache_read_tokens':'Cached'}
    token_bottom=np.zeros(len(configs))
    for metric in ('input_tokens','output_tokens','cache_read_tokens'):
        vals=[]; lows=[]; highs=[]
        for key,_,_,_ in configs:
            e=next((z for z in estimates if z['config']==key and z['metric']==metric),None)
            vals.append(e['estimate'] if e else np.nan); lows.append(e['ci_low'] if e else np.nan); highs.append(e['ci_high'] if e else np.nan)
        vals=np.array(vals); ax=axes[0]; ax.bar(x,vals,bottom=token_bottom,color=colors[metric],label=token_labels[metric],edgecolor='white',linewidth=.8)
        # Component uncertainty is available in the CSV; stack bars show estimates without implying independent total CI.
        token_bottom += vals
    axes[0].set_title('Mixed token use',fontsize=16); axes[0].set_ylabel('Estimated tokens',fontsize=13); axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3,fontsize=12)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v/1e6:.1f}M')); axes[0].grid(axis='y',alpha=.35)
    for ax,metric,title,fmt in [(axes[1],'cost_usd','Cost','${:.3f}'),(axes[2],'wall_seconds','Speed (wall time)','{:.0f}s')]:
        vals=[]; lo=[]; hi=[]
        for key,_,_,_ in configs:
            e=next((z for z in estimates if z['config']==key and z['metric']==metric),None)
            vals.append(e['estimate'] if e else np.nan); lo.append(e['ci_low'] if e else np.nan); hi.append(e['ci_high'] if e else np.nan)
        vals=np.array(vals); lo=np.maximum(np.array(lo), 0.0); hi=np.array(hi); ax.errorbar(x,vals,yerr=[vals-lo,hi-vals],fmt='none',ecolor='#344054',capsize=6,lw=2.2,zorder=2)
        ax.scatter(x,vals,s=210,c=['#7c3aed','#f79009','#0891b2'],edgecolors='#101828',linewidths=.8,zorder=3)
        if metric in ('cost_usd','wall_seconds') and vals[1] > 0:
            decline=(vals[1]-vals[2])/vals[1]*100
            ax.text(.995,.97,f'V4 Ponytail:\n{decline:.0f}% lower',transform=ax.transAxes,ha='right',va='top',fontsize=13,fontweight='bold',color='#087f5b',bbox={'boxstyle':'round,pad=.3','fc':'white','ec':'#087f5b','alpha':.95})
        ax.set_title(title,fontsize=16); ax.set_ylabel('Estimated USD' if metric=='cost_usd' else 'Estimated seconds',fontsize=13); ax.grid(axis='y',alpha=.35)
        if metric=='cost_usd':
            ax.set_ylim(0, .05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'${v:.3f}'))
        else:
            ax.set_ylim(0, 360)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v:.0f}s'))
    for ax in axes:
        ax.set_xticks(x,labels,rotation=0,ha='center',fontsize=13); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Focused comparison: baseline V4.1, baseline V4, and Ponytail V4',y=.995,fontsize=20,fontweight='bold')
    fig.text(.5,.015,'Pooled mixed-effects estimates with study and task random-intercept blocking. Lower cost and wall time are better; token bars show input, output, and cached components.',ha='center',fontsize=12,color='#475467')
    fig.tight_layout(rect=(0,.06,1,.90),w_pad=2); savefig(fig,'focused_configuration_comparison')

def interaction_figure(rows: list[dict[str, Any]]) -> None:
    metrics=[('total_tokens','Total tokens'),('tool_calls','Tool calls'),('cost_usd','Cost (USD)'),('wall_seconds','Wall time (seconds)')]
    estimates=[]
    for metric,_ in metrics: estimates.extend(mixed_level_estimates(rows,metric))
    write_csv(OUT/'mixed_interaction_estimates.csv',estimates)
    fig,axes=plt.subplots(2,2,figsize=(18,11))
    treatment_colors={'baseline':'#667085','caveman':'#f79009','ponytail':'#0891b2','caveman_ponytail':'#7c3aed'}
    treatment_labels={'baseline':'Baseline Hermes','caveman':'Caveman','ponytail':'Ponytail','caveman_ponytail':'Caveman + Ponytail'}
    model_style={MODEL_ORDER[0]:('#172033','o','-','V4 Flash 0731'),MODEL_ORDER[1]:('#475467','s','--','V4.1 Flash')}
    x=np.arange(len(ARMS))
    for ax,(metric,title) in zip(axes.flat,metrics):
        all_hi=[]
        for model in MODEL_ORDER:
            vals=[next((e for e in estimates if e['model']==model and e['arm']==arm and e['metric']==metric),None) for arm in ARMS]
            y=np.array([v['estimate'] if v else np.nan for v in vals]); lo=np.array([v['ci_low'] if v else np.nan for v in vals]); hi=np.array([v['ci_high'] if v else np.nan for v in vals])
            all_hi.extend([v for v in hi if np.isfinite(v)])
            color,marker,ls,label=model_style[model]
            ax.plot(x,y,color=color,marker=marker,linestyle=ls,lw=3.8,ms=13,label=label,zorder=3)
            for i,v in enumerate(vals):
                if v: ax.scatter(i,v['estimate'],s=180,c=treatment_colors[ARMS[i]],marker=marker,edgecolors='#101828',linewidths=.8,zorder=4)
        ax.set_xticks(x,['Baseline','Caveman','Ponytail','Both'],rotation=0,ha='center',fontsize=14)
        ax.set_title(title,fontsize=16); ax.set_ylabel('Mixed-effects estimated value',fontsize=13)
        ax.grid(axis='y',alpha=.35); ax.spines[['top','right']].set_visible(False)
        if metric=='cost_usd':
            ax.set_ylim(0, .05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'${v:.3f}'))
        elif metric=='wall_seconds':
            ax.set_ylim(0, 360)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v:.0f}s'))
        elif metric=='total_tokens':
            ax.set_ylim(0, 3_000_000)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v/1e6:.1f}M'))
        elif metric=='tool_calls':
            ax.set_ylim(0, 90)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v:.0f}'))
    axes[0,0].legend(loc='upper left',frameon=True,fontsize=13,handlelength=2.8)
    fig.suptitle('Model × treatment interaction: pooled mixed-effects estimates',y=.995,fontsize=20,fontweight='bold')
    fig.text(.5,.015,'Linear axes. Lines are models; marker color is treatment. Estimates use log1p values with study and task random-intercept blocking. Uncertainty estimates remain in the accompanying CSV files.',ha='center',fontsize=12,color='#475467')
    fig.tight_layout(rect=(0,.05,1,.94),h_pad=1.8,w_pad=1.4); savefig(fig,'combined_interactions')


def ponytail_effect_figure(rows: list[dict[str, Any]]) -> None:
    """Plot paired Ponytail-vs-baseline usage changes for each task and study."""
    metric_specs=[('total_tokens','Total tokens'),('tool_calls','Tool calls')]
    study_order=[s['id'] for s in SOURCES]
    study_style={SOURCES[0]['id']:{'color':'#35445c','marker':'o','label':'Previous study'},SOURCES[1]['id']:{'color':'#b45309','marker':'s','label':'Current study'}}
    tasks=sorted({r['task'] for r in rows})
    task_label={
        'django__django-13516':'Django\n13516',
        'pytest-dev__pytest-7571':'Pytest\n7571',
        'django__django-15731':'Django\n15731',
        'django__django-16662':'Django\n16662',
        'django__django-7530':'Django\n7530',
    }
    paired=[]; summaries=[]
    tcrit={2:4.303,3:3.182,4:2.776}
    for model in MODEL_ORDER:
        for source in SOURCES:
            valid={(r['task'],r['arm']):r for r in rows if r['study_id']==source['id'] and r['model']==model and r.get('completed') and r.get('valid')}
            for metric,_ in metric_specs:
                values=[]
                for task in tasks:
                    baseline=valid.get((task,'baseline')); ponytail=valid.get((task,'ponytail'))
                    if not baseline or not ponytail: continue
                    b=baseline.get(metric); p=ponytail.get(metric)
                    if b is None or p is None or float(b)<=0 or float(p)<=0: continue
                    ratio=float(p)/float(b)
                    logratio=math.log2(ratio)
                    record={'study_id':source['id'],'study_label':source['label'],'model':model,'model_short':MODEL_SHORT[model],
                            'metric':metric,'task':task,'baseline_value':float(b),'ponytail_value':float(p),
                            'ponytail_to_baseline_ratio':ratio,'percent_change':(ratio-1)*100,'log2_ratio':logratio,
                            'baseline_valid':bool(baseline.get('valid')),'ponytail_valid':bool(ponytail.get('valid'))}
                    paired.append(record);values.append(logratio)
                n=len(values)
                if n:
                    center=float(np.mean(values));sem=float(np.std(values,ddof=1)/math.sqrt(n)) if n>1 else 0.0
                    crit=tcrit.get(n-1,1.96) if n>1 else 0.0
                    low=center-crit*sem;high=center+crit*sem
                    summaries.append({'study_id':source['id'],'study_label':source['label'],'model':model,'model_short':MODEL_SHORT[model],
                                      'metric':metric,'n_paired_tasks':n,'n_lower_usage_tasks':sum(v<0 for v in values),'geometric_mean_ratio':2**center,
                                      'mean_percent_change':(2**center-1)*100,'ci95_low_percent_change':(2**low-1)*100,
                                      'ci95_high_percent_change':(2**high-1)*100})
    write_csv(OUT/'ponytail_effect_by_task.csv',paired)
    write_csv(OUT/'ponytail_effect_summary.csv',summaries)
    pooled_summaries=[]
    for model in MODEL_ORDER:
        for metric,_ in metric_specs:
            values=[r['log2_ratio'] for r in paired if r['model']==model and r['metric']==metric]
            n=len(values)
            if not n: continue
            center=float(np.mean(values)); sem=float(np.std(values,ddof=1)/math.sqrt(n)) if n>1 else 0.0
            crit=tcrit.get(n-1,1.96) if n>1 else 0.0
            low=center-crit*sem; high=center+crit*sem
            pooled_summaries.append({'study_id':'pooled','study_label':'Both studies pooled','model':model,'model_short':MODEL_SHORT[model],
                'metric':metric,'n_paired_tasks':n,'n_lower_usage_tasks':sum(v<0 for v in values),'geometric_mean_ratio':2**center,
                'mean_percent_change':(2**center-1)*100,'ci95_low_percent_change':(2**low-1)*100,
                'ci95_high_percent_change':(2**high-1)*100})
    write_csv(OUT/'ponytail_effect_pooled_summary.csv',pooled_summaries)
    mixed_summaries=mixed_effect_summary(paired)
    write_csv(OUT/'ponytail_effect_mixed_summary.csv',mixed_summaries)
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    model_titles={MODEL_ORDER[0]:'DeepSeek V4 Flash 0731 · Relace in both studies',MODEL_ORDER[1]:'DeepSeek V4.1 Flash · Relace previously, DeepSeek currently'}
    # The vertical coordinate is log2(Ponytail / baseline), with ticks printed as percent change.
    tick_percent=[-90,-75,-50,-25,0,50,100,300,700,1500]
    tick_positions=[math.log2(1+p/100) for p in tick_percent]
    for ri,model in enumerate(MODEL_ORDER):
        for ci,(metric,title) in enumerate(metric_specs):
            ax=axes[ri,ci];sub=[r for r in paired if r['model']==model and r['metric']==metric]
            for task_i,task in enumerate(tasks):
                for source_i,source in enumerate(SOURCES):
                    obs=next((r for r in sub if r['task']==task and r['study_id']==source['id']),None)
                    if not obs: continue
                    style=study_style[source['id']]
                    x=task_i+(-0.09 if source_i==0 else 0.09)
                    ax.scatter(x,obs['log2_ratio'],s=92,marker=style['marker'],c=style['color'],edgecolors='white',linewidths=1.0,zorder=4)
            summary=next((s for s in mixed_summaries if s['model']==model and s['metric']==metric),None)
            if summary:
                color='#344054'
                center=summary['estimate_log2_ratio']
                low=math.log2(1+summary['ci95_low_percent_change']/100)
                high=math.log2(1+summary['ci95_high_percent_change']/100)
                ax.axhspan(low,high,color=color,alpha=.06,zorder=1)
                ax.axhline(center,color=color,lw=2.4,ls='--',zorder=2)
                pct=summary['mean_percent_change']
                stat=f"Mixed pooled: {pct:+.0f}% (95% CI {summary['ci95_low_percent_change']:+.0f} to {summary['ci95_high_percent_change']:+.0f})"
                ax.text(.02,.98,stat,transform=ax.transAxes,ha='left',va='top',fontsize=8.2,color=color,
                        bbox={'boxstyle':'round,pad=.22','fc':'white','ec':color,'alpha':.9,'lw':.7})
            ax.axhline(0,color='#101828',lw=1.2,zorder=2)
            ax.set_title(f"{MODEL_SHORT[model]} · {title}",fontsize=11)
            ax.set_xticks(range(len(tasks)),[task_label.get(t,t) for t in tasks],fontsize=8)
            ax.grid(axis='y',alpha=.42);ax.spines[['top','right']].set_visible(False)
            values=[r['log2_ratio'] for r in sub]
            if values:
                matching=[s for s in pooled_summaries if s['model']==model and s['metric']==metric]
                lows=[math.log2(1+s['ci95_low_percent_change']/100) for s in matching]
                highs=[math.log2(1+s['ci95_high_percent_change']/100) for s in matching]
                lo=min(values+lows); hi=max(values+highs)
                margin=max(.25,(hi-lo)*.12)
                ax.set_ylim(lo-margin,hi+margin)
            visible=[(pos,label) for pos,label in zip(tick_positions,tick_percent) if ax.get_ylim()[0]<=pos<=ax.get_ylim()[1]]
            ax.set_yticks([x[0] for x in visible],[f'{x[1]:+d}%' for x in visible])
            if ci==0: ax.set_ylabel('Ponytail vs baseline\n(% change)')
    legend=[Line2D([],[],marker=study_style[s['id']]['marker'],linestyle='None',color=study_style[s['id']]['color'],markerfacecolor=study_style[s['id']]['color'],markersize=8,label=study_style[s['id']]['label']) for s in SOURCES] + [Line2D([],[],linestyle='--',color='#344054',lw=2,label='Mixed-effects pooled estimate')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,1.015),ncol=2,frameon=False)
    fig.suptitle('Ponytail effect by task: paired with each task’s baseline',y=1.06,fontsize=16,fontweight='bold')
    fig.text(.5,.012,'Each point is a task pair from one study; below 0 means lower use with Ponytail. Colored points identify the study. The dark dashed line and shaded band are a crossed mixed-effects estimate pooling both studies while blocking study and task. Ticks show percent change on a log-ratio scale. V4.1 used different providers across studies.',ha='center',fontsize=9,color='#475467')
    fig.tight_layout(rect=(0,.045,1,.95),h_pad=1.5,w_pad=1.3)
    savefig(fig,'combined_ponytail_effects')


def resolution_figure(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    offset = {MODEL_ORDER[0]: -0.12, MODEL_ORDER[1]: 0.12}
    for ax, source in zip(axes, SOURCES):
        rs = [r for r in rows if r['study_id'] == source['id']]
        for xi, arm in enumerate(ARMS):
            for model in MODEL_ORDER:
                group = [r for r in rs if r['arm'] == arm and r['model'] == model]
                n = len(group)
                solved = sum(bool(r.get('resolved')) for r in group)
                rate = solved/n if n else math.nan
                x = xi + offset[model]
                ax.scatter(x, rate*100, s=115, marker=MODEL_MARKER[model], c=ARM_COLOR[arm], edgecolors='white', linewidths=1.0, zorder=3)
                if n:
                    ax.annotate(f'{solved}/{n}', (x, rate*100), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(range(4), [ARM_SHORT[a] for a in ARMS], rotation=20, ha='right')
        ax.set_ylim(-5, 108); ax.set_yticks([0,20,40,60,80,100]); ax.grid(axis='y')
        ax.set_title(source['label'], fontsize=11)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Resolved runs (% of 5 assigned cells)')
    handles=[Line2D([],[],marker=MODEL_MARKER[m],linestyle='None',color='#344054',markerfacecolor='#98a2b3',markersize=9,label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    for a in ARMS: handles.append(Line2D([],[],marker='o',linestyle='None',color=ARM_COLOR[a],markersize=8,label=ARM_LABEL[a]))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=6,frameon=False)
    fig.suptitle('SWE-bench resolution by study, model, and treatment', y=1.10, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0,0,1,0.93))
    savefig(fig,'combined_resolution')


def pca_fit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[str]]:
    features = ['total_tokens','wall_seconds','api_calls','tool_calls','cost_usd','diff_lines']
    selected=[]; matrix=[]
    for row in rows:
        if not row.get('completed'):
            continue
        vals=[]
        okay=True
        for f in features:
            try: value=float(row.get(f))
            except (TypeError,ValueError): okay=False; break
            if not math.isfinite(value) or value < 0: okay=False; break
            vals.append(value)
        if okay:
            selected.append(row); matrix.append(vals)
    X=np.asarray(matrix,dtype=float)
    # Log transform right-skewed count/cost features, then standardize once over both studies.
    for j,f in enumerate(features):
        if f in ('total_tokens','wall_seconds','api_calls','tool_calls','cost_usd','diff_lines'):
            X[:,j]=np.log1p(X[:,j])
    mean=X.mean(axis=0); std=X.std(axis=0,ddof=0); std[std==0]=1.0
    Z=(X-mean)/std
    u,s,vt=np.linalg.svd(Z,full_matrices=False)
    scores=u*s
    loadings=vt.T.copy()
    for j in range(min(2,loadings.shape[1])):
        anchor=int(np.argmax(np.abs(loadings[:,j])))
        if loadings[anchor,j] < 0:
            loadings[:,j]*=-1; scores[:,j]*=-1
    variance=(s*s)/(s*s).sum()
    score_rows=[]
    for row,score in zip(selected,scores):
        score_rows.append({k:row.get(k) for k in ('study_id','study_label','study_date','source_run_id','source_run_path','run_index','task','model','model_short','upstream_provider','arm','treatment_label','valid','resolved','evaluation_completed','total_tokens','tool_calls','api_calls','cost_usd','wall_seconds')} | {'pc1':float(score[0]),'pc2':float(score[1])})
    loading_rows=[{'feature':f,'pc1_loading':float(loadings[i,0]),'pc2_loading':float(loadings[i,1])} for i,f in enumerate(features)]
    return score_rows,loading_rows,[float(x) for x in variance],features


def box_overlap(a: Any,b: Any) -> float:
    x0=max(a.x0,b.x0); y0=max(a.y0,b.y0); x1=min(a.x1,b.x1); y1=min(a.y1,b.y1)
    return max(0.0,x1-x0)*max(0.0,y1-y0)


def place_centroid_labels(ax: Any, centers: list[dict[str, Any]], study_edge: str) -> None:
    offsets=[(32,24),(32,-24),(-32,24),(-32,-24),(0,42),(0,-42),(52,0),(-52,0),(58,34),(58,-34),(-58,34),(-58,-34),(0,64),(0,-64),(76,0),(-76,0)]
    occupied=[]
    renderer=ax.figure.canvas.get_renderer()
    for c in sorted(centers,key=lambda z:(z['pc2'],z['pc1'])):
        model=c['model']; arm=c['arm']; label=f"{MODEL_SHORT[model]} · {ARM_SHORT[arm]}"
        best=None; best_score=None
        for dx,dy in offsets:
            ann=ax.annotate(label,(c['pc1'],c['pc2']),xytext=(dx,dy),textcoords='offset points',ha='center',va='center',fontsize=7.5,
                            bbox={'boxstyle':'round,pad=0.22','fc':'white','ec':ARM_COLOR[arm],'lw':1.1,'alpha':0.97},
                            arrowprops={'arrowstyle':'-','color':ARM_COLOR[arm],'lw':0.75,'shrinkA':3,'shrinkB':8},zorder=7,annotation_clip=False)
            ax.figure.canvas.draw()
            bbox=ann.get_window_extent(renderer).expanded(1.04,1.08)
            overlap=sum(box_overlap(bbox,old) for old in occupied)
            axes_box=ax.get_window_extent(renderer)
            outside=max(0,axes_box.x0-bbox.x0)+max(0,bbox.x1-axes_box.x1)+max(0,axes_box.y0-bbox.y0)+max(0,bbox.y1-axes_box.y1)
            score=overlap*1000+outside*100+math.hypot(dx,dy)*0.01
            ann.remove()
            if best_score is None or score<best_score:
                best_score=score;best=(dx,dy,bbox)
            if overlap==0 and outside==0: break
        dx,dy,bbox=best
        ann=ax.annotate(label,(c['pc1'],c['pc2']),xytext=(dx,dy),textcoords='offset points',ha='center',va='center',fontsize=7.5,
                        bbox={'boxstyle':'round,pad=0.22','fc':'white','ec':ARM_COLOR[arm],'lw':1.1,'alpha':0.97},
                        arrowprops={'arrowstyle':'-','color':ARM_COLOR[arm],'lw':0.75,'shrinkA':3,'shrinkB':8},zorder=7,annotation_clip=False)
        occupied.append(bbox)


def pca_figures(rows: list[dict[str, Any]]) -> None:
    """PCA for the current combined study with raw points and a loading biplot."""
    current_id=SOURCES[1]['id']; current_rows=[r for r in rows if r.get('study_id')==current_id]
    scores,loadings,variance,features=pca_fit(current_rows)
    write_csv(OUT/'combined_pca_scores.csv',scores)
    write_csv(OUT/'combined_pca_loadings.csv',loadings)
    write_csv(OUT/'combined_pca_explained_variance.csv',[{'component':i+1,'explained_variance_ratio':v} for i,v in enumerate(variance)])
    write_csv(OUT/'combined_pca_centroids.csv',[])
    abs_scores=np.array([abs(v) for r in scores for v in (r['pc1'],r['pc2'])],float)
    lim=max(3.5,float(np.quantile(abs_scores,.97))*1.15)
    fig,ax=plt.subplots(figsize=(12,10))
    source=SOURCES[1]
    for r in scores:
        ax.scatter(r['pc1'],r['pc2'],s=145,marker=MODEL_MARKER[r['model']],c=ARM_COLOR[r['arm']],edgecolors=source['edge'],linewidths=2.0,alpha=.9,zorder=3)
    ax.axhline(0,color='#98a2b3',lw=1.0,alpha=.7); ax.axvline(0,color='#98a2b3',lw=1.0,alpha=.7)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal',adjustable='box'); ax.set_box_aspect(1)
    # Loading vectors are shown as correlations/directions, scaled into the score plane.
    load=np.array([[r['pc1_loading'],r['pc2_loading']] for r in loadings],float)
    max_load=max(float(np.max(np.abs(load))),1e-9); arrow_scale=lim*.72/max_load
    for row,(lx,ly) in zip(loadings,load):
        ex,ey=float(lx*arrow_scale),float(ly*arrow_scale)
        ax.annotate('',xy=(ex,ey),xytext=(0,0),arrowprops={'arrowstyle':'-|>','color':'#c2410c','lw':1.8,'alpha':.8},zorder=2)
        label_offsets={'tool_calls':(0,.20),'wall_seconds':(0,-.20),'cost_usd':(0,-.34)}
        dx,dy=label_offsets.get(row['feature'],(0,0)); ax.text(ex*1.06+dx,ey*1.06+dy,row['feature'],fontsize=11,color='#9a3412',ha='center',va='center',zorder=5,
                bbox={'boxstyle':'round,pad=.18','fc':'white','ec':'#fed7aa','alpha':.9,'lw':.7})
    ax.set_title(f'Current study PCA · {len(scores)} completed runs',fontsize=19,fontweight='bold')
    ax.set_xlabel(f'PC1 ({variance[0]*100:.1f}% variance)',fontsize=14); ax.set_ylabel(f'PC2 ({variance[1]*100:.1f}% variance)',fontsize=14); ax.tick_params(labelsize=12)
    ax.grid(alpha=.32); ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],marker='o',linestyle='None',color=ARM_COLOR[a],markersize=10,label=ARM_LABEL[a]) for a in ARMS]
    handles += [Line2D([],[],marker=MODEL_MARKER[m],linestyle='None',color='#344054',markerfacecolor='#98a2b3',markersize=10,label='Shape: '+MODEL_SHORT[m]) for m in MODEL_ORDER]
    handles.append(Line2D([],[],color='#c2410c',lw=2,marker='>',label='Biplot loading direction'))
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.14),ncol=4,frameon=False,fontsize=10)
    fig.text(.5,.012,'Raw current-study runs only; no treatment/model centroids. Orange arrows show standardized feature loading directions for the first two PCs.',ha='center',fontsize=11,color='#475467')
    fig.tight_layout(rect=(0,.04,1,.90)); savefig(fig,'combined_shared_pca')
    fig,ax=plt.subplots(figsize=(9,4.8)); inds=np.arange(len(features)); width=.36
    ax.barh(inds-width/2,[r['pc1_loading'] for r in loadings],height=width,label='PC1',color='#2478a8'); ax.barh(inds+width/2,[r['pc2_loading'] for r in loadings],height=width,label='PC2',color='#d26a2e')
    ax.set_yticks(inds,features); ax.axvline(0,color='#667085',lw=.8); ax.grid(axis='x',alpha=.4); ax.legend(frameon=False); ax.set_xlabel('Standardized loading'); ax.set_title('Features behind the current-study PCA axes',fontweight='bold'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); savefig(fig,'combined_shared_pca_loadings')


def resolution_data(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for source in SOURCES:
        for model in MODEL_ORDER:
            for arm in ARMS:
                g=[r for r in rows if r['study_id']==source['id'] and r['model']==model and r['arm']==arm]
                resolved=sum(bool(r.get('resolved')) for r in g)
                out.append({'study_id':source['id'],'study_label':source['label'],'model':model,'model_short':MODEL_SHORT[model],'arm':arm,'treatment_label':ARM_LABEL[arm],'assigned_runs':len(g),'agent_completed':sum(bool(r.get('completed')) for r in g),'nonempty_patches':sum(bool(r.get('patch_nonempty')) for r in g),'graded_runs':sum(bool(r.get('evaluation_completed')) for r in g),'resolved_runs':resolved,'resolution_rate':resolved/len(g) if g else math.nan})
    return out


_PUBLISHED_TRACE_CACHE: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}


def published_trace_records(run: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Read wrapped release JSONL, retaining the original event payload."""
    merged_name = TRACE_SOURCES[name]
    if merged_name not in _PUBLISHED_TRACE_CACHE:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in jsonl_rows(OUT / 'traces' / merged_name):
            key = (str(item.get('study_id', '')), str(item.get('source_run_id', '')))
            payload = item.get('record')
            if isinstance(payload, dict):
                grouped[key].append(payload)
        _PUBLISHED_TRACE_CACHE[merged_name] = grouped
    return _PUBLISHED_TRACE_CACHE[merged_name].get(
        (str(run.get('study_id', '')), str(run.get('source_run_id', ''))), []
    )


def trace_records(run: dict[str, Any], name: str) -> list[dict[str, Any]]:
    folder=Path(run['source_run_path'])
    if not folder.is_absolute(): folder=REPO/folder
    events=jsonl_rows(folder/name)
    return events if events else published_trace_records(run, name)


def load_event_times(run: dict[str, Any], name: str) -> tuple[list[int],int|None,int|None]:
    events=trace_records(run, name)
    times=[int(e['captured_at_monotonic_ns']) for e in events if e.get('captured_at_monotonic_ns') is not None]
    if name=='api-events.jsonl': selected=[int(e['captured_at_monotonic_ns']) for e in events if e.get('event')=='post_api_request' and e.get('captured_at_monotonic_ns') is not None]
    else: selected=[int(e['captured_at_monotonic_ns']) for e in events if e.get('event')=='post_tool_call' and e.get('captured_at_monotonic_ns') is not None]
    return selected,min(times) if times else None,max(times) if times else None


def trace_figure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grid=np.linspace(0,1,101)
    grouped=defaultdict(list)
    trace_summary=[]
    for run in rows:
        if not run.get('completed'):
            continue
        api,start_a,end_a=load_event_times(run,'api-events.jsonl')
        tool,start_t,end_t=load_event_times(run,'tool-events.jsonl')
        hermes=trace_records(run, 'hermes-events.jsonl')
        starts=[int(e['captured_at_monotonic_ns']) for e in hermes if e.get('event')=='on_session_start' and e.get('captured_at_monotonic_ns') is not None]
        ends=[int(e['captured_at_monotonic_ns']) for e in hermes if e.get('event')=='on_session_end' and e.get('captured_at_monotonic_ns') is not None]
        start=min(starts) if starts else min([t for t in (start_a,start_t) if t is not None],default=None)
        end=max(ends) if ends else max([t for t in (end_a,end_t) if t is not None],default=None)
        if start is None or end is None or end<=start:
            continue
        for kind,vals in [('API responses',api),('Completed tool calls',tool)]:
            fractions=np.clip((np.asarray(vals,dtype=float)-start)/(end-start),0,1) if vals else np.asarray([])
            counts=np.searchsorted(np.sort(fractions),grid,side='right').astype(float)
            key=(run['study_id'],run['model'],run['arm'],kind)
            grouped[key].append(counts)
            trace_summary.append({'study_id':run['study_id'],'source_run_id':run['source_run_id'],'model':run['model'],'arm':run['arm'],'event_kind':kind,'n_events':len(vals),'duration_ns':end-start,'trajectory_included':True})
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    kinds=['API responses','Completed tool calls']
    for ri,kind in enumerate(kinds):
        for ci,source in enumerate(SOURCES):
            ax=axes[ri,ci]
            for model in MODEL_ORDER:
                for arm in ARMS:
                    ys=grouped.get((source['id'],model,arm,kind),[])
                    if not ys: continue
                    arr=np.vstack(ys); mean=arr.mean(axis=0); q25=np.quantile(arr,.25,axis=0);q75=np.quantile(arr,.75,axis=0)
                    ax.plot(grid*100,mean,color=ARM_COLOR[arm],linestyle='-' if model==MODEL_ORDER[0] else '--',lw=2.0,label=f'{ARM_SHORT[arm]} · {MODEL_SHORT[model]}')
                    ax.fill_between(grid*100,q25,q75,color=ARM_COLOR[arm],alpha=0.07)
            ax.set_title(('Previous' if ci==0 else 'Current')+' · '+kind,fontsize=10)
            ax.set_ylabel('Mean cumulative count');ax.grid(alpha=.4);ax.spines[['top','right']].set_visible(False)
            if ri==1: ax.set_xlabel('Share of agent trace duration (%)')
    treatment_handles=[Line2D([],[],color=ARM_COLOR[a],lw=2,label=ARM_SHORT[a]) for a in ARMS]
    model_handles=[Line2D([],[],color='#475467',lw=2,ls='-' if m==MODEL_ORDER[0] else '--',label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    fig.legend(handles=treatment_handles+model_handles,loc='upper center',bbox_to_anchor=(0.5,1.015),ncol=6,frameon=False)
    fig.suptitle('Activity over normalized Hermes traces · combined raw observer events',y=1.075,fontsize=15,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,0.94),h_pad=1.2,w_pad=1.2)
    savefig(fig,'combined_trace_trajectories')
    write_csv(OUT/'combined_trace_trajectory_inventory.csv',trace_summary)
    return trace_summary


def create_readme(rows: list[dict[str, Any]], metas: list[dict[str, Any]], manifest: list[dict[str, Any]], scores_n: int, trace_inventory: list[dict[str, Any]]) -> None:
    outcomes=resolution_data(rows)
    total=len(rows);complete=sum(bool(r.get('completed')) for r in rows);resolved=sum(bool(r.get('resolved')) for r in rows)
    nonempty=sum(bool(r.get('patch_nonempty')) for r in rows)
    trace_files=sorted({r['merged_file'] for r in manifest if r.get('exists')})
    lines=[
        '# Forge Bench',
        '',
        'Forge Bench is a basic, reproducible tool for comparing AI coding-agent harnesses, models, and providers. It uses experimental design and modern data-science methods to understand long-running agent traces: how interventions change behavior, where tokens and time go, and which pain points are worth optimizing.',
        '',
        'The project is meant to grow beyond this first study. Future work can add more models, providers, tasks, harnesses, and broader execution environments such as Harbor. The aim is to test community assumptions with pinned configurations, raw traces, and analysis code that other researchers can inspect, reproduce, and challenge.',
        '',
        '## Why test token-saving tools?',
        '',
        'People want coding agents to finish useful work with fewer tokens, lower cost, and less waiting. [Caveman](https://github.com/JuliusBrussee/caveman/tree/542442bab314973709f95b85b1ac0b3f6f5b5dc6/skills/caveman) asks agents to communicate more directly and avoid unnecessary response tokens. [Ponytail](https://github.com/DietrichGebert/ponytail/tree/e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156) encourages reuse, native features, and the smallest solution that meets the task.',
        '',
        'Ponytail explicitly suggests using these approaches together. That creates a reasonable experimental hypothesis: if each intervention reduces work, combining them should deliver additive or compounding savings. Forge Bench was built to test assumptions like this instead of accepting them from a tool description or a single successful run.',
        '',
        '## Featured study',
        '',
        'This release combines two five-task SWE-bench Verified studies of a last-generation model and its current-generation successor under four harness conditions: Baseline Hermes, Caveman, Ponytail, and Caveman + Ponytail. The run records and raw traces retain the model, provider, treatment, task, and source-study identity.',
        '',
        '> The short preview: the tools do not save work uniformly, their benefits do not reliably add, and the current-generation model is economical at baseline for a reason that is easy to miss from headline token prices. The figures below show the interaction.',
        '',
        '## The first result: newer is cheaper at baseline; older wins with Ponytail',
        '',
        'The current-generation V4.1 baseline is already cost-effective in this cache-heavy workload. Its non-cached tokens cost more, but cache reads are much cheaper, so its baseline cost is lower than the last-generation V4 baseline.',
        '',
        'The unexpected result comes when the tools enter: Ponytail lowers V4 token use and tool calls, while V4.1 uses more of both with Ponytail. Caveman + Ponytail also does not show a reliable additive gain. The plot puts that interaction first.',
        '',
        '![Model treatment interaction](combined_interactions.png)',
        '',
        'This may reflect a model × tool interaction, provider routing, or differences in trace and task handling. V4 used Relace in both studies; V4.1 used Relace in the earlier study and DeepSeek in the later study. **Community validation is requested:** can others reproduce the pattern with the same tasks and pinned providers, and can trace inspection explain the model-specific behavior?',
        '',
        '## Focused case study: an efficient last-generation configuration',
        '',
        'The focused comparison asks how the current-generation baseline compares with the last-generation baseline and the last-generation model plus Ponytail. Two findings matter:',
        '',
        '- The current-generation baseline is already cost-effective. Its non-cached token prices are higher, but cache reads are much cheaper; this cache-heavy workload makes its baseline cost lower than the last-generation baseline.',
        '- Adding Ponytail to the last-generation model goes further: it costs about **48% less** and takes about **32% less wall time** than the last-generation baseline while also using substantially fewer tokens.',
        '',
        'The current generation therefore looks efficient on its own, yet it does not respond well to the same intervention. That is more informative than a simple old-versus-new ranking.',
        '',
        'Three pooled configurations with stacked token components, cost, and wall-clock time.',
        '',
        '![Focused configuration comparison](focused_configuration_comparison.png)',
        '',
        '### Shared PCA',
        '',
        'PCA uses the current study only, combining its models and treatments. Color encodes treatment and point shape encodes model. Orange arrows are biplot loading directions for the underlying standardized features; no centroids are shown.',
        '',
        '![Shared PCA](combined_shared_pca.png)',
        '',
        '![Shared PCA loadings](combined_shared_pca_loadings.png)',
        '',
        '### Raw trace trajectories',
        '',
        'These curves are derived from merged `api-events.jsonl`, `tool-events.jsonl`, and `hermes-events.jsonl` observer events. Event times are normalized to the start/end of each Hermes session; curves show mean cumulative completed events with the interquartile range shaded.',
        '',
        '![Combined raw trace trajectories](combined_trace_trajectories.png)',
        '',
        '## Experimental design and pooled analysis',
        '',
        'The two five-task studies are complementary randomized blocks. We merge their run records and raw traces while retaining study ID, provider, model, treatment, task, and source path for every row.',
        '',
        'For interaction and focused comparisons, each usage outcome uses a log1p mixed-effects model with study and task random intercepts, then transforms estimates back to original units. The interaction figure omits intervals for readability; the accompanying CSV files retain estimates and uncertainty.',
        '',
        'The model pools descriptive results but cannot separate the V4.1 provider change from the study change, and two study blocks are too few to estimate study-level variation precisely. That limitation is itself a reason to test provider × model and provider × tool interactions directly.',
        '',
        '## What this study suggests',
        '',
        'The results show surprising interactions between models, tools, and providers. Ponytail reduces usage for DeepSeek V4 in this study but does not show the same pattern for V4.1, and the combined Caveman + Ponytail treatment does not reliably add the individual gains. The V4.1 provider change also shows that routing can be part of the behavior being measured. These are starting points for broader community experiments, not universal claims about the tools or models.',
        '',
        '## Install and use Forge Bench',
        '',
        'Forge Bench installs as a normal Python package and provides the `forge-bench` command:',
        '',
        '```bash',
        'git clone https://github.com/AgenticForge-Labs/forge-bench.git',
        'cd forge-bench',
        'uv sync',
        'uv run forge-bench --check-credentials',
        '```',
        '',
        'Plan a design without making model calls, then run it into a named output directory:',
        '',
        '```bash',
        'uv run forge-bench --design designs/your-study.yaml --plan-only',
        'uv run forge-bench --design designs/your-study.yaml --output benchmark-results/my-study',
        '```',
        '',
        'Each output directory contains run metadata, traces, patches, figures, and reports. Analysis scripts can combine studies without discarding model, provider, task, treatment, or source provenance.',
        '',
        '## Data and reproducibility',
        '',
        '- `combined_runs.json` and `combined_runs.csv`: the 80 joined run records with study, model, treatment, provider, source run ID, validity, and grading fields.',
        '- `traces/`: raw JSONL records from both studies, wrapped with run metadata. Includes Hermes events, API events, tool events, lifecycle events, native Hermes session traces/transcripts, usage, timing summaries, treatment evidence, and model patches.',
        '- `source_trace_manifest.csv`: source paths, byte counts, SHA-256 checksums, and line counts for every merged artifact.',
        '- `combined_pca_*.csv`, `combined_resolution.csv`, `mixed_interaction_estimates.csv`, `focused_configuration_estimates.csv`, `ponytail_effect_pooled_summary.csv`, `ponytail_effect_mixed_summary.csv`, `combined_trace_trajectory_inventory.csv`: plotted scores, loadings, outcomes, pooled effects, and trace coverage.',
        '- `analysis.py`: rebuilds the derived tables and figures directly from this release bundle. If the two original source directories are available locally, it can also rebuild the merged export.',
        '',
        'Run from the Forge Bench repository root with Python that has NumPy and Matplotlib installed:',
        '',
        '```bash',
        '.venv/bin/python studies/last-generation-current-generation-tool-interactions/analysis.py',
        '```',
        '',
        'The analysis is descriptive. There is one run per model × treatment × task cell in each study, and the upstream provider changed for V4.1 between studies.',
        '',
    ]
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def write_html_report(rows: list[dict[str, Any]]) -> None:
    total=len(rows)
    completed=sum(bool(r.get('completed')) for r in rows)
    patches=sum(bool(r.get('patch_nonempty')) for r in rows)
    resolved=sum(bool(r.get('resolved')) for r in rows)
    study_meta=[{k:v for k,v in source.items() if k!='dir'} for source in SOURCES]
    figure_info=[
        ('Model × treatment interaction','combined_interactions.png','Pooled model-by-treatment means from mixed-effects estimates; uncertainty is provided in the accompanying CSV files.') ,
        ('Focused configuration comparison','focused_configuration_comparison.png','Baseline V4.1, baseline V4, and Ponytail V4: mixed token components, cost, and wall time.'),
        ('Shared PCA','combined_shared_pca.png','The current study PCA combines its models and treatments. Color marks treatment, shape marks model, and orange arrows show biplot loading directions for the underlying standardized features.'),
        ('PCA loadings','combined_shared_pca_loadings.png','Log transformed, jointly standardized features that contribute to PC1 and PC2.'),
        ('Trace trajectories','combined_trace_trajectories.png','Mean cumulative API responses and completed tool calls across normalized Hermes trace duration; shaded bands show the interquartile range.'),
    ]
    figures='\n'.join(f'<figure><a href="{file}"><img src="{file}" alt="{html_lib.escape(title)}" loading="lazy"></a><figcaption><strong>{html_lib.escape(title)}.</strong> {html_lib.escape(caption)}</figcaption></figure>' for title,file,caption in figure_info)
    tablerows=[]
    for r in resolution_data(rows):
        tablerows.append('<tr>'+''.join(f'<td>{html_lib.escape(str(v))}</td>' for v in (
            r['study_label'],r['model_short'],r['treatment_label'],
            f"{r['resolved_runs']}/{r['assigned_runs']}",
            f"{r['nonempty_patches']}/{r['assigned_runs']}",
            f"{r['agent_completed']}/{r['assigned_runs']}",
        ))+'</tr>')
    table='\n'.join(tablerows)
    source_table=''.join(f'<li><strong>{html_lib.escape(s["label"])}</strong> — <code>{html_lib.escape(relpath(s["dir"]))}</code></li>' for s in SOURCES)
    template='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forge Bench</title>
<style>
:root{color-scheme:light;--ink:#152238;--muted:#52627a;--line:#d9e1ec;--panel:#f7f9fc;--orange:#d26a2e}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{background:#13233b;color:white;padding:48px max(24px,calc((100vw - 1200px)/2)) 40px}header p{max-width:850px;color:#d1d9e6;margin:12px 0 0}
main{max-width:1240px;margin:28px auto;padding:0 22px 60px}h1{font-size:clamp(2rem,4vw,3rem);line-height:1.1;margin:0}h2{margin:38px 0 14px;font-size:1.5rem}h3{margin:0 0 8px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}
.stat{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}.stat b{display:block;font-size:1.6rem}.stat span,.small{color:var(--muted)}.note{border-left:4px solid var(--orange);background:#fff7ed;padding:14px 18px;border-radius:6px;color:#593415}
figure{margin:0 0 24px;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}figcaption{padding:10px 3px 18px;color:var(--muted)}figure img{display:block;width:100%;height:auto;border-radius:6px}
.table-wrap{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;min-width:820px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--panel);font-size:.9rem}tr:last-child td{border-bottom:0}
.links{display:flex;flex-wrap:wrap;gap:10px}.links a{padding:9px 12px;background:white;border:1px solid var(--line);border-radius:8px;color:#155d88;text-decoration:none}.links a:hover{text-decoration:underline}code{overflow-wrap:anywhere}.small{font-size:.92rem}footer{border-top:1px solid var(--line);margin-top:36px;padding-top:18px;color:var(--muted)}
@media(max-width:760px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}header{padding-top:34px}}
</style></head><body>
<header><h1>Forge Bench</h1><p>A reproducible test bench for comparing coding-agent harnesses, models, providers, and the assumptions people make about them.</p></header>
<main>
<section><p>Forge Bench uses experimental design and trace-level analysis to understand long-running agent runs: how interventions change behavior, where tokens and time go, and which pain points are worth optimizing. The project can expand to more models, providers, tasks, harnesses, and broader execution environments such as Harbor.</p><h2>Why test token-saving tools?</h2><p>People want coding agents to finish useful work with fewer tokens, lower cost, and less waiting. <a href="https://github.com/JuliusBrussee/caveman/tree/542442bab314973709f95b85b1ac0b3f6f5b5dc6/skills/caveman">Caveman</a> asks agents to communicate more directly. <a href="https://github.com/DietrichGebert/ponytail/tree/e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156">Ponytail</a> encourages reuse, native features, and smaller solutions. Ponytail explicitly suggests using the approaches together, creating a testable additive-savings hypothesis.</p><h2>Featured study</h2><p>This release combines two five-task SWE-bench Verified studies of a last-generation model and its current-generation successor under Baseline Hermes, Caveman, Ponytail, and Caveman + Ponytail. The short preview: savings depend on the model, the benefits do not reliably add, and the current generation is economical at baseline for a reason that headline token prices obscure.</p></section>
<section><h2>How the two studies are combined</h2><p>Run records and raw traces are merged while retaining study, provider, model, treatment, task, and source provenance. Usage outcomes are modeled on a log1p scale with study and task as random-intercept blocking factors, then transformed back to their original units. The headline plot omits intervals for readability; estimates and uncertainty remain in the CSV files.</p></section>
<div class="stats"><div class="stat"><b>{{total}}</b><span>assigned runs</span></div><div class="stat"><b>{{completed}}</b><span>completed agent runs</span></div><div class="stat"><b>{{patches}}</b><span>nonempty patches</span></div><div class="stat"><b>{{resolved}}</b><span>resolved tasks</span></div></div>
<div class="note"><strong>Comparison caveat:</strong> the previous study routed both model pins through Relace. The current study routed V4 through Relace and V4.1 through DeepSeek. Provider is confounded with study for V4.1. There is one run per task × model × treatment cell in each study.</div>
<section><h2>Results: efficiency depends on the model × tool combination</h2><p><strong>Ponytail lowers usage for the last-generation model but does not play well with the current-generation model in these runs.</strong> The combined Caveman + Ponytail condition also fails to show a reliable additive benefit. This may be a model × tool interaction, a provider effect, or both, which is why the raw traces and provider provenance are part of the release.</p><p><strong>Community validation requested:</strong> can others reproduce the interaction with the same tasks and pinned providers, or explain it through tool selection, context handling, caching, retries, or routing?</p></section>
<section><h2>Focused case study: an efficient last-generation configuration</h2><p>The current-generation baseline is already cost-effective: although its non-cached tokens cost more, much cheaper cache reads dominate this cache-heavy workload. Adding Ponytail to the last-generation model goes further, cutting cost by about <strong>48%</strong> and wall time by about <strong>32%</strong> relative to the last-generation baseline while using substantially fewer tokens.</p><p>The current generation is effective on its own yet less compatible with this intervention. That interaction is more useful than a simple old-versus-new ranking.</p></section>
<h2>Figures</h2>{{figures}}
<h2>Resolution by condition</h2><p class="small">The denominator is all five assigned cells, including timeouts and empty patches.</p><div class="table-wrap"><table><thead><tr><th>Study</th><th>Model</th><th>Treatment</th><th>Resolved</th><th>Nonempty patches</th><th>Agent completed</th></tr></thead><tbody>{{table}}</tbody></table></div>
<section><h2>What this study suggests</h2><p>The results show surprising interactions between models, tools, and providers. A tool that saves work for one model can increase it for another; two efficiency tools do not necessarily compose; and provider routing can change the behavior being measured. These are starting points for broader community experiments, not universal claims.</p></section>
<section><h2>Install and use Forge Bench</h2><p>Forge Bench installs as a normal Python package and provides the <code>forge-bench</code> command:</p><pre><code>git clone https://github.com/AgenticForge-Labs/forge-bench.git
cd forge-bench
uv sync
uv run forge-bench --check-credentials
uv run forge-bench --design designs/your-study.yaml --plan-only
uv run forge-bench --design designs/your-study.yaml --output benchmark-results/my-study</code></pre></section>
<h2>Data and reproducibility</h2><p>The current-study PCA is fitted once over the current study's 40 completed runs. It uses log1p transformed total tokens, wall time, API calls, tool calls, cost, and diff lines, then standardizes features within that study.</p>
<div class="links"><a href="analysis.py">Python analysis</a><a href="combined_runs.csv">Combined run CSV</a><a href="combined_runs.json">Combined run JSON</a><a href="combined_pca_scores.csv">PCA scores</a><a href="source_trace_manifest.csv">Trace provenance and checksums</a><a href="traces/api_events.jsonl">Merged API events</a><a href="traces/tool_events.jsonl">Merged tool events</a><a href="traces/hermes_session_trace.jsonl">Merged Hermes session traces</a></div>
<h3 style="margin-top:24px">Source studies</h3><ul>{{sources}}</ul>
<p class="small">Figures are generated by the included Python script using NumPy and Matplotlib. Run it from the Forge Bench repository root: <code>.venv/bin/python studies/last-generation-current-generation-tool-interactions/analysis.py</code></p>
<footer>Forge Bench experimental comparison · descriptive results, small task panel</footer></main></body></html>'''
    page=(template.replace('{{total}}',str(total)).replace('{{completed}}',str(completed)).replace('{{patches}}',str(patches)).replace('{{resolved}}',str(resolved)).replace('{{figures}}',figures).replace('{{table}}',table).replace('{{sources}}',source_table))
    (OUT/'report.html').write_text(page,encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'traces').mkdir(exist_ok=True)
    source_mode=all((source['dir']/'runs.json').is_file() for source in SOURCES)
    if source_mode:
        for old in (OUT/'traces').glob('*.jsonl'): old.unlink()
        combined,manifest,metas=load_sources()
        write_json(OUT/'combined_runs.json',combined)
        write_csv(OUT/'combined_runs.csv',combined)
        write_json(OUT/'source_studies.json',metas)
        write_csv(OUT/'source_trace_manifest.csv',manifest)
    else:
        combined,manifest,metas=load_published_bundle()
    write_csv(OUT/'combined_resolution.csv',resolution_data(combined))
    theme()
    interaction_figure(combined)
    focused_comparison_figure(combined)
    current_for_pca=[r for r in combined if r.get('study_id')==SOURCES[1]['id']]
    pca_scores,_,_,_=pca_fit(current_for_pca)
    pca_figures(combined)
    trace_inventory=trace_figure(combined)
    create_readme(combined,metas,manifest,len(pca_scores),trace_inventory)
    figures=sorted(p.name for p in OUT.glob('combined_*.png'))
    print(f'Wrote {len(combined)} run records, {len(manifest)} source-artifact manifest entries, {len(pca_scores)} PCA scores, and {len(trace_inventory)} trajectory rows.')
    print('Figures:',', '.join(figures))
    print('Output:',OUT)

if __name__=='__main__':
    main()
