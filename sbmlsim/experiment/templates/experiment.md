[Experiments](index.html)

# {{ exp_id }}

## Model
* **SBML**: [{{ model_path }}]({{ model_path }})
{% if report_path %}
* **HTML**: [{{ report_path }}]({{ report_path }})
{% endif %}

## Datasets
{% for dset_id in datasets %}
* [{{results_path}}/{{ exp_id }}_data_{{ dset_id }}.tsv]({{results_path}}/{{ exp_id }}_data_{{ dset_id }}.tsv)
{% endfor %}

## Simulations
{% for sim_id in simulations %}
* [{{results_path}}/{{ exp_id }}_sim_{{ sim_id }}.h5]({{results_path}}/{{ exp_id }}_sim_{{ sim_id }}.h5)
{% endfor %}

## Scans
{% for scan_id in scans %}
* [{{results_path}}/{{ exp_id }}_scan_{{ scan_id }}.h5]({{results_path}}/{{ exp_id }}_scan_{{ scan_id }}.h5)
{% endfor %}

## Figures
{% for fig_id in figures %}
* [{{results_path}}/{{ exp_id }}_{{ fig_id }}.svg]({{results_path}}/{{ exp_id }}_{{ fig_id }}.svg)
{% endfor %}

{% for fig_id, meta in figures.items() %}
### {{ fig_id }}
{% if meta %}
{% for k, v in meta.items() %}
{% if v %}
**{{ k }}**: {{ v }}  
{% endif %}
{% endfor %}
{% endif %}
![{{results_path}}/{{ exp_id }}_{{ fig_id }}.svg]({{results_path}}/{{ exp_id }}_{{ fig_id }}.svg)
{% endfor %}

## Code
[{{ code_path }}]({{ code_path }})

```python
{{ code }}
```