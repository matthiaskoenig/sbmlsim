[Simulation experiments](../index.md)

# {{ exp_id }}

## Models
{% for model_id, model_path in models.items() %}
* **{{ model_id }}**: [{{ model_path | basename }}]({{ model_path }})
{% endfor %}

## Datasets
{% for dset_id, dset_path in datasets.items() %}
* **{{ dset_id }}**: [{{ dset_path }}]({{ dset_path }})
{% endfor %}

## Figures
{% if error %}

**The experiment failed:** {{ error }}
{% endif %}
{% for fig_id, fig_error in failed_figures.items() %}

**The figure {{ fig_id }} failed:** {{ fig_error }}
{% endfor %}
{% for fig_id, fig in figures.items() %}

### {{ fig_id }}

{% if fig.static %}
![{{ fig_id }}]({{ fig.path }}.svg)

{% endif %}
{% if fig.interactive %}
[{{ fig.path }}.html]({{ fig.path }}.html)
{% endif %}
{% endfor %}

## Code

[{{ code_path | basename }}]({{ code_path }})

````python
{{code}}
````
