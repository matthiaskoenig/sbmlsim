# Simulation Experiments
{% for exp_id, context in exp_ids.items() %}
## [{{ exp_id }}]({{ exp_id }}.html)

{% for fig_id, meta in context.figures.items() %}
### {{ fig_id }}
<a href="{{ exp_d }}.html"><img src="{{context.results_path}}/{{ exp_id }}_{{ fig_id }}.svg" width=150/></a>

{% if meta %}
{% for k, v in meta.items() %}
{% if v %}
**{{ k }}**: {{ v }}  
{% endif %}
{% endfor %}
{% endif %}
{% endfor %}
{% endfor %}