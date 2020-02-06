# Simulation Experiments
{% for exp_id, context in exp_ids.items() %}
## [{{ exp_id }}]({{ exp_id }}.html)

{% for fig_id, meta in context.figures.items() %}
### {{ fig_id }}
<table><tr><td>
<a href="{{ exp_id }}.html"><img src="{{context.results_path}}/{{ exp_id }}_{{ fig_id }}.svg" width=150/></a>
</td><td>
{% if meta %}
{% for k, v in meta.items() %}
{% if v %}
**{{ k }}**: {{ v }}  
{% endif %}
{% endfor %}
{% endif %}
{% endfor %}
</td></tr>
</table>
{% endfor %}