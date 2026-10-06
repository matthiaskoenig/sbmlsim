# Simulation experiments
{% for exp_id, context in data.items() %}

## [{{ exp_id }}]({{ exp_id }}/{{ exp_id }}.md)

{{ context.figures | length }} figure(s), {{ context.models | length }} model(s)
{% for fig_id, fig in context.figures.items() %}

### {{ fig_id }}

{% if fig.static %}
![{{ fig_id }}]({{ exp_id }}/{{ fig.path }}.svg)

{% endif %}
{% if fig.interactive %}
[interactive]({{ exp_id }}/{{ fig.path }}.html)
{% endif %}
{% endfor %}
{% endfor %}
