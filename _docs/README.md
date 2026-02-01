# Create documentation

```bash
cd _docs
```

quarto add machow/quartodoc

# For deployment
```bash
quartodoc build && quarto render
```

# For development
```bash
quartodoc build --watch
quarto preview
```
