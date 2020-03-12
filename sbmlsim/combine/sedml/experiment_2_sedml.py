def modelToPython(self, model):
    """ Python code for SedModel.

    :param model: SedModel instance
    :type model: SedModel
    :return: python str
    :rtype: str
    """
    lines = []
    mid = model.getId()
    language = model.getLanguage()
    source = self.model_sources[mid]

    if not language:
        warnings.warn("No model language specified, defaulting to SBML for: {}".format(source))

    def isUrn():
        return source.startswith('urn') or source.startswith('URN')

    def isHttp():
        return source.startswith('http') or source.startswith('HTTP')

    # read SBML
    if 'sbml' in language or len(language) == 0:
        if isUrn():
            lines.append("import tellurium.temiriam as temiriam")
            lines.append("__{}_sbml = temiriam.getSBMLFromBiomodelsURN('{}')".format(mid, source))
            lines.append("{} = te.loadSBMLModel(__{}_sbml)".format(mid, mid))
        elif isHttp():
            lines.append("{} = te.loadSBMLModel('{}')".format(mid, source))
        else:
            lines.append("{} = te.loadSBMLModel(os.path.join(workingDir, '{}'))".format(mid, source))
    # read CellML
    elif 'cellml' in language:
        warnings.warn("CellML model encountered. Tellurium CellML support is very limited.".format(language))
        if isHttp():
            lines.append("{} = te.loadCellMLModel('{}')".format(mid, source))
        else:
            lines.append("{} = te.loadCellMLModel(os.path.join(workingDir, '{}'))".format(mid, self.model_sources[mid]))
    # other
    else:
        warnings.warn("Unsupported model language: '{}'.".format(language))

    # apply model changes
    for change in self.model_changes[mid]:
        lines.extend(SEDMLCodeFactory._apply_model_change(model, change))

    return '\n'.join(lines)