

# dosing
if dosing is not None:
    # get bodyweight
    if "BW" in changes:
        bodyweight = changes["BW"]
    else:
        bodyweight = r.BW

    set_dosing(r, dosing, bodyweight=bodyweight)
