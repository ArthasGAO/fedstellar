def get_table_styles():
    styles = [
        dict(
            selector="caption",
            props=[('padding', '10px 0px')]
        ),
        dict(
            selector="table",
            props=[("border-collapse", "seperate"), ("border-spacing", "10px")],
        ),
        dict(
            selector="tbody",
            props=[("border-bottom", "2px solid")],
        ),
        dict(
            selector="thead",
            props=[("border-top", "2px solid")],
        ),
        dict(
            selector="thead tr:first-child .col_heading",
            props=[("border-bottom", "0.5px solid"),
                   ("text-align", "left"),],
        ),
        dict(
            selector=".index_name",
            props=[("font-weight", "normal")],
        ),
        dict(
            selector=".row_heading",
            props=[("font-weight", "normal")],
        )
    ]
    return styles