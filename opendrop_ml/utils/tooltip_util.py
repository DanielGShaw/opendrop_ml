def create_tooltip(widget, text: str, position="above-align-left"):
    import tkinter as tk
    import platform

    system = platform.system()
    is_mac = system == "Darwin"

    bind_target = widget
    hide_job = None

    def is_descendant(child, ancestor):
        while child is not None:
            if child == ancestor:
                return True
            child = child.master
        return False

    def _cancel_hide():
        nonlocal hide_job
        if hide_job is not None:
            widget.after_cancel(hide_job)
            hide_job = None

    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    if not is_mac:
        tooltip.attributes("-alpha", 0.90)

    frame = tk.Frame(
        tooltip,
        background="#f0f0f0",
        borderwidth=1,
        relief="solid",
    )
    frame.pack(fill="both", expand=True)

    label = tk.Label(
        frame,
        text=text,
        bg="#f0f0f0",
        fg="#111111",
        font=("Segoe UI", 10),
        justify="left",
        wraplength=350 if is_mac else 250,
        borderwidth=0,
    )
    label.pack(padx=4, pady=2)

    def show(event):
        _cancel_hide()
        tooltip.update_idletasks()
        tooltip_w = tooltip.winfo_reqwidth()
        tooltip_h = tooltip.winfo_reqheight()

        widget_x = widget.winfo_rootx()
        widget_y = widget.winfo_rooty()
        widget_w = widget.winfo_width()
        widget_h = widget.winfo_height()

        margin = 4
        if position == "above-align-left":
            x_pos = widget_x
            y_pos = widget_y - tooltip_h - margin
        elif position == "below-align-left":
            x_pos = widget_x
            y_pos = widget_y + widget_h + margin
        elif position == "below-align-right":
            x_pos = widget_x + widget_w - tooltip_w
            y_pos = widget_y + widget_h + margin
        elif position == "below-left-cursor":
            x_pos = event.x_root - tooltip_w - margin
            y_pos = event.y_root + margin
        else:
            x_pos = widget_x
            y_pos = widget_y - tooltip_h - margin

        # Keep tooltip on screen
        screen_w = widget.winfo_screenwidth()
        screen_h = widget.winfo_screenheight()
        x_pos = max(0, min(x_pos, screen_w - tooltip_w))
        y_pos = max(0, min(y_pos, screen_h - tooltip_h))

        tooltip.geometry(f"+{x_pos}+{y_pos}")
        tooltip.deiconify()

    def schedule_hide(event):
        nonlocal hide_job

        def maybe_hide():
            nonlocal hide_job
            hide_job = None
            pointer_x, pointer_y = widget.winfo_pointerxy()
            hovered = widget.winfo_containing(pointer_x, pointer_y)
            if is_descendant(hovered, bind_target) or is_descendant(hovered, tooltip):
                hide_job = widget.after(120, maybe_hide)
                return
            tooltip.withdraw()

        _cancel_hide()
        hide_job = widget.after(80, maybe_hide)

    def tooltip_enter(event):
        _cancel_hide()

    bind_target.bind("<Enter>", show, add="+")
    bind_target.bind("<Leave>", schedule_hide, add="+")
    tooltip.bind("<Enter>", tooltip_enter, add="+")
    tooltip.bind("<Leave>", schedule_hide, add="+")
