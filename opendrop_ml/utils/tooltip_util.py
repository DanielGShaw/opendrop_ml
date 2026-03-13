import tkinter as tk
import platform


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

    if is_mac:
        tooltip_parent = widget.winfo_toplevel()
        tooltip = tk.Label(
            tooltip_parent,
            text=text,
            bg="#f0f0f0",
            fg="#111111",
            font=("Segoe UI", 10),
            justify="left",
            wraplength=350,
            relief="solid",
            borderwidth=1,
        )

        def enter(event):
            nonlocal hide_job
            if hide_job is not None:
                widget.after_cancel(hide_job)
                hide_job = None
            margin = 4

            widget_x = widget.winfo_rootx()
            widget_y = widget.winfo_rooty()
            widget_w = widget.winfo_width()
            widget_h = widget.winfo_height()

            parent_x = tooltip_parent.winfo_rootx()
            parent_y = tooltip_parent.winfo_rooty()

            tooltip.update_idletasks()
            tooltip_w = tooltip.winfo_width()
            tooltip_h = tooltip.winfo_height()

            if position == "above-align-left":
                rel_x = widget_x
                rel_y = widget_y - tooltip_h - margin
            elif position == "below-align-left":
                rel_x = widget_x
                rel_y = widget_y + widget_h + margin
            elif position == "below-align-right":
                rel_x = widget_x + widget_w - tooltip_w
                rel_y = widget_y + widget_h + margin
            elif position == "below-left-cursor":
                rel_x = event.x_root - tooltip_w - margin
                rel_y = event.y_root + margin
            else:
                # fallback: above-align-left
                rel_x = widget_x
                rel_y = widget_y - tooltip_h - margin

            place_x = rel_x - parent_x
            place_y = rel_y - parent_y

            tooltip.place(x=place_x, y=place_y)
            tooltip.lift()

        def leave(event):
            nonlocal hide_job

            def maybe_hide():
                nonlocal hide_job
                pointer_x, pointer_y = widget.winfo_pointerxy()
                hovered = widget.winfo_containing(pointer_x, pointer_y)
                if is_descendant(hovered, bind_target) or is_descendant(
                    hovered, tooltip
                ):
                    hide_job = None
                    return
                tooltip.place_forget()
                hide_job = None

            if hide_job is not None:
                widget.after_cancel(hide_job)
            hide_job = widget.after(80, maybe_hide)

        bind_target.bind("<Enter>", enter, add="+")
        bind_target.bind("<Leave>", leave, add="+")

    else:
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        tooltip.attributes("-alpha", 0.90)  # semi-transparent

        canvas = tk.Canvas(
            tooltip, background="#e0e0e0", borderwidth=0, highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)

        padding_x, padding_y = 10, 6
        label = tk.Label(
            canvas,
            text=text,
            bg="#e0e0e0",
            fg="#111111",
            font=("Segoe UI", 10),
            justify="left",
            wraplength=250,
            borderwidth=0,
        )
        label.pack(padx=padding_x, pady=padding_y)

        def enter(event):
            nonlocal hide_job
            if hide_job is not None:
                widget.after_cancel(hide_job)
                hide_job = None
            tooltip.update_idletasks()
            tooltip_w = tooltip.winfo_reqwidth()
            tooltip_h = tooltip.winfo_reqheight()
            if position == "below-left-cursor":
                x_pos = event.x_root - tooltip_w - 12
                y_pos = event.y_root + 12
            else:
                x_pos = event.x_root + 12
                y_pos = event.y_root + 12

            screen_w = widget.winfo_screenwidth()
            screen_h = widget.winfo_screenheight()
            x_pos = max(0, min(x_pos, screen_w - tooltip_w))
            y_pos = max(0, min(y_pos, screen_h - tooltip_h))
            tooltip.geometry(f"+{x_pos}+{y_pos}")
            tooltip.deiconify()

        def leave(event):
            nonlocal hide_job

            def maybe_hide():
                nonlocal hide_job
                pointer_x, pointer_y = widget.winfo_pointerxy()
                hovered = widget.winfo_containing(pointer_x, pointer_y)
                if is_descendant(hovered, bind_target) or is_descendant(
                    hovered, tooltip
                ):
                    hide_job = None
                    return
                tooltip.withdraw()
                hide_job = None

            if hide_job is not None:
                widget.after_cancel(hide_job)
            hide_job = widget.after(80, maybe_hide)

        bind_target.bind("<Enter>", enter, add="+")
        bind_target.bind("<Leave>", leave, add="+")
