def log(head_str, tail_str, actual_item, total_items, bar_size=20, progress_char=">", void_char="-", end=" "*15, start="\r"):
    # if total_items < bar_size:
    #     bar_size = total_items

    progress = int(actual_item*bar_size / total_items)
    void_char = void_char * (bar_size - progress)
    progress = progress_char * progress
    bar = f"[{progress}{void_char}]"
    print(f"{start}{head_str}  {bar}  {tail_str}", flush=True, end=end)