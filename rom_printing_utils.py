
__all__ = [
    "printColorSameLine",
    "printColor",
    "printPropertyValue",
    "printColorList",
    "printCredits",
    "print_json",
    "printPrettyPropertyValue",
    "printPrettyColor"
]

import shutil
import textwrap

# Simple ANSI color maps (extend as needed)

COLOR_CODES = {
    "black":   "30",
    "red":     "31",
    "green":   "32",
    "yellow":  "33",
    "blue":    "34",
    "magenta": "35",
    "cyan":    "36",
    "white":   "37",
    "pink":    "95",
    "gray":    "90",
    "light_red":    "91",
    "light_green":  "92",
    "light_yellow": "93",
    "light_blue":   "94",
    "light_magenta":"95",
    "light_cyan":   "96",
    "light_white":  "97",
    "bold":            "1",
    "bold_black":      "1;30",
    "bold_red":        "1;31",
    "bold_green":      "1;32",
    "bold_yellow":     "1;33",
    "bold_blue":       "1;34",
    "bold_magenta":    "1;35",
    "bold_cyan":       "1;36",
    "bold_white":      "1;37",
    "bold_pink":       "1;95",
    "bold_gray":       "1;90",
    "bold_light_red":  "1;91",
    "bold_light_green":"1;92",
    "bold_light_yellow":"1;93",
    "bold_light_blue": "1;94",
    "bold_light_cyan": "1;96",
    "bold_light_white":"1;97"
}

RESET = "\033[0m"


def printColorSameLine(text: str, color:str="white"):
    """ funtion to print a text in color WITHOUT newline """
    code = COLOR_CODES.get(color.strip().lower())
    if code:
        print(f"\033[{code}m{text}{RESET}", end="")
    else:
        print(text, end="")
    return


def _color_segment(text: str, color_name: str | None) -> str:
    """Color only this segment (no padding logic)."""
    if not color_name:
        return text
    code = COLOR_CODES.get(color_name.lower())
    if not code:
        return text
    return f"\033[{code}m{text}{RESET}"

def printPrettyColor(
    text: str,
    color: str | None = "white",
    start_column: int = 0,
    term_width: int | None = None,
) -> None:
    """
    Print `text` in `color`, starting at `start_column`, wrapping to terminal width.

    - Every line begins at the same `start_column`.
    - Wrapping is done so that no line exceeds the terminal width.
    """
    if term_width is None:
        term_width = shutil.get_terminal_size(fallback=(80, 20)).columns

    start_column = max(0, start_column)

    # How many characters we can fit per line after the indent
    available = max(1, term_width - start_column)

    # Split text into chunks that fit in the available width
    chunks = textwrap.wrap(text, width=available) or [""]

    indent = " " * start_column
    for chunk in chunks:
        print(indent + _color_segment(chunk, color))

def printPrettyPropertyValue(
    property_name: str,
    property_value,
    property_color: str | None = None,
    value_color: str | None = None,
    name_start_column: int | None = 0,      # absolute column for the name
    property_start_column: int | None = None,  # absolute column for the ':'
    term_width: int | None = None,
) -> None:
    """
    Layout:

      1) Normal case (name fits before property_start_column or no property_start_column):
         <spaces to name_start_column>name<spaces to property_start_column>:(same color) value...

      2) If the name would extend past property_start_column:
         <spaces to name_start_column>name
         <spaces to property_start_column>:(same color) value...

    - name_start_column: absolute column where property_name starts (default 0)
    - property_start_column: absolute column where ':' is printed; if None, it
      is placed one space after the end of the name.
    - value starts after ': ' and is wrapped to the terminal width, with
      continuation lines aligned under the value column.
    """
    if term_width is None:
        term_width = shutil.get_terminal_size(fallback=(80, 20)).columns
      
    if name_start_column is None:
        name_start_column = 0
    name_start_column = max(0, name_start_column)

    if property_start_column is not None:
        property_start_column = max(0, property_start_column)

    name = str(property_name)
    value_str = "" if property_value is None else str(property_value)

    # Name span
    name_end_col = name_start_column + len(name)

    # Decide colon column
    if property_start_column is None:
        # Default: one space after the name
        colon_col = name_end_col + 1
        split_to_next_line = False
    else:
        colon_col = property_start_column
        # If name would “run over” the property_start_column, we move the
        # property to the *next line*
        split_to_next_line = name_end_col > property_start_column

    # Column where the value starts
    value_col = colon_col + 2  # ": "

    # Available space for value
    available = max(1, term_width - value_col)

    # Wrap value
    chunks = textwrap.wrap(value_str, width=available) or [""]

    # ---------------- First line(s) ----------------
    if split_to_next_line:
        # Line 1: name only
        line1 = " " * name_start_column + _color_segment(name, property_color)
        print(line1)

        # Line 2+: colon and value
        first_line = (
            " " * colon_col
            + _color_segment(":", property_color)
            + " "
            + _color_segment(chunks[0], value_color)
        )
        print(first_line)
    else:
        # Single-line layout for the first line: name + spaces + ': ' + value
        spaces_after_name = max(0, colon_col - name_end_col)
        first_line = (
            " " * name_start_column
            + _color_segment(name, property_color)
            + " " * spaces_after_name
            + _color_segment(":", property_color)
            + " "
            + _color_segment(chunks[0], value_color)
        )
        print(first_line)

    # ---------------- Continuation lines ----------------
    indent = " " * value_col
    for chunk in chunks[1:]:
        print(indent + _color_segment(chunk, value_color))
    
    return

def printColor(text: str, color:str="white") -> None:
    """ print a text in the defined color """
    code = COLOR_CODES.get(color.strip().lower())
    if code:
        print(f"\033[{code}m{text}{RESET}")
    else:
        print(text)
    return

def printPropertyValue(property_name:str, property_value:str, color_name:str="white", color_value:str="white")->None:
    """ print a property name and value in color """
    printColorSameLine(property_name + ": ", color_name)
    printColor(property_value, color_value)
    return


#print a list of strings in color
def printColorList(my_list, color:str="white") -> None:
    """ print a list in the defined color """
    for i in range(len(my_list)):
        printColor(my_list[i], color)
    return

def printCredits(program_name:str, version:str, author:str, date:str, copyritght:str, License:str)->None:

    printColor("\n" + program_name + " - Version: " + version, "cyan")
    printColorSameLine("Author      : ", "cyan")
    printColor(author, "blue")
    printColorSameLine("Date        : ", "cyan")
    printColor(date, "blue")
    printColorSameLine("Copyritght  : ", "cyan")
    printColor(copyritght, "blue")
    printColorSameLine("License     : ", "cyan")
    printColor(License, "blue")

    return

def print_json(data, indent=1) -> None:
    """ 
    Print the JSON data in a structured format with indentation. JSON data must be created by the json.load() function.
    """
    TABSTRING:str = " "

    for key, value in data.items():
        printColor(TABSTRING * indent + str(key), "blue")
        if isinstance(value, dict):
            print_json(value, indent + 1)
        else:
            printColor(TABSTRING * (indent + 1) + str(value), "cyan")
    return
