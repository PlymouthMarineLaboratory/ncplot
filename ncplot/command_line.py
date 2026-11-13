
from ncplot import view
import sys
import re
import os
import glob

def is_url(x):
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return re.match(regex, x) is not None


def main():
    """
    The ``ncplot`` command: view the files, URLs or glob patterns it is given.
    Returns a message (which ``sys.exit`` prints, exiting with status 1) if
    there is nothing to view.
    """
    if len(sys.argv) == 1:
        return "Please supply a file"

    ff = []
    for x in sys.argv[1:]:
        if os.path.exists(x) or is_url(x):
            ff.append(x)
        elif glob.has_magic(x):
            # a pattern the shell did not expand (it was quoted)
            ff += sorted(glob.glob(x))

    if len(ff) == 0:
        return f"{sys.argv[1]} does not exist!"

    if len(ff) == 1:
        view(ff[0])
    else:
        view(ff)


if __name__ == '__main__':
    main()
