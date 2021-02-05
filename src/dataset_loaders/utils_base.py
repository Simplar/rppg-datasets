import string


def escape_filename(s: str):
    s = s.replace('/', '_').replace('\\', '_').replace(' ', '_')
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    escaped_filename = []
    for ch in s:
        if ch in valid_chars:
            escaped_filename.append(ch)
        else:
            escaped_filename.append("%{0:04x}".format(ord(ch)))
    escaped_filename = ''.join(escaped_filename)
    return escaped_filename
