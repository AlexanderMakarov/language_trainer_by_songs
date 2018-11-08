import sys, re


class SongLine:
    def __init__(self, text):
        self.text = text.strip()
        if self.text:
            self.words = text.split()
            self.is_delimeter = False
        else:
            self.words = []
            self.is_delimeter = True

    def __str__(self):
        return "---" if self.is_delimeter else self.text


class Song:
    def __init__(self, lines):
        self.lines = lines

    def __str__(self):
        return "\n".join(str(x) for x in self.lines)


def parse_song(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(SongLine(line))
    return Song(lines)


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Path to file with song is not specified."
    song = parse_song(sys.argv[1])
    print(song)
