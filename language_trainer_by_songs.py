#!/usr/bin/python3
import sys
import os
import re
import argparse
from typing import List, Tuple, Dict, NamedTuple
from collections import namedtuple
from difflib import SequenceMatcher
# For Unix terminal:
try:
    import termios
except ImportError:
    pass
# For Windows terminal:
try:
    import msvcrt
except ImportError:
    pass


Hint = namedtuple('Hint', ('prefix', 'word_len', 'suffix'))
Progress = namedtuple('Progress', ('position', 'total'))


class SongWord:
    def __init__(self, text: str, word: str, word_start_index: int):
        self.text = text
        self.word = word
        self.word_start_index = word_start_index

    def get_hint(self) -> Hint:
        word_len = len(self.word)
        return Hint(self.text[:self.word_start_index], word_len, self.text[self.word_start_index + word_len:])

    def __str__(self):
        return self.text


class SongLine:
    """
    Splits song line to tokens - words. Punctuation should be avoided because useless for words comprehension.
    but difference between "cats" in "They are cats" and "cat's" in "My cat's toy" is important.
    """
    WORDS_SPLIT_RE = r'[;:,.\s()]+'

    def __init__(self, text: str):
        self.text = text.strip()
        self.words = []
        if self.text:
            self.is_delimiter = False
            # Split words by "split characters". So word - characters between splits.
            word_text_start = 0
            word_start = 0
            for split_match in re.finditer(SongLine.WORDS_SPLIT_RE, self.text):
                split_start, split_end = split_match.span()  # Borders of no-word characters.
                # Skip case when string (already trimmed) started from split character(s). Add to next word.
                if split_start == 0:
                    word_start = split_end
                    continue
                word = self.text[word_start:split_start]
                # Check 'bug' case when word is single '-'.
                if word == '-':
                    # Add ' - ' it to previous word as "split characters".
                    prev_song_word = self.words[-1]
                    prev_song_word.text += self.text[word_text_start:split_end]
                else:
                    # Append to line new SongWord.
                    word_text = self.text[word_text_start:split_end]
                    self.words.append(SongWord(word_text, word, word_start - word_text_start))
                word_text_start = split_end
                word_start = split_end
            # If line doesn't over with split character(s) (i.e. line over with word) then add one more word.
            line_end = len(self.text)
            if word_start < line_end - 1:
                word = self.text[word_start:line_end]
                self.words.append(SongWord(word, word, 0))
        else:
            self.is_delimiter = True

    def build_text(self):
        return "".join(x.text for x in self.words)

    def __str__(self):
        return "---" if self.is_delimiter else "|".join(str(x) for x in self.words)


class Song:
    def __init__(self, lines: List[SongLine]):
        self.lines = lines

    def __str__(self):
        return "\n".join(str(x) for x in self.lines)


class MatcherResult:
    def __init__(self, is_match: bool, is_shift_next: bool, answer: str, matched_text: str, match_desc: str,
                 attempt: int, is_line_end: bool, is_song_end: bool):
        """
        Match result constructor.
        :param is_match: flag that matched.
        :param is_shift_next: flag that matcher was shifted to the next word/line.
        :param answer: received answer which was analyzed.
        :param matched_text: part of song which is matched for now, including first letters as a hint if need.
        :param match_desc: description of match (usually match percent).
        :param attempt: attempt number (starting from 1).
        :param is_line_end: flag that matched line is over.
        :param is_song_end: flag that matched song is over (time to print statistic).
        """
        self.is_match = is_match
        self.is_shift_next = is_shift_next
        self.answer = answer
        self.matched_text = matched_text
        self.match_desc = match_desc
        self.attempt = attempt
        self.is_line_end = is_line_end
        self.is_song_end = is_song_end


class Matcher:
    """
    Stateful song words matcher.
    """

    attempts_number: int

    def __init__(self, song: Song, args: NamedTuple):
        """
        Constructor.
        :param song: Song to match words in.
        :param args: Set of settings. Supported:
            skip_words_len - min length of word to match,
            attempts - attempts number,
            by_lines - match by lines, not by words,
            disable_open_letters_on_attempts - don't open letters after failed attempt.
            tolerance - percent of allowable tolerance.
        """
        if not song:
            raise AttributeError("Please specify song.")
        if not song.lines:
            raise AttributeError("Please specify song with at least 1 line.")
        self.song = song
        self.lines_number = len(self.song.lines)
        self.args = args
        self.line_index = -1
        self.word_index = -1
        self.open_letters_at_start = 0
        self.attempts_number = 0
        self.is_song_end = False
        self._shift_next_line()

    def _shift_next_word(self) -> bool:  # Flag that line is over.
        line = self.get_current_line()
        skip_words_len = self.args.skip_words_len
        for i in range(self.word_index + 1, len(line.words)):
            song_word = line.words[i]
            if len(song_word.word) > skip_words_len:
                self.word_index = i
                break
        else:
            return True
        return False

    def _shift_next_line(self) -> SongLine:
        line = None
        self.word_index = -1  # Prepare word_index for _shift_next_word call.
        for i in range(self.line_index + 1, len(self.song.lines)):
            line = self.song.lines[i]
            if not line.is_delimiter:
                self.line_index = i
                if self._shift_next_word():
                    continue
                return line
        else:
            self.is_song_end = True
        return line

    def get_current_line(self) -> SongLine:
        return self.song.lines[self.line_index]

    def get_current_word(self, line: SongLine) -> SongWord:
        return line.words[self.word_index]

    def get_song_progress(self) -> Progress:
        return Progress(self.line_index + 1, self.lines_number)

    def get_line_progress(self) -> Progress:
        return Progress(self.word_index + 1, len(self.get_current_line().words))

    def get_remained_attempts(self):
        return self.args.attempts - self.attempts_number

    def get_hint(self) -> List[Hint]:
        line = self.get_current_line()
        if self.args.by_lines:
            return list(line.words[i].get_hint() for i in range(len(line.words)))
        else:
            if self.args.disable_open_letters_on_attempts:
                return [self.get_current_word(line).get_hint()]
            else:
                song_word = self.get_current_word(line)
                word = song_word.word
                hint = song_word.get_hint()
                # Leave at least 1 character to match.
                self.open_letters_at_start = min(self.attempts_number, hint[1] - 1)
                return [Hint(hint.prefix + word[0:self.open_letters_at_start],
                             hint.word_len - self.open_letters_at_start, hint.suffix)]

    def current_line_text(self):
        line = self.get_current_line()
        if line.is_delimiter:
            return "---"
        else:
            return "".join(line.words[i].text for i in range(0, self.word_index))

    def _next_word(self) -> Tuple[bool, str]:  # (is_line_end, matched_text).
        self.attempts_number = 0
        self.open_letters_at_start = 0
        if self._shift_next_word():  # If current line is over.
            start_line_index = self.line_index
            self._shift_next_line()
            matched_text = ""
            for line_index in range(start_line_index, self.line_index):
                line = self.song.lines[line_index]
                matched_text += line.build_text() + '\n'
            matched_text += self.get_current_line().build_text() if self.is_song_end else self.current_line_text()
            return True, matched_text
        return False, self.current_line_text()

    def _next_line(self) -> str:
        self.attempts_number = 0
        self.open_letters_at_start = 0
        line = self.get_current_line()
        matched_text = line.build_text() + '\n'
        self.is_song_end = True  # If no more not-delimiter lines till song end then song is over.
        for i in range(self.line_index + 1, self.lines_number):
            line = self.song.lines[i]
            if line.is_delimiter:
                matched_text += line.build_text() + '\n'
            else:
                self.line_index = i
                self.is_song_end = False
                break
        self.word_index = 0
        return matched_text

    def match(self, actual: str) -> MatcherResult:
        expected_line = self.get_current_line()
        expected_word = self.get_current_word(expected_line)
        self.attempts_number += 1

        expected = expected_line.text if self.args.by_lines else expected_word.word
        if self.open_letters_at_start:
            expected = expected[self.open_letters_at_start - 1:]
        match_ratio = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        match_tolerance_percent = (1.0 - match_ratio) * 100
        if self.args.tolerance:
            is_match = match_tolerance_percent < self.args.tolerance
        else:
            is_match = match_ratio == 1.0
        match_desc = "%3.0f%%" % (match_ratio * 100.0)

        # Check need go next in song or reattempt. Clone values into MatcherResult constructor to keep them separate.
        if is_match or self.attempts_number >= self.args.attempts:
            attempt = self.attempts_number
            if self.args.by_lines:
                is_line_end = True
                matched_text = self._next_line()
            else:
                is_line_end, matched_text = self._next_word()
            return MatcherResult(is_match, True, actual, matched_text, match_desc,
                                 attempt, is_line_end, bool(self.is_song_end))
        else:
            return MatcherResult(False, False, actual, self.current_line_text(), match_desc,
                                 int(self.attempts_number), False, False)


class TerminalLeader(object):
    """
    Abstract matching song leader which expects user answers in terminal. Wraps `Matcher`, manages "ask - answer -
    check" flow and prints statistic at the end. Delegates operating system specific actions to inheritors.
    """

    def __init__(self, matcher: Matcher):
        self.matcher = matcher

    def args(self):
        return self.matcher.args

    def read_user_input(self) -> str:
        raise NotImplementedError()

    def get_status_current_line(self, prev_error: str) -> str:
        if self.args().by_lines:
            line_progress = self.matcher.get_song_progress()
            status = '[%2d/%2d, %s, %d]' % (
                line_progress.position, line_progress.total, prev_error, self.matcher.get_remained_attempts())
        else:
            line_progress = self.matcher.get_song_progress()
            word_progress = self.matcher.get_line_progress()
            status = '[%2d/%2d, %2d/%2d, %s, %2d]' % (
                line_progress.position, line_progress.total, word_progress.position, word_progress.total, prev_error,
                self.matcher.get_remained_attempts())
        return status

    def ask_user_input_this_line(self, matched_line_text: str, prev_match_desc: str):
        raise NotImplementedError("see 'get_status_current_line' for help.")

    def build_letters_number_hint_word(self, hint: Hint) -> str:
        # Simple realization with '_' character to show gaps.
        return '%s%s%s' % (hint.prefix, "_" * hint.word_len, hint.suffix)

    def build_back_characters(self, number: int) -> str:  # Simple realization with '\b'.
        return '\b' * number

    def build_letters_number_hint(self) -> str:
        hint_text = ''
        is_first = True
        backs_number = 0
        for hint in self.matcher.get_hint():  # Returns 1 item in by-word mode.
            if is_first:
                hint_text = self.build_letters_number_hint_word(hint)
                backs_number += hint[1] + len(hint[2])
                is_first = False
            else:
                hint_text += self.build_letters_number_hint_word(hint)
                backs_number += hint.word_len + len(hint.suffix) + len(hint.prefix)
        return hint_text + self.build_back_characters(backs_number)

    @staticmethod
    def get_completed_line_with_statistic(matched_text: str, statistic: List[MatcherResult]) -> str:
        lines_number = 0
        last_line_words_number = 0
        last_line_words_matched = 0
        for result in reversed(statistic):
            if lines_number < 2:  # First result is always with is_line_end=True, next - flag that current line over.
                if result.is_line_end:
                    lines_number += 1
                if lines_number < 2:
                    last_line_words_number += 1
                    if result.is_match:
                        last_line_words_matched += 1
            else:
                if result.is_line_end:
                    lines_number += 1
        return '[%2d line, %2d/%2d matched] %s' % (
            lines_number, last_line_words_matched, last_line_words_number, matched_text)

    def complete_line(self, matched_line_text: str, statistic: List[MatcherResult]):
        raise NotImplementedError("use 'get_completed_line' for implementation.")

    def listen(self):
        matched_text = self.matcher.current_line_text()  # May be a hint at start.
        statistic = []
        prev_match_desc = "?"
        while True:
            self.ask_user_input_this_line(matched_text, prev_match_desc)
            user_input = self.read_user_input()
            result = self.matcher.match(user_input)
            # Check if current word/line matching is over.
            if result.is_shift_next:
                statistic.append(result)
            # Update data for the next iteration.
            matched_text = result.matched_text
            prev_match_desc = result.match_desc
            # Check if need start new line.
            if result.is_line_end:
                self.complete_line(matched_text, statistic)
                matched_text = self.matcher.current_line_text()  # May be a hint at start.
            # Check the song is over.
            if result.is_song_end:
                self.print_statistic(statistic)
                return

    def print_statistic(self, statistic: List[MatcherResult]):
        total = len(statistic)
        matched = 0
        missed = 0
        attempts = 0
        attempts_matched = 0
        for result in statistic:
            if result.is_match:
                matched += 1
                attempts_matched += result.attempt
            else:
                missed += 1
            attempts += result.attempt
        # Note that "leader" may don't over line with '\n' so force add it.
        print("\nUsed settings: %s" % self.args())
        print("From %d %s:" % (total, "lines" if self.args().by_lines else "words"))
        print("  Matched: %d, %d%%" % (matched, matched / total * 100))
        print("  Missed: %d, %d%%" % (missed, missed / total * 100))
        print("  Attempts: total %d, for matched %d" % (attempts, attempts_matched))
        print("  Efficiency for matched: %d%%" % ((matched / attempts_matched * 100) if attempts_matched else 0))
        print("Total efficiency: %d%%" % (matched / attempts * 100))


class UnixTerminalLeader(TerminalLeader):
    ANSI_ERASE_LINE = '\u001b[2K'  # Clear the whole line.
    ANSI_FONT_DECORATION_STOP = '\u001b[0m'
    ANSI_FONT_DECORATION_BOLD = '\u001b[1m'
    ANSI_FONT_DECORATION_UNDERLINE = '\u001b[4m'
    ANSI_SHIFT_CURSOR_LEFT_PREFIX = '\u001b['
    ANSI_SHIFT_CURSOR_LEFT_SUFFIX = 'D'

    def read_user_input(self) -> str:  # Based on https://stackoverflow.com/a/47955341
        # Get TTY attributes.
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # Clone existing TTY attributes and correct them to read by one char.
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        # Read by char.
        user_input = ''
        try:
            termios.tcsetattr(fd, termios.TCSANOW, new)
            while True:
                character = os.read(fd, 1)
                # If user typed terminating key then stop listen input.
                if character == b'\n' or (not self.args().by_lines and character == b' '):
                    break
                elif character == b'\x7f':  # Backspace/delete is hit.
                    if self.args().hide_letters_number:
                        sys.stdout.write("\b \b")
                    else:
                        sys.stdout.write("\b%s %s\b" % (
                            self.ANSI_FONT_DECORATION_UNDERLINE, self.ANSI_FONT_DECORATION_STOP))
                    user_input = user_input[:-1]
                else:
                    try:
                        character_str = character.decode("utf-8")
                    except UnicodeDecodeError as e:
                        print("Unsupported character: %s" % e)
                        return user_input
                    sys.stdout.write(character_str)
                    user_input += character_str
                sys.stdout.flush()
        finally:
            # Revert TTY attributes.
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        return user_input

    def build_letters_number_hint_word(self, hint: Hint) -> str:
        return '%s%s%s%s%s' % (hint.prefix, self.ANSI_FONT_DECORATION_UNDERLINE, " " * hint.word_len,
                               self.ANSI_FONT_DECORATION_STOP, hint.suffix)

    def ask_user_input_this_line(self, matched_line_text: str, prev_match_desc: str):
        status = self.get_status_current_line(prev_match_desc)
        hint = '' if self.args().hide_letters_number else self.build_letters_number_hint()
        if self.args().by_lines:
            line = "\r%s%s%s %s" % (
                self.ANSI_FONT_DECORATION_BOLD, status, self.ANSI_FONT_DECORATION_STOP, hint)
        else:
            line = "\r%s%s%s %s%s" % (
                self.ANSI_FONT_DECORATION_BOLD, status, self.ANSI_FONT_DECORATION_STOP, matched_line_text, hint)
        sys.stdout.write(self.ANSI_ERASE_LINE + line)
        sys.stdout.flush()

    def complete_line(self, matched_text: str, statistic: List[MatcherResult]):
        sys.stdout.write('\r%s%s' % (self.ANSI_ERASE_LINE,
                                     self.get_completed_line_with_statistic(matched_text, statistic)))
        sys.stdout.flush()


class WindowsTerminalLeader(TerminalLeader):
    # Windows console doesn't support ANSI control characters (at least up to newest Win 10 + registry switch).
    # Python 'colorama' lib adds support of ANSI for Windows but it is extra dependency.
    # So Windows support is realized without font/color features and without clearing "extra characters garbage".

    def ask_user_input_this_line(self, matched_line_text: str, prev_match_desc: str):
        status = self.get_status_current_line(prev_match_desc)
        hint = '' if self.args().hide_letters_number else self.build_letters_number_hint()
        if self.args().by_lines:
            line = "\r%s %s" % (status, hint)
        else:
            line = "\r%s %s%s" % (status, matched_line_text, hint)
        sys.stdout.write(line)
        sys.stdout.flush()

    def read_user_input(self) -> str:  # Based on https://stackoverflow.com/a/12179724
        # Read by char.
        user_input = ''
        while True:
            character = msvcrt.getch()
            # If user typed terminating key then stop listen input.
            if character == b'\r' or (not self.args().by_lines and character == b' '):
                break
            elif character == b'\x08':  # Backspace is hit.
                if self.args().hide_letters_number:
                    sys.stdout.write("\b \b")
                else:
                    sys.stdout.write("\b_\b")
                user_input = user_input[:-1]
            elif character == b'\x03':  # Ctrl+C is hit.
                exit(1)
            else:
                try:
                    character_str = character.decode("866")
                except Exception as e:
                    print("Unsupported character: %s" % e)
                    return user_input
                sys.stdout.write(character_str)
                user_input += character_str
            sys.stdout.flush()
        return user_input

    def complete_line(self, matched_text: str, statistic: List[MatcherResult]):
        sys.stdout.write('\r%s' % (self.get_completed_line_with_statistic(matched_text, statistic)))
        sys.stdout.flush()


class ParseSettingsFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def parse_settings():
    parser = argparse.ArgumentParser(
        description='Interactive script which asks user to type song words or lines and checks that user typed them'
                    ' right by given text file with song lyrics. Song lyrics may be found on the Internet, for'
                    ' example at https://www.lyricsera.com.')
    parser.add_argument('--settings', type=open, action=ParseSettingsFromFile,
                        help="All settings from file. Options are separated by space and specified by pairs, i.e. as in"
                             " command line. See --help output for details.")
    parser.add_argument('song_text_path', nargs='?', help="Path to file with song text.")
    parser.add_argument('--by-lines', action='store_true',
                        help='Flag to recognize the whole lines instead of words.')
    parser.add_argument('-a', '--attempts', metavar='N', type=int, default=2,
                        help='Number of attempts to recognize the world or line (default: 2).')
    parser.add_argument('-t', '--tolerance', metavar='P', default=30,
                        help='Tolerance of matching in integer percent (default: 30%%).')
    parser.add_argument('--hide-letters-number', action='store_true',
                        help="Flag to don't show tip about letters number.")
    parser.add_argument('--skip-words-len', type=int, default=2,
                        help="Display short words as hints, don't ask match them. Value is max length of such words.")
    parser.add_argument('--disable-open-letters-on-attempts', action='store_true',
                        help="Disables opening by one letter at start on each failed attempt.")
    parser.add_argument('--debug', action='store_true',
                        help="Flag to show debug output.")
    # TODO: add presets for 'advanced', 'intermediate', 'beginner'.
    # TODO: show hints as random letters.
    # TODO: download lyrics: https://www.quora.com/Whats-a-good-api-to-use-to-get-song-lyrics
    args = parser.parse_args()
    # Ask user for single required parameter - path to file with lyrics.
    if not args.song_text_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        args.song_text_path = filedialog.askopenfilename(title="Specify text file with song words (lyrics) to match.")
    return args


def parse_song(file_path: str):
    lines = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                lines.append(SongLine(line))
    except TypeError:
        raise FileNotFoundError("Can't work without specified lyrics file. Run with '--help' argument for details.")
    return Song(lines)


if __name__ == "__main__":
    args = parse_settings()
    song = parse_song(args.song_text_path)
    if args.debug:
        print(song)
        print(args)
    print("Listen song (by few seconds) and type words or lines (lyrics) from it. Use Space or Enter to check result.")
    print("Square brackets at current line start shows progress and error percent from previous attempt. Format is"
          " ['line number'/'total lines', 'words pass'/'total words in line', 'last attempt error',"
          " 'remained attempts number']")
    print("All 'completed' lines also have prefix in square brackets with line statistic. At the end will be printed"
          " whole song statistic.")
    print("OK - let's go! If you get bored or want to change song then press Ctrl+C to exit immediately.")
    if sys.platform.startswith("win"):
        leader = WindowsTerminalLeader(Matcher(song, args))
    else:
        leader = UnixTerminalLeader(Matcher(song, args))
    leader.listen()
    input("Press Enter to exit.")
