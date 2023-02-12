# language_trainer_by_songs
Language comprehension trainer/game by songs. See promo video (Russian) at https://vk.com/wall99277406_1166

Terminal-based version (in this public repo, more comprehensive UI is developed separately).

# How to use

1. Download lyrics for you favorite song and put it into simple text file without song name or any extra words/characters.
2. Run [language_trainer_by_songs.py](/language_trainer_by_songs.py), it will ask for file in "open file" dialog - specify lyrics file.
3. Start song play in you favorite player (tip - better to setup hotkey "rewind 5 seconds back" - it would be useful).
4. Try to distinguish words and type them. Interface would add hints about word length, first character(s), also it will limit number of attempts until count word as missed, open it and ask for the next (BTW settings are configurable). Rewing song and listen few times to clearly understand words.
5. At the end program will show your total score.

It is MVP. Following development (Web UI + improvements) moved to private repository.
