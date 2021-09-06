# Files with interruptions in seconds
# Dictionary - File_ID - onset/offset time
interrupt = {373: [395, 428],
             444: [286, 387]}

# Files with missing virtual agent transcriptions - onset/offset timings
transcripts = [451, 458, 480]

# Files with mis-aligned transcripts
# Dictionary - File ID - misalignment
misaligned = {318: 34.319917,
              321: 3.8379167,
              341: 6.1892,
              362: 16.8582}

wrong_labels = {409: 1}

synchronise_labels = ['<sync>', '<synch>', '[sync]', '[synch]', '[syncing]',
                      '[synching]']

# 'xxx' 'xxxx' denotes unrecognised words
# <> denotes that there was an interruption
# [] denotes non-verbal sound
words_to_remove = {0: 'xxx', 1: 'xxxx', 2: ' ', 3: '  ', 4: '   ', 5: '    ',
                   6: '     '}
symbols_to_remove = ['<', '>', '[', ']']

excluded_sessions = [342, 394, 398, 460]

