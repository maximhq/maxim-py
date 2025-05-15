import logging

# Create a logger for maxim
# When users set the level with logging.getLogger('maxim').setLevel(logging.DEBUG)
# this logger will respect that level setting


def scribe():
    return logging.getLogger("maxim")


if scribe().level == logging.NOTSET:
    print("\033[32m[MaximSDK] Using global logging level\033[0m")
else:
    print(
        f"\033[32m[MaximSDK] Log level set to {logging.getLevelName(scribe().level)}.\nYou can change it by calling logging.getLogger('maxim').setLevel(newLevel)\033[0m"
    )
