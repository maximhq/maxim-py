import logging

# Create a logger for maxim
# When users set the level with logging.getLogger('maxim').setLevel(logging.DEBUG)
# this logger will respect that level setting

_scribe_instance = None


class Scribe:
    """
    Scribe logger wrapper for maxim.
    Log level is managed externally via set_level or the standard logging API.
    By default, the logger uses the global logging configuration.
    """

    def __init__(self, name):
        self.name = name
        self.disable_internal_logs = False
        self.logger = logging.getLogger(name)        
        

    def _should_log(self, msg):
        return not (
            self.disable_internal_logs
            and isinstance(msg, str)
            and msg.startswith("[Internal]")
        )

    def debug(self, msg, *args, **kwargs):
        if not self._should_log(msg):
            return
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not self._should_log(msg):
            return
        self.logger.warning(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        if not self._should_log(msg):
            return
        self.logger.log(level, msg, *args, **kwargs)

    def silence(self):
        self.logger.setLevel(logging.CRITICAL + 1)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if not self._should_log(msg):
            return
        if self.get_level() > logging.INFO:
            return
        self.logger.info(msg, *args, **kwargs)

    def get_level(self):
        return self.logger.level
    
    def set_level(self, level):
        self.logger.setLevel(level)


def scribe():
    global _scribe_instance
    if _scribe_instance is None:
        _scribe_instance = Scribe("maxim")        
        if _scribe_instance.get_level() == logging.NOTSET:
            print("\033[32m[MaximSDK] Using info logging level.\033[0m")
            print(
                "\033[32m[MaximSDK] For debug logs, set global logging level to debug logging.basicConfig(level=logging.DEBUG).\033[0m"
            )
            _scribe_instance.set_level(logging.ERROR)
        else:
            print(
                f"\033[32m[MaximSDK] Log level set to {logging.getLevelName(_scribe_instance.get_level())}.\nYou can change it by calling logging.getLogger('maxim').setLevel(newLevel)\033[0m"
            )
    return _scribe_instance
