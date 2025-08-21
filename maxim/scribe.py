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
        """Initialize a scribe logger.

        Args:
            name: The name of the logger.
        """
        self.name = name
        self.disable_internal_logs = True
        self.logger = logging.getLogger(name)        

    def _should_log(self, msg):
        """Check if the message should be logged.

        Args:
            msg: The message to check.

        Returns:
            bool: True if the message should be logged, False otherwise.
        """
        return not (
            self.disable_internal_logs
            and isinstance(msg, str)
            and msg.startswith("[Internal]")
        )

    def debug(self, msg, *args, **kwargs):
        """Log a debug message.

        Args:
            msg: The message to log.
            *args: The arguments to log.
            **kwargs: The keyword arguments to log.
        """
        if not self._should_log(msg):
            return
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message.

        Args:
            msg: The message to log.
            *args: The arguments to log.
            **kwargs: The keyword arguments to log.
        """
        if not self._should_log(msg):
            return
        self.logger.warning(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """Log a message.

        Args:
            level: The level of the message.
            msg: The message to log.
            *args: The arguments to log.
            **kwargs: The keyword arguments to log.
        """
        if not self._should_log(msg):
            return
        self.logger.log(level, msg, *args, **kwargs)

    def silence(self):
        """Silence the logger.

        This method sets the logger level to CRITICAL + 1.
        """
        self.logger.setLevel(logging.CRITICAL + 1)

    def error(self, msg, *args, **kwargs):
        """Log an error message.

        Args:
            msg: The message to log.
            *args: The arguments to log.
            **kwargs: The keyword arguments to log.
        """
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message.

        Args:
            msg: The message to log.
            *args: The arguments to log.
            **kwargs: The keyword arguments to log.
        """
        if not self._should_log(msg):
            return
        if self.get_level() > logging.INFO:
            return
        self.logger.info(msg, *args, **kwargs)

    def get_level(self):
        """Get the level of the logger.

        Returns:
            int: The level of the logger.
        """
        return self.logger.level

    def set_level(self, level):
        """Set the level of the logger.

        Args:
            level: The level to set.
        """
        self.logger.setLevel(level)


def scribe():
    global _scribe_instance
    if _scribe_instance is None:
        _scribe_instance = Scribe("maxim")
        # Take global logging level and set it for _scribe_instance if set
        root_level = logging.getLogger().getEffectiveLevel()
        if root_level != logging.NOTSET:
            _scribe_instance.set_level(root_level)
        elif _scribe_instance.get_level() == logging.NOTSET:
            print("\033[32m[MaximSDK] Using warning logging level.\033[0m")
            print(
                "\033[32m[MaximSDK] For debug or info logs, set global logging level using logging.basicConfig(level=logging.DEBUG) or logging.basicConfig(level=logging.INFO).\033[0m"
            )
            _scribe_instance.set_level(logging.DEBUG)
        else:
            print(
                f"\033[32m[MaximSDK] Log level set to {logging.getLevelName(_scribe_instance.get_level())}.\nYou can change it by calling logging.getLogger('maxim').setLevel(newLevel)\033[0m"
            )
    return _scribe_instance
