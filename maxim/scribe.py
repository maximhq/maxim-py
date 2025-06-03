import logging

# Create a logger for maxim
# When users set the level with logging.getLogger('maxim').setLevel(logging.DEBUG)
# this logger will respect that level setting


class Scribe():
    
    def __init__(self, name):
        self.disable_internal_logs = True
        self.logger = logging.getLogger(name)
        
    def debug(self, msg, *args, **kwargs):
        if self.disable_internal_logs and msg.startswith("[Internal]"):
            return
        self.logger.debug(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        if self.disable_internal_logs and msg.startswith("[Internal]"):
            return
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.disable_internal_logs and msg.startswith("[Internal]"):
            return
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.disable_internal_logs and msg.startswith("[Internal]"):
            return
        self.logger.info(msg, *args, **kwargs)    

    def set_level(self, level):
        self.logger.setLevel(level)

    def get_level(self):
        return self.logger.getLevel()
    
    @property
    def level(self):
        return self.logger.level
    
    @level.setter
    def level(self, level):
        self.logger.level = level
    

def scribe():
    logger = Scribe("maxim")    
    return logger


if scribe().level == logging.NOTSET:
    print("\033[32m[MaximSDK] Using global logging level\033[0m")
else:
    print(
        f"\033[32m[MaximSDK] Log level set to {logging.getLevelName(scribe().level)}.\nYou can change it by calling logging.getLogger('maxim').setLevel(newLevel)\033[0m"
    )
