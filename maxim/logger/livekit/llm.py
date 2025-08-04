import functools
import inspect
import traceback

from ...scribe import scribe

llm_f_skip_list = []


def pre_hook(self, hook_name, args, kwargs):
    """Pre-hook for LLM methods"""
    try:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
        )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} pre-hook failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    """Post-hook for LLM methods"""
    try:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] {hook_name} completed; result_type={type(result).__name__}"
        )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} post-hook failed; error={e!s}\n{traceback.format_exc()}"
        )


def instrument_llm_init(orig, name, class_name):
    """Special instrumentation for LLM __init__ method"""
    if name in llm_f_skip_list:
        return orig

    def wrapper(self, *args, **kwargs):
        pre_hook(self, name, args, kwargs)
        result = None
        try:
            result = orig(self, *args, **kwargs)
            scribe().debug(f"[Internal][{class_name}] initialized")
            return result
        except Exception as e:
            scribe().warning(
                f"[Internal][{class_name}] {name} failed; error={e!s}\n{traceback.format_exc()}"
            )
            raise
        finally:
            post_hook(self, result, name, args, kwargs)

    return functools.wraps(orig)(wrapper)


def instrument_llm(orig, name):
    """General instrumentation for LLM methods"""
    if name in llm_f_skip_list:
        return orig

    if inspect.iscoroutinefunction(orig):

        async def async_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = await orig(self, *args, **kwargs)
                return result
            except Exception as e:
                scribe().warning(
                    f"[Internal][{self.__class__.__name__}] {name} failed; error={e!s}\n{traceback.format_exc()}"
                )
                raise
            finally:
                post_hook(self, result, name, args, kwargs)

        wrapper = async_wrapper
    else:

        def sync_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = orig(self, *args, **kwargs)
                return result
            except Exception as e:
                scribe().warning(
                    f"[Internal][{self.__class__.__name__}] {name} failed; error={e!s}\n{traceback.format_exc()}"
                )
                raise
            finally:
                post_hook(self, result, name, args, kwargs)

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper) 