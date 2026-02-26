"""
Reverse interface: media post create.
"""

import orjson
from typing import Any
from curl_cffi.requests import AsyncSession

from app.core.logger import logger
from app.core.config import get_config
from app.core.exceptions import UpstreamException
from app.services.token.service import TokenService
from app.services.reverse.utils.headers import build_headers
from app.services.reverse.utils.retry import retry_on_status
from app.services.reverse.set_birth import SetBirthReverse

MEDIA_POST_API = "https://grok.com/rest/media/post/create"


class MediaPostReverse:
    """/rest/media/post/create reverse interface."""

    @staticmethod
    async def request(
        session: AsyncSession,
        token: str,
        mediaType: str,
        mediaUrl: str,
        prompt: str = "",
    ) -> Any:
        """Create media post in Grok.

        Args:
            session: AsyncSession, the session to use for the request.
            token: str, the SSO token.
            mediaType: str, the media type.
            mediaUrl: str, the media URL.

        Returns:
            Any: The response from the request.
        """
        try:
            # Get proxies
            base_proxy = get_config("proxy.base_proxy_url")
            proxies = {"http": base_proxy, "https": base_proxy} if base_proxy else None

            # Build headers
            headers = build_headers(
                cookie_token=token,
                content_type="application/json",
                origin="https://grok.com",
                referer="https://grok.com",
            )

            # Build payload
            payload = {"mediaType": mediaType}
            if mediaUrl:
                payload["mediaUrl"] = mediaUrl
            if prompt:
                payload["prompt"] = prompt

            # Curl Config
            timeout = get_config("video.timeout")
            browser = get_config("proxy.browser")
            age_verify_attempted = False

            async def _do_request():
                nonlocal age_verify_attempted
                response = await session.post(
                    MEDIA_POST_API,
                    headers=headers,
                    data=orjson.dumps(payload),
                    timeout=timeout,
                    proxies=proxies,
                    impersonate=browser,
                )

                if response.status_code != 200:
                    content = ""
                    try:
                        content = await response.text()
                    except Exception:
                        pass

                    if response.status_code == 403 and not age_verify_attempted:
                        age_verify_attempted = True
                        try:
                            await SetBirthReverse.request(session, token)
                        except Exception as e:
                            logger.warning(
                                f"MediaPostReverse: age verify attempt failed: {str(e)[:120]}"
                            )
                        else:
                            retry_response = await session.post(
                                MEDIA_POST_API,
                                headers=headers,
                                data=orjson.dumps(payload),
                                timeout=timeout,
                                proxies=proxies,
                                impersonate=browser,
                            )
                            if retry_response.status_code == 200:
                                return retry_response
                            try:
                                content = await retry_response.text()
                            except Exception:
                                pass
                            response = retry_response

                    logger.error(
                        f"MediaPostReverse: Media post create failed, {response.status_code}",
                        extra={"error_type": "UpstreamException"},
                    )
                    raise UpstreamException(
                        message=f"MediaPostReverse: Media post create failed, {response.status_code}",
                        details={
                            "status": response.status_code,
                            "body": content,
                            "age_verify_attempted": age_verify_attempted,
                        },
                    )

                return response

            def _extract_status(e: Exception) -> int | None:
                if isinstance(e, UpstreamException):
                    status = None
                    if e.details and "status" in e.details:
                        status = e.details["status"]
                    else:
                        status = getattr(e, "status_code", None)
                    if status in (429, 403):
                        return None
                    return status
                return None

            return await retry_on_status(_do_request, extract_status=_extract_status)

        except Exception as e:
            # Handle upstream exception
            if isinstance(e, UpstreamException):
                status = None
                if e.details and "status" in e.details:
                    status = e.details["status"]
                else:
                    status = getattr(e, "status_code", None)
                if status == 401:
                    try:
                        await TokenService.record_fail(token, status, "media_post_auth_failed")
                    except Exception:
                        pass
                raise

            # Handle other non-upstream exceptions
            logger.error(
                f"MediaPostReverse: Media post create failed, {str(e)}",
                extra={"error_type": type(e).__name__},
            )
            raise UpstreamException(
                message=f"MediaPostReverse: Media post create failed, {str(e)}",
                details={"status": 502, "error": str(e)},
            )


__all__ = ["MediaPostReverse"]
