"""
ASGI config for friendship_inference_in_mobile_social_networks.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'friendship_inference_in_mobile_social_networks.settings')

application = get_asgi_application()
