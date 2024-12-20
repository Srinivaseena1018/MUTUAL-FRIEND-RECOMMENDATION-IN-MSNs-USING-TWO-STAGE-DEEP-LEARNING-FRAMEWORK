"""
WSGI config for friendship_inference_in_mobile_social_networks.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'friendship_inference_in_mobile_social_networks.settings')
application = get_wsgi_application()
