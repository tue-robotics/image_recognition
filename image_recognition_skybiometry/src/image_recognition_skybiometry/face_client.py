# -*- coding: utf-8 -*-
#
# Name: SkyBiometry Face Detection and Recognition API Python client library
# Description: SkyBiometry Face Detection and Recognition REST API Python client library.
#
# For more information about the API and the return values,
# visit the official documentation at http://www.skybiometry.com/Documentation
#
# Author: Tomaž Muraus (http://www.tomaz.me)
# License: BSD

import os.path
import requests

try:
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlencode

try:
    import json
except ImportError:
    import simplejson as json

from future.utils import iteritems

API_HOST = 'api.skybiometry.com/fc'
USE_SSL = True


class FaceClient(object):
    def __init__(self, api_key=None, api_secret=None):
        if not api_key or not api_secret:
            raise AttributeError('Missing api_key or api_secret argument')

        self.api_key = api_key
        self.api_secret = api_secret
        self.format = 'json'

        self.twitter_credentials = None
        self.facebook_credentials = None

    def set_twitter_oauth_credentials(self, user=None, secret=None, token=None):
        if not user or not secret or not token:
            raise AttributeError('Missing one of the required arguments')

        self.twitter_credentials = {'twitter_oauth_user': user,
                                    'twitter_oauth_secret': secret,
                                    'twitter_oauth_token': token}

    def set_facebook_oauth_credentials(self, user_id=None, session_id=None, oauth_token=None):
        for (key, value) in [('user_id', user_id), ('session_id', session_id), ('oauth_token', oauth_token)]:
            if not value:
                raise AttributeError('Missing required argument: %s' % key)

        self.facebook_credentials = {'fb_user_id': user_id,
                                     'fb_session_id': session_id,
                                     'fb_oauth_token': oauth_token}

    # Recognition engine methods
    def faces_detect(self, urls=None, file_=None, buffer_=None, aggressive=False):
        """
        Returns tags for detected faces in one or more photos, with geometric
        information of the tag, eyes, nose and mouth, as well as the gender,
        glasses, and smiling attributes.

        http://www.skybiometry.com/Documentation#faces/detect
        """
        if not urls and not file_ and not buffer_:
            raise AttributeError('Missing URLs/filename/buffer argument')

        data = {'attributes': 'all', 'force_reprocess_image': 'true'}
        files = []
        buffers = []

        if file_:
            # Check if the file exists
            if not hasattr(file_, 'read') and not os.path.exists(file_):
                raise IOError('File %s does not exist' % file_)

            files.append(file_)
        elif buffer_:
            buffers.append(buffer_)
        else:
            data['urls'] = urls

        if aggressive:
            data['detector'] = 'aggressive'

        response = self.send_request('faces/detect', data, files, buffers)
        return response

    def faces_status(self, uids=None, namespace=None):
        """
        Reports training set status for the specified UIDs.

        http://www.skybiometry.com/Documentation#faces/status
        """
        if not uids:
            raise AttributeError('Missing user IDs')

        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uids)

        data = {'uids': uids}
        self.__append_user_auth_data(data, facebook_uids, twitter_uids)
        self.__append_optional_arguments(data, namespace=namespace)

        response = self.send_request('faces/status', data)
        return response

    def faces_recognize(self, uids=None, urls=None, files_=None, buffers_=None, aggressive=False, train=None,
                        namespace=None):
        """
        Attempts to detect and recognize one or more user IDs' faces, in one
        or more photos.
        For each detected face, the SkyBiometry engine will return the most likely
        user IDs, or empty result for unrecognized faces. In addition, each
        tag includes a threshold score - any score below this number is
        considered a low-probability hit.

        http://www.skybiometry.com/Documentation#faces/recognize
        """
        if not uids or (not urls and not files_ and not buffers_):
            raise AttributeError('Missing required arguments')

        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uids)

        data = {'uids': uids, 'attributes': 'all'}
        files = []
        buffers = []

        if files_:
            for file_ in files_:
                # Check if the file exists
                if not file_:
                    continue
                if not hasattr(file_, 'read') and not os.path.exists(file_):
                    raise IOError('File %s does not exist' % file_)
                files.append(file_)
        elif buffers_:
            buffers.extend(buffers_)
        else:
            data.update({'urls': urls})

        if aggressive:
            data['detector'] = 'aggressive'

        self.__append_user_auth_data(data, facebook_uids, twitter_uids)
        self.__append_optional_arguments(data, train=train, namespace=namespace)

        response = self.send_request('faces/recognize', data, files, buffers)
        return response

    def faces_train(self, uids=None, namespace=None):
        """
        Calls the training procedure for the specified UIDs, and reports back
        changes.

        http://www.skybiometry.com/Documentation#faces/train
        """
        if not uids:
            raise AttributeError('Missing user IDs')

        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uids)

        data = {'uids': uids}
        self.__append_user_auth_data(data, facebook_uids, twitter_uids)
        self.__append_optional_arguments(data, namespace=namespace)

        response = self.send_request('faces/train', data)
        return response

    # Methods for managing face tags
    def tags_get(self, uids=None, urls=None, pids=None, order='recent', limit=5, together=False, filters=None,
                 namespace=None):
        """
        Returns saved tags in one or more photos, or for the specified
        User ID(s).
        This method also accepts multiple filters for finding tags
        corresponding to a more specific criteria such as front-facing,
        recent, or where two or more users appear together in same photos.

        http://www.skybiometry.com/Documentation#tags/get
        """
        if not uids and not urls:
            raise AttributeError('Missing user IDs or URLs')
        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uids)

        data = {'together': 'true' if together else 'false', 'limit': limit}
        self.__append_user_auth_data(data, facebook_uids, twitter_uids)
        self.__append_optional_arguments(data, uids=uids, urls=urls, pids=pids, filter=filters, namespace=namespace)

        response = self.send_request('tags/get', data)
        return response

    def tags_add(self, url=None, x=None, y=None, width=None, uid=None, tagger_id=None, label=None, password=None):
        """
        Add a (manual) face tag to a photo. Use this method to add face tags
        where those were not detected for completeness of your service.

        http://www.skybiometry.com/Documentation#tags/add
        """
        if not url or not x or not y or not width or not uid or not tagger_id:
            raise AttributeError('Missing one of the required arguments')

        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uid)

        data = {'url': url,
                'x': x,
                'y': y,
                'width': width,
                'uid': uid,
                'tagger_id': tagger_id}
        self.__append_user_auth_data(data, facebook_uids, twitter_uids)
        self.__append_optional_arguments(data, label=label, password=password)

        response = self.send_request('tags/add', data)
        return response

    def tags_save(self, tids=None, uid=None, tagger_id=None, label=None, password=None):
        """
        Saves a face tag. Use this method to save tags for training the
        index, or for future use of the faces.detect and tags.get methods.

        http://www.skybiometry.com/Documentation#tags/save
        """
        if not tids or not uid:
            raise AttributeError('Missing required argument')

        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uid)

        data = {'tids': tids,
                'uid': uid}
        self.__append_user_auth_data(data, facebook_uids, twitter_uids)
        self.__append_optional_arguments(data, tagger_id=tagger_id, label=label, password=password)

        response = self.send_request('tags/save', data)
        return response

    def tags_remove(self, tids=None, password=None):
        """
        Remove a previously saved face tag from a photo.

        http://www.skybiometry.com/Documentation#tags/remove
        """
        if not tids:
            raise AttributeError('Missing tag IDs')

        data = {'tids': tids}

        response = self.send_request('tags/remove', data)
        return response

    # Account management methods
    def account_limits(self):
        """
        Returns current rate limits for the account represented by the passed
        API key and Secret.

        http://www.skybiometry.com/Documentation#account/limits
        """
        response = self.send_request('account/limits')
        return response['usage']

    def account_users(self, namespaces=None):
        """
        Returns current rate limits for the account represented by the passed
        API key and Secret.

        http://www.skybiometry.com/Documentation#account/users
        """
        if not namespaces:
            raise AttributeError('Missing namespaces argument')

        response = self.send_request('account/users', {'namespaces': namespaces})

        return response

    def account_namespaces(self):
        """
        Returns all valid data namespaces for user authorized by specified API key.

        http://www.skybiometry.com/Documentation#account/namespaces
        """

        response = self.send_request('account/namespaces')

        return response

    def __check_user_auth_credentials(self, uids):
        # Check if needed credentials are provided
        facebook_uids = [uid for uid in uids.split(',') if uid.find('@facebook.com') != -1]
        twitter_uids = [uid for uid in uids.split(',') if uid.find('@twitter.com') != -1]

        if facebook_uids and not self.facebook_credentials:
            raise AttributeError('You need to set Facebook credentials ' +
                                 'to perform action on Facebook users')

        if twitter_uids and not self.twitter_credentials:
            raise AttributeError('You need to set Twitter credentials to ' +
                                 'perform action on Twitter users')

        return facebook_uids, twitter_uids

    def __append_user_auth_data(self, data, facebook_uids, twitter_uids):
        if facebook_uids:
            data.update({'user_auth': 'fb_user: %s, fb_session: %s, ''fb_oauth_token: %s' %
                                      (self.facebook_credentials['fb_user_id'],
                                       self.facebook_credentials['fb_session_id'],
                                       self.facebook_credentials['fb_oauth_token'])})

        if twitter_uids:
            data.update({'user_auth': ('twitter_oauth_user:%s,twitter_oauth_secret:%s,'
                                       'twitter_oauth_token:%s' %
                                       (self.twitter_credentials['twitter_oauth_user'],
                                        self.twitter_credentials['twitter_oauth_secret'],
                                        self.twitter_credentials['twitter_oauth_token']))})

    @staticmethod
    def __append_optional_arguments(data, **kwargs):
        for key, value in iteritems(kwargs):
            if value:
                data.update({key: value})

    def send_request(self, method=None, parameters=None, files=None, buffers=None):
        protocol = 'https://' if USE_SSL else 'http://'
        url = '%s%s/%s.%s' % (protocol, API_HOST, method, self.format)
        data = {'api_key': self.api_key, 'api_secret': self.api_secret}

        if parameters:
            data.update(parameters)

        # Local file is provided, use multi-part form
        if files or buffers:
            from .multipart import Multipart
            form = Multipart()

            for key, value in iteritems(data):
                form.field(key, value)

            if files:
                for i, file_ in enumerate(files, 1):
                    if hasattr(file_, 'read'):
                        if hasattr(file_, 'name'):
                            name = os.path.basename(file_.name)
                        else:
                            name = 'attachment_%d' % i
                        close_file = False
                    else:
                        name = os.path.basename(file_)
                        file_ = open(file_, 'rb')
                        close_file = True

                    try:
                        form.file(name, name, file_.read())
                    finally:
                        if close_file:
                            file_.close()
            else:
                for i, buffer_ in enumerate(buffers, 1):
                    name = 'attachment_%d' % i
                    form.file(name, name, buffer_)
            (content_type, post_data) = form.get()
            headers = {'Content-Type': content_type}
        else:
            post_data = urlencode(data)
            headers = {}

        try:
            r = requests.post(url, headers=headers, data=post_data)
            response = r.text
        except requests.HTTPError as e:
            response = e.response.text

        response_data = json.loads(response)

        if 'status' in response_data and response_data['status'] == 'failure':
            raise FaceError(response_data['error_code'], response_data['error_message'])

        return response_data


class FaceError(Exception):
    def __init__(self, error_code, error_message):
        self.error_code = error_code
        self.error_message = error_message

    def __str__(self):
        return '%s (%d)' % (self.error_message, self.error_code)
