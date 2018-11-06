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

import urllib
import urllib2
import os.path
import warnings

try:
    import json
except ImportError:
    import simplejson as json

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

    def set_twitter_user_credentials(self, *args, **kwargs):
        warnings.warn(('Twitter username & password auth has been ' +
                       'deprecated. Please use oauth based auth - ' +
                       'set_twitter_oauth_credentials()'))

    def set_twitter_oauth_credentials(self, user=None, secret=None, token=None):
        if not user or not secret or not token:
            raise AttributeError('Missing one of the required arguments')

        self.twitter_credentials = {'twitter_oauth_user': user,
                                    'twitter_oauth_secret': secret,
                                    'twitter_oauth_token': token}

    def set_facebook_access_token(self, *args, **kwargs):
        warnings.warn(('Method has been renamed to ' +
                       'set_facebook_oauth_credentials(). Support for ' +
                       'username & password based auth has also been dropped. ' +
                       'Now only oAuth2 token based auth is supported'))

    def set_facebook_oauth_credentials(self, user_id=None, session_id=None, oauth_token=None):
        for (key, value) in [('user_id', user_id), ('session_id', session_id), ('oauth_token', oauth_token)]:
            if not value:
                raise AttributeError('Missing required argument: %s' % (key))

        self.facebook_credentials = {'fb_user_id': user_id,
                                     'fb_session_id': session_id,
                                     'fb_oauth_token': oauth_token}

    ### Recognition engine methods ###
    def faces_detect(self, urls=None, file=None, buffer=None, aggressive=False):
        """
		Returns tags for detected faces in one or more photos, with geometric
		information of the tag, eyes, nose and mouth, as well as the gender,
		glasses, and smiling attributes.

		http://www.skybiometry.com/Documentation#faces/detect
		"""
        if not urls and not file and not buffer:
            raise AttributeError('Missing URLs/filename/buffer argument')

        data = {'attributes': 'all'}
        files = []
        buffers = []

        if file:
            # Check if the file exists
            if not hasattr(file, 'read') and not os.path.exists(file):
                raise IOError('File %s does not exist' % (file))

            files.append(file)
        elif buffer:
            buffers.append(buffer)
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

    def faces_recognize(self, buffers):
        uids = "guido"
        urls = None
        file = None
        buffer = None
        aggressive = False
        train = None
        namespace = "robocup"

        (facebook_uids, twitter_uids) = self.__check_user_auth_credentials(uids)

        data = {'uids': uids, 'attributes': 'all'}
        files = []

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

    ### Methods for managing face tags ###
    def tags_get(self, uids=None, urls=None, pids=None, order='recent', limit=5, together=False, filter=None,
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
        self.__append_optional_arguments(data, uids=uids, urls=urls, pids=pids, filter=filter, namespace=namespace)

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

    ### Account management methods ###
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

        return (facebook_uids, twitter_uids)

    def __append_user_auth_data(self, data, facebook_uids, twitter_uids):
        if facebook_uids:
            data.update({'user_auth': 'fb_user:%s,fb_session:%s,' +
                                      'fb_oauth_token:%s' %
                                      (self.facebook_credentials['fb_user_id'],
                                       self.facebook_credentials['fb_session_id'],
                                       self.facebook_credentials['fb_oauth_token'])})

        if twitter_uids:
            data.update({'user_auth':
                             ('twitter_oauth_user:%s,twitter_oauth_secret:%s,'
                              'twitter_oauth_token:%s' %
                              (self.twitter_credentials['twitter_oauth_user'],
                               self.twitter_credentials['twitter_oauth_secret'],
                               self.twitter_credentials['twitter_oauth_token']))})

    def __append_optional_arguments(self, data, **kwargs):
        for key, value in kwargs.iteritems():
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
            from multipart import Multipart
            form = Multipart()

            for key, value in data.iteritems():
                form.field(key, value)

            if files:
                for i, file in enumerate(files, 1):
                    if hasattr(file, 'read'):
                        if hasattr(file, 'name'):
                            name = os.path.basename(file.name)
                        else:
                            name = 'attachment_%d' % i
                        close_file = False
                    else:
                        name = os.path.basename(file)
                        file = open(file, 'rb')
                        close_file = True

                    try:
                        form.file(name, name, file.read())
                    finally:
                        if close_file:
                            file.close()
            else:
                for i, buffer in enumerate(buffers, 1):
                    name = 'attachment_%d' % i
                    form.file(name, name, buffer)
            (content_type, post_data) = form.get()
            headers = {'Content-Type': content_type}
        else:
            post_data = urllib.urlencode(data)
            headers = {}

        request = urllib2.Request(url, headers=headers, data=post_data)
        try:
            response = urllib2.urlopen(request)
            response = response.read()
        except urllib2.HTTPError as e:
            response = e.read()
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
