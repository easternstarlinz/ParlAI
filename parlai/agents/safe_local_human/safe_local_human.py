#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent that gets the local keyboard input in the act() function.

Applies safety classifier(s) to process user and partner messages.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.utils.misc import display_messages
from parlai.utils.strings import colorize
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier


OFFENSIVE_USER_REPLY = '[ 这个事儿我不懂，聊点别的吧 ]'
OFFENSIVE_BOT_REPLY = (
    '[ 你们人类太那个了，AI甘拜下风 ]'
)


class SafeLocalHumanAgent(LocalHumanAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Safe Local Human Arguments')
        agent.add_argument(
            '--safety',
            type=str,
            default='all',
            choices={'none', 'string_matcher', 'classifier', 'all'},
            help='Apply safety filtering to messages',
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'safeLocalHuman'
        self._init_safety(opt)

    def _init_safety(self, opt):
        """
        Initialize safety modules.
        """
        if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
            self.offensive_string_matcher = OffensiveStringMatcher()
        if opt['safety'] == 'classifier' or opt['safety'] == 'all':
            self.offensive_classifier = OffensiveLanguageClassifier()

        self.self_offensive = False

    def check_offensive(self, text):
        """
        Check if text is offensive using string matcher and classifier.
        """
        if text == '':
            return False
        if (
            hasattr(self, 'offensive_string_matcher')
            and text in self.offensive_string_matcher
        ):
            return True
        if hasattr(self, 'offensive_classifier') and text in self.offensive_classifier:
            return True

        return False

    def observe(self, msg):
        """
        Observe bot reply if and only if it passes.
        """
        if self.self_offensive:
            # User was offensive, they must try again
            return

        # Now check if bot was offensive
        bot_offensive = self.check_offensive(msg.get('text', ''))
        if not bot_offensive:
            # View bot message
            print(
                display_messages(
                    [msg],
                    add_fields=self.opt.get('display_add_fields', ''),
                    prettify=self.opt.get('display_prettify', False),
                    verbose=self.opt.get('verbose', False),
                )
            )
            msg.force_set('bot_offensive', False)
        else:
            msg.force_set('bot_offensive', True)
            print(OFFENSIVE_BOT_REPLY)

    def get_reply(self):
        reply_text = input(colorize('请输入:', 'field') + ' ')
        reply_text = reply_text.replace('\\n', '\n')
        from transformers import pipeline
        en_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
        zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
        translation =  zh_en(reply_text)
        return translation[0]['translation_text']

    def act(self):
        # get human reply
        reply = Message(
            {
                'id': self.getID(),
                'label_candidates': self.fixedCands_txt,
                'episode_done': False,
            }
        )
        reply_text = self.get_reply()

        # check if human reply is offensive
        self.self_offensive = self.check_offensive(reply_text)
        while self.self_offensive:
            print(OFFENSIVE_USER_REPLY)
            reply_text = self.get_reply()
            # check if human reply is offensive
            self.self_offensive = self.check_offensive(reply_text)

        # check for episode done
        if '[DONE]' in reply_text or self.opt.get('single_turn', False):
            raise StopIteration

        # set reply text
        reply['text'] = reply_text

        # check if finished
        if '[EXIT]' in reply_text:
            self.finished = True
            raise StopIteration

        return reply
