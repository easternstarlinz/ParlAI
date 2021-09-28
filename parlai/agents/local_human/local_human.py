#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent does gets the local keyboard input in the act() function.

Example: parlai eval_model -m local_human -t babi:Task1k:1 -dt valid
"""
from typing import List, Union
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.utils.misc import display_messages, load_cands
from parlai.utils.strings import colorize


class LocalHumanAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Local Human Arguments')
        agent.add_argument(
            '-fixedCands',
            '--local-human-candidates-file',
            default=None,
            type=str,
            help='File of label_candidates to send to other agent',
        )
        agent.add_argument(
            '--single_turn',
            type='bool',
            default=False,
            help='If on, assumes single turn episodes.',
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False
        self.finished = False
        self.fixedCands_txt = load_cands(self.opt.get('local_human_candidates_file'))
        print(
            colorize(
                "Enter [DONE] if you want to end the episode, [EXIT] to quit.",
                'highlight',
            )
        )

    def epoch_done(self):
        return self.finished

    def observe(self, msg):
        print(
            display_messages(
                [msg],
                add_fields=self.opt.get('display_add_fields', ''),
                prettify=self.opt.get('display_prettify', False),
                verbose=self.opt.get('verbose', False),
            )
        )

    def act(self):
        reply = Message()
        reply['id'] = self.getID()
        try:
            reply_text = input(colorize("请输入:", 'text') + ' ')
        except EOFError:
            self.finished = True
            return {'episode_done': True}

        reply_text = reply_text.replace('\\n', '\n')
        from transformers import pipeline
        #en_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
        zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
        translation =  zh_en(reply_text)
        reply_text =  translation[0]['translation_text']
        reply['episode_done'] = False
        if self.opt.get('single_turn', False):
            reply.force_set('episode_done', True)
        reply['label_candidates'] = self.fixedCands_txt
        if '[DONE]' in reply_text:
            # let interactive know we're resetting
            raise StopIteration
        reply['text'] = reply_text
        if '[EXIT]' in reply_text:
            self.finished = True
            raise StopIteration
        return reply
    def respond(
        self, text_or_message: Union[str, Message], **other_message_fields
    ) -> str:
        """
        An agent convenience function which calls the act() and provides a string
        response to a text or message field.
        :param Union[str, Message] text_or_message:
            A string for the 'text' field or a message which MUST
            comprise of the 'text' field apart from other fields.
        :param kwargs other_message_fields:
            Provide fields for the message in the form of keyword arguments.
        :return:
            Agent's response to the message.
        :rtype:
            str
        """
        if isinstance(text_or_message, str):
            observation = Message(text=text_or_message, **other_message_fields)
        else:
            observation = Message(**text_or_message, **other_message_fields)
            if 'text' not in observation:
                raise RuntimeError('The agent needs a \'text\' field in the message.')

        if 'episode_done' not in observation:
            observation['episode_done'] = True
        agent = self.clone()
        agent.observe(observation)
        response = agent.act()
        return response['text']+"123"
    
    def episode_done(self):
        return self.episodeDone
