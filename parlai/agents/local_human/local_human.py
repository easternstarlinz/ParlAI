#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent does gets the local keyboard input in the act() function.

Example: parlai eval_model -m local_human -t babi:Task1k:1 -dt valid
"""

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
        agent.add_argument(
            '--foreign_en',
            type=str,
            default="Helsinki-NLP/opus-mt-zh-en",
            help='translate foreign language into English',
        )
        agent.add_argument(
            '--en_foreign',
            type=str,
            default="Helsinki-NLP/opus-mt-en-zh",
            help='translate English into foreign language',
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
        from transformers import pipeline
        if not 'en_foreign' in globals():
            global en_foreign
            en_foreign = pipeline("translation", model=self.opt["en_foreign"])
        translation =  en_foreign(msg['text'])
        #print(msg)
        print("AI尬聊："+translation[0]['translation_text'].replace("你的人:","你的人设:"))

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
        if not 'foreign_en' in globals():
            global foreign_en
            foreign_en = pipeline("translation", model=self.opt["foreign_en"])
        translation =  foreign_en(reply_text)
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

    
    def episode_done(self):
        return self.episodeDone
