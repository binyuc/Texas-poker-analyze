# 首先定义常量
# 我们把牌面成为 poker_face，把牌力称为 poker_num
import random
import time
import numpy as np
import pandas as pd
import math
from scipy.special import comb, perm
import itertools
from tqdm import tqdm

pd.set_option('display.max_rows', 1000)

poker_num_reference = "0123456789TJQKA"
poker_face_reference = ("S", "C", "H", "D")
# 黑桃 S - Spade，梅花C - Club，方块D - Diamonds，红桃H - Hearts
poker_face_to_num_reference = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5,
                               "4": 4, "3": 3, "2": 2}
poker_num_to_face_reference = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'J', 9: '9', 8: '8', 7: '7', 6: '6', 5: '5',
                               4: '4', 3: '3', 2: '2'}
hand_type_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                      "Straight", "Flush", "Full House", "Four of a Kind",
                      "Straight Flush", "Royal Flush")

global full_cards_list


def build_new_poker_bag():
    global full_cards_list
    ''':return 构建新的卡包，52张牌'''
    poker_number_list = '2,3,4,5,6,7,8,9,T,J,Q,K,A'.split(',')
    poker_color_list = 'H,C,D,S'.split(',')

    full_cards_list = []
    for color in poker_color_list:
        for number in poker_number_list:
            one_card = number + color
            full_cards_list.append(one_card)
    print('构建新卡包，洗牌完成，共有{}张牌'.format(len(full_cards_list)))
    return full_cards_list


def generate_rest_of_poker_bag(hands_list, board_deck):
    global full_cards_list
    '''读取已有的牌，并剔除现有的牌，返回剩余的牌'''
    for each_hands in hands_list:
        if each_hands is not None:
            for each_card in each_hands.split():
                full_cards_list.remove(each_card)
    for each_card in board_deck:
        full_cards_list.remove(each_card)
    print(len(full_cards_list))
    return full_cards_list


def number_ranks(cards: str):
    '''
    :param: cards → str
    :return：[14,12,11,10,7] 按大小排序的牌面列表列表'''
    ranks = [poker_num_reference.index(number) for number, color in cards.split()]
    ranks.sort(reverse=True)
    return ranks


class Basic_rule:
    '''基础方程工具类，用于判断排序'''

    def __init__(self, cards):
        self.cards = ' '.join(sorted(cards.split()))
        pass

    def same_color_count(self, same_num):
        '''判断花色数量和花色，返回的是花色，列表'''
        color_list = [color for number, color in self.cards.split()]
        res = []
        for i in color_list:
            if color_list.count(i) == same_num:
                res.append(i)
        return list(set(res))

    def same_number_count(self, same_num):
        '''判断数字数量,返回的是牌面'''
        number_list = [number for number, color in self.cards.split()]
        res = []
        for i in number_list:
            if number_list.count(i) == same_num:
                res.append(i)
        return sorted(list(set(res)), reverse=True)

    def analyze_straight(self):
        '''desc 牌是不同的5张数值，所以最大-最小=4'''
        card_list = number_ranks(self.cards)
        return len(set(number_ranks(self.cards))) == 5 and max(card_list) - min(card_list) == 4

    def analyze_same_color(self):
        '''desc 判断是否同花'''
        if len(self.same_color_count(5)) > 0:
            return True

    def analyze_boom(self):
        '''desc判断是否为炸弹'''
        if len(self.same_number_count(4)) > 0:
            return True

    def analyze_fullhouse(self):
        '''desc 判断是否为葫芦fullhouse 3带2'''
        if len(self.same_number_count(3)) > 0 and len(self.same_number_count(2)) > 0:
            return True

    def analyze_three_set(self):
        '''desc 判断是否为3条'''
        if len(self.same_number_count(3)) > 0:
            return True

    def analyze_one_pair(self):
        '''desc 判断是否为一对'''
        if len(self.same_number_count(2)) > 0:
            return True

    def analyze_two_pair(self):
        '''desc 判断是否为两对'''
        if len(self.same_number_count(2)) == 2:
            return True

    def analyze_one_hand_rank(self):
        '''
        :desc:分析每个人的手牌大小总函数
        :returns:
        '''
        cards_list = number_ranks(self.cards)
        # 皇家同花顺
        if self.analyze_same_color() and self.analyze_straight() and max(cards_list) == 14:
            return {'hand_card_score': 10, 'first_card': poker_num_to_face_reference[max(cards_list)],
                    'first_card_score': max(cards_list), 'second_card': None, 'second_card_score': None,
                    'hand_card_type_name': '皇家同花顺', 'cards_detail': self.cards}

        # 同花顺
        if self.analyze_same_color() and self.analyze_straight():
            return {'hand_card_score': 9, 'first_card': poker_num_to_face_reference[max(cards_list)],
                    'first_card_score': max(cards_list), 'second_card': None, 'second_card_score': None,
                    'hand_card_type_name': '同花顺', 'cards_detail': self.cards}

        # 四条
        if self.analyze_boom():
            return {'hand_card_score': 8, 'first_card': self.same_number_count(4)[0],
                    'first_card_score': poker_face_to_num_reference[self.same_number_count(4)[0]], 'second_card': None,
                    'second_card_score': None, 'hand_card_type_name': '炸弹四条', 'cards_detail': self.cards}

        # 葫芦
        if self.analyze_fullhouse():
            return {'hand_card_score': 7, 'first_card': self.same_number_count(3)[0],
                    'first_card_score': poker_face_to_num_reference[self.same_number_count(3)[0]],
                    'second_card': self.same_number_count(2)[0],
                    'second_card_score': poker_face_to_num_reference[self.same_number_count(2)[0]],
                    'hand_card_type_name': '葫芦',
                    'cards_detail': self.cards}
        # 同花
        if self.analyze_same_color():
            return {'hand_card_score': 6, 'first_card': poker_num_to_face_reference[max(cards_list)],
                    'first_card_score': max(cards_list), 'second_card': None, 'second_card_score': None,
                    'hand_card_type_name': '同花', 'cards_detail': self.cards}

        # 顺子
        if self.analyze_straight():
            return {'hand_card_score': 5, 'first_card': poker_num_to_face_reference[max(cards_list)],
                    'first_card_score': max(cards_list), 'second_card': None, 'second_card_score': None,
                    'hand_card_type_name': '顺子', 'cards_detail': self.cards}

        # 三条
        if self.analyze_three_set():
            return {'hand_card_score': 4, 'first_card': self.same_number_count(3)[0],
                    'first_card_score': poker_face_to_num_reference[self.same_number_count(3)[0]], 'second_card': None,
                    'second_card_score': None, 'hand_card_type_name': '三条', 'cards_detail': self.cards}

        # 两对
        if self.analyze_two_pair():
            if poker_face_to_num_reference[self.same_number_count(2)[0]] > poker_face_to_num_reference[
                self.same_number_count(2)[1]]:
                return {'hand_card_score': 3, 'first_card': self.same_number_count(2)[0],
                        'first_card_score': poker_face_to_num_reference[self.same_number_count(2)[0]],
                        'second_card': self.same_number_count(2)[1],
                        'second_card_score': poker_face_to_num_reference[self.same_number_count(2)[1]],
                        'hand_card_type_name': '两对',
                        'cards_detail': self.cards}
            else:
                return {'hand_card_score': 3, 'first_card': self.same_number_count(2)[1],
                        'first_card_score': poker_face_to_num_reference[self.same_number_count(2)[1]],
                        'second_card': self.same_number_count(2)[0],
                        'second_card_score': poker_face_to_num_reference[self.same_number_count(2)[0]],
                        'hand_card_type_name': '两对',
                        'cards_detail': self.cards}

        # 一对
        if self.analyze_one_pair():
            return {'hand_card_score': 2, 'first_card': self.same_number_count(2)[0],
                    'first_card_score': poker_face_to_num_reference[self.same_number_count(2)[0]], 'second_card': None,
                    'second_card_score': None, 'hand_card_type_name': '一对', 'cards_detail': self.cards}

        # 高牌
        return {'hand_card_score': 1, 'first_card': poker_num_to_face_reference[max(cards_list)],
                'first_card_score': max(cards_list), 'second_card': None, 'second_card_score': None,
                'hand_card_type_name': '高牌', 'cards_detail': self.cards}


def compare_hands(hands_list):
    '''比较两个手牌的大小'''
    hands_result_list = [None] * len(hands_list)
    for mark, single_hand in enumerate(hands_list):
        temp = Basic_rule(single_hand).analyze_one_hand_rank()
        hands_result_list[mark] = (temp['hand_card_score'], temp['first_card_score'], temp['second_card_score'])

    best_hand = max(hands_result_list)
    winning_player_index = hands_result_list.index(best_hand) + 1
    # Check for ties
    if best_hand in hands_result_list[winning_player_index:]:
        return 0
    return winning_player_index


def run_simulation(hands_list, board_deck):
    num_players = len(hands_list)

    result_histograms, winner_list = [], [0] * (num_players + 1)
    for _ in range(num_players):
        result_histograms.append([0] * len(hand_type_rankings))

    board_length = 0 if board_deck is None else len(board_deck)

    if (None, None) in hands_list:
        unknown_index = hands_list.index((None, None))
        for filler_hole_cards in itertools.combinations(deck, 2):  # todo 这一部是为对方构建底牌，把每个底牌轮进去算
            hands_list[unknown_index] = filler_hole_cards
            deck_list = list(deck)
            deck_list.remove(filler_hole_cards[0])
            deck_list.remove(filler_hole_cards[1])
            # todo 解析参数
            # todo generate_boards - 函数
            '''
            :param tuple(deck_list) 元组，剩余牌
            hold_cards_list: 手牌和对方的手牌，这个是组合， ''((Ks, Jc), (As, 8s))''
            num = 1
            board_length = 3 or 4 or 5 翻牌区的卡牌数量
            given_board = 翻牌区的列表 [Ac, Kh, Ts]
            winner_list = [33902, 619963, 124275] 猜测是牌局结果： tie win lose
            result_list = 10种牌型结果 [[0, 36456, 27841, 5196, 12561, 0, 2015, 81, 0, 0], [8568, 35532, 22895, 4946, 8984, 681, 2398, 143, 3, 0]]
            '''
            holdem_functions.find_winner(generate_boards, tuple(deck_list),
                                         tuple(hole_cards_list), num,
                                         board_length, given_board, winner_list,
                                         result_histograms)

    else:
        holdem_functions.find_winner(generate_boards, deck, hole_cards, num,
                                     board_length, given_board, winner_list,
                                     result_histograms)
    # if verbose: todo 没啥用 verbose
    '''
    result_histograms,每种牌型的赢次数 = [[0, 0, 990, 0, 0, 0, 0, 0, 0, 0], [0, 532, 352, 70, 12, 0, 23, 1, 0, 0]]
    winner_list = [70, 688, 232] 总次数
    '''
    players_histograms = holdem_functions.calc_histogram(result_histograms, winner_list)
    return [holdem_functions.find_winning_percentage(winner_list), players_histograms]


def poker_run(hold_cards, enemy_hold_cards, board_deck):
    ''':param
    hold_cards : 玩家手上的牌
    enemy_hold_Cards : 对方手上的牌
    board_deck : 已翻开的牌,翻牌flop，转牌turn 河牌river
    :return
    win_rate:打平、赢、输的百分比概率
    what_to_win：赢的每种牌型的可能性、需要的卡
    what_to_lose：输的每种牌型的可能性、输掉的卡
    '''
    full_cards_list = build_new_poker_bag()
    generate_rest_of_poker_bag([hold_cards, enemy_hold_cards], board_deck)
    return print('done')


if __name__ == '__main__':
    # hold_cards = "AS KH"
    hold_cards = "AS AH TH TS 9C"
    hold_cards_2 = "5S 6C 7C JH 2D"
    enemy_hold_cards = None
    board_deck = ["AC", "KD", "TC"]
    poker_run(hold_cards=hold_cards, enemy_hold_cards=enemy_hold_cards, board_deck=board_deck)
