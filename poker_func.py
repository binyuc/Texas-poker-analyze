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
from poker_tools import *
from poker_tools import Basic_rule
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
    '''读取已有的牌，并剔除现有的牌，返回剩余的牌'''
    for each_hands in hands_list:
        if each_hands is not None:
            for each_card in each_hands.split():
                full_cards_list.remove(each_card)
    for each_card in board_deck:
        full_cards_list.remove(each_card)

    return print('剔除已知的牌，牌堆还剩{}张牌'.format(len(full_cards_list)))


def preprocess_hand_card_to_tuple(hand_card):
    '''处理手牌,返回元组'''
    if hand_card is None:
        return (None, None)
    return tuple(hand_card.split())


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
    ''':param
    hands_list: [我的手牌，对方手上的牌],[str,str]
    board_deck : 翻牌的牌
    '''
    num_players = len(hands_list)

    card_type_result_histograms, tie_win_lose_list = [], [0] * (num_players + 1)
    for _ in range(num_players):
        card_type_result_histograms.append([0] * len(hand_type_rankings))

    board_length = 0 if board_deck is None else len(board_deck)
    ''':param
    card_type_result_histograms: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] 
    tie_win_lose_list: [0, 0, 0] 
    board_length: 3
    '''
    hands_list_tuple = [preprocess_hand_card_to_tuple(hands_list[0]), preprocess_hand_card_to_tuple(hands_list[1])]
    '''hands_list_tuple : [('AS', 'KH'), (None, None)]'''
    if (None, None) in hands_list_tuple:
        unknown_index = hands_list_tuple.index((None, None))
        # todo 这一部是为对方构建底牌，把每个底牌轮进去算
        for filler_hole_cards in itertools.combinations(full_cards_list, 2):
            hands_list_tuple[unknown_index] = filler_hole_cards

            deck_list = full_cards_list.copy()
            deck_list.remove(filler_hole_cards[0])
            deck_list.remove(filler_hole_cards[1])

            find_winner(tuple(hands_list_tuple), board_length, deck_list, board_deck)


def preprocess_player_hands(flat_board):
    suit_histogram, histogram = [0] * 4, [0] * 13
    # Reversing the order in histogram so in the future, we can traverse
    # starting from index 0
    for card in flat_board:
        histogram[14 - card.value] += 1
        suit_histogram[card.suit_index] += 1
    return suit_histogram, histogram, max(suit_histogram)


def find_winner(hands_list_tuple, board_length, deck_list, board_deck):
    result_list = [None] * len(hands_list_tuple)

    print(list(hands_list_tuple))
    for remaining_board in itertools.combinations(deck_list, 5 - board_length):
        if board_deck is not None:
            board = board_deck[:]
            board.extend(remaining_board)
        print(board)

        for mark, hold_cards in enumerate(list(hands_list_tuple)):
            result_list[mark] = print(Basic_rule(' '.join(hold_cards)).analyze_one_hand_rank())

    return


def poker_run(hold_cards: str, enemy_hold_cards: str, board_deck: list):
    ''':param
    hold_cards : 玩家手上的牌
    enemy_hold_Cards : 对方手上的牌
    board_deck : 已翻开的牌,翻牌flop，转牌turn 河牌river
    :return
    win_rate:打平、赢、输的百分比概率
    what_to_win：赢的每种牌型的可能性、需要的卡
    what_to_lose：输的每种牌型的可能性、输掉的卡
    '''
    begin_time = time.time()
    build_new_poker_bag()
    generate_rest_of_poker_bag(hands_list=[hold_cards, enemy_hold_cards], board_deck=board_deck)

    run_simulation([hold_cards, enemy_hold_cards], board_deck)
    print('总程序运行完毕，共耗时{}'.format(time.time() - begin_time))
    return


if __name__ == '__main__':
    hold_cards = "AS KH"
    # hold_cards = "AS AH TH TS 9C"
    # hold_cards_2 = "5S 6C 7C JH 2D"
    enemy_hold_cards = None
    board_deck = ["AC", "KD", "TC"]
    poker_run(hold_cards=hold_cards, enemy_hold_cards=enemy_hold_cards, board_deck=board_deck)
