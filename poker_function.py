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
from loguru import logger

pd.set_option('display.max_rows', 1000)

poker_num_reference = "0123456789TJQKA"
poker_color_reference = {"S": 0, "C": 1, "H": 2, "D": 3}
# 黑桃 S - Spade，梅花C - Club，方块D - Diamonds，红桃H - Hearts
poker_face_to_num_reference = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5,
                               "4": 4, "3": 3, "2": 2}
poker_num_to_face_reference = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'J', 9: '9', 8: '8', 7: '7', 6: '6', 5: '5',
                               4: '4', 3: '3', 2: '2'}
hand_type_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                      "Straight", "Flush", "Full House", "Four of a Kind",
                      "Straight Flush", "Royal Flush")


def generate_suit_board(flat_board, flush_index):
    histogram = [Card(card).value for card in flat_board
                 if Card(card).color_index == flush_index]
    histogram.sort(reverse=True)
    return histogram


def detect_straight_flush(suit_board):
    contiguous_length, fail_index = 1, len(suit_board) - 5
    # Won't overflow list because we fail fast and check ahead
    for index, elem in enumerate(suit_board):
        current_val, next_val = elem, suit_board[index + 1]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if index >= fail_index:
                if (index == fail_index and next_val == 5 and
                        suit_board[0] == 14):
                    return True, 5
                break
            contiguous_length = 1
    return False,


def get_high_cards(histogram_board):
    return histogram_board[:5]


# Returns a list of two tuples of the form: (value of card, frequency of card)
def preprocess(histogram):
    return [(14 - index, frequency) for index, frequency in
            enumerate(histogram) if frequency]


# Returns tuple: (Is there a straight?, high card)
def detect_straight(histogram_board):
    contiguous_length, fail_index = 1, len(histogram_board) - 5
    # Won't overflow list because we fail fast and check ahead
    for index, elem in enumerate(histogram_board):
        current_val, next_val = elem[0], histogram_board[index + 1][0]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if index >= fail_index:
                if (index == fail_index and next_val == 5 and
                        histogram_board[0][0] == 14):
                    return True, 5
                break
            contiguous_length = 1
    return False,


# Returns tuple of the two highest kickers that result from the three of a kind
def detect_three_of_a_kind_kickers(histogram_board):
    kicker1 = -1
    for elem in histogram_board:
        if elem[1] != 3:
            if kicker1 == -1:
                kicker1 = elem[0]
            else:
                return kicker1, elem[0]


# Returns the highest kicker available
def detect_highest_kicker(histogram_board):
    for elem in histogram_board:
        if elem[1] == 1:
            return elem[0]


# Returns tuple: (kicker1, kicker2, kicker3)
def detect_pair_kickers(histogram_board):
    kicker1, kicker2 = -1, -1
    for elem in histogram_board:
        if elem[1] != 2:
            if kicker1 == -1:
                kicker1 = elem[0]
            elif kicker2 == -1:
                kicker2 = elem[0]
            else:
                return kicker1, kicker2, elem[0]


def detect_highest_quad_kicker(histogram_board):
    for elem in histogram_board:
        if elem[1] < 4:
            return elem[0]


def detect_hand(hold_cards, given_board, poker_color_res_list,
                poker_num_res_list, max_poker_color_cnt):
    # Determine if flush possible. If yes, four of a kind and full house are
    # impossible, so return royal, straight, or regular flush.
    if max_poker_color_cnt >= 3:
        flush_index = poker_color_res_list.index(max_poker_color_cnt)
        for card in hold_cards:
            if Card(card).color_index == flush_index:
                max_poker_color_cnt += 1
        if max_poker_color_cnt >= 5:
            flat_board = given_board
            flat_board.extend(hold_cards)
            suit_board = generate_suit_board(flat_board, flush_index)
            result = detect_straight_flush(suit_board)
            if result[0]:
                return (8, result[1]) if result[1] != 14 else (9,)
            return 5, get_high_cards(suit_board)

    # Add hole cards to histogram data structure and process it
    full_histogram = poker_num_res_list[:]
    for card in hold_cards:
        full_histogram[14 - Card(card).value] += 1
    histogram_board = preprocess(full_histogram)

    # Find which card value shows up the most and second most times
    current_max, max_val, second_max, second_max_val = 0, 0, 0, 0
    for item in histogram_board:
        val, frequency = item[0], item[1]
        if frequency > current_max:
            second_max, second_max_val = current_max, max_val
            current_max, max_val = frequency, val
        elif frequency > second_max:
            second_max, second_max_val = frequency, val

    # Check to see if there is a four of a kind
    if current_max == 4:
        return 7, max_val, detect_highest_quad_kicker(histogram_board)
    # Check to see if there is a full house
    if current_max == 3 and second_max >= 2:
        return 6, max_val, second_max_val
    # Check to see if there is a straight
    if len(histogram_board) >= 5:
        result = detect_straight(histogram_board)
        if result[0]:
            return 4, result[1]
    # Check to see if there is a three of a kind
    if current_max == 3:
        return 3, max_val, detect_three_of_a_kind_kickers(histogram_board)
    if current_max == 2:
        # Check to see if there is a two pair
        if second_max == 2:
            return 2, max_val, second_max_val, detect_highest_kicker(
                histogram_board)
        # Return pair
        else:
            return 1, max_val, detect_pair_kickers(histogram_board)
    # Check for high cards
    return 0, get_high_cards(histogram_board)


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
    return full_cards_list


def preprocess_hand_card_to_tuple(hand_card):
    '''处理手牌,返回元组'''
    if hand_card is None:
        return (None, None)
    return tuple(hand_card)


def preprocess_board(flat_board):
    ''':return
    poker_face_res_list:
    poker_num_res_list
    max(poker_face_res_list):
    '''
    poker_color_res_list, poker_num_res_list = [0] * 4, [0] * 13
    for card in flat_board:
        poker_num_res_list[14 - Card(card).value] += 1
        poker_color_res_list[Card(card).color_index] += 1
    return poker_color_res_list, poker_num_res_list, max(poker_color_res_list)


def compare_hands(result_list):
    '''比较两个手牌的大小'''
    # print(result_list)
    best_hand = max(result_list)
    # print(best_hand)
    winning_player_index = result_list.index(best_hand) + 1
    # Check for ties
    if best_hand in result_list[winning_player_index:]:
        return 0
    return winning_player_index


def preprocess_player_hands(flat_board):
    suit_histogram, histogram = [0] * 4, [0] * 13
    # Reversing the order in histogram so in the future, we can traverse
    # starting from index 0
    for card in flat_board:
        histogram[14 - card.value] += 1
        suit_histogram[card.suit_index] += 1
    return suit_histogram, histogram, max(suit_histogram)


def generate_rest_of_poker_bag(hands_list, open_board_deck):
    '''读取已有的牌，并剔除现有的牌，返回剩余的牌'''
    for each_hands in hands_list:
        for card in each_hands:
            if card is not None:
                full_cards_list.remove(card)
    for card in open_board_deck:
        full_cards_list.remove(card)

    return


class Card:
    """用于声明牌的类，返回牌色和牌面"""

    def __init__(self, card_string):
        value, color = card_string[0], card_string[1]
        self.value = poker_face_to_num_reference[value]
        self.color_index = poker_color_reference[color]


def calc_histogram(result_histograms, winner_list):
    '''计算牌型的成功率和对应关系'''
    float_iterations = float(sum(winner_list))
    players_hist = []
    for player_index, histogram in enumerate(result_histograms):
        hist = {}
        for index, elem in enumerate(histogram):
            hist.update({hand_type_rankings[index]: float(elem) / float_iterations})
        players_hist.append(hist)
    return (players_hist)


# Returns the winning percentages
def find_winning_percentage(winner_list):
    '''计算tie win lose百分比'''
    float_iterations = float(sum(winner_list))
    percentages = []
    for num_wins in winner_list:
        winning_percentage = float(num_wins) / float_iterations
        percentages.append(winning_percentage)
    return {'tie': percentages[0],
            'win': percentages[1],
            'lose': percentages[2]}


def run_simulation(hands_list, open_board_deck):
    global full_card_list
    """:param
    hands_list: [我的手牌，对方手上的牌],[list, list]
    open_board_deck : 翻牌的牌
    """
    num_players = len(hands_list)

    card_type_result_histograms, tie_win_lose_list = [], [0] * (num_players + 1)
    for _ in range(num_players):
        card_type_result_histograms.append([0] * len(hand_type_rankings))

    board_length = 0 if open_board_deck is None else len(open_board_deck)
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

            find_winner(hands_list_tuple, board_length, deck_list, open_board_deck, tie_win_lose_list,
                        card_type_result_histograms)
    else:
        find_winner(hands_list_tuple, board_length, full_card_list, open_board_deck, tie_win_lose_list,
                    card_type_result_histograms)

    players_histograms = calc_histogram(card_type_result_histograms, tie_win_lose_list)
    return [find_winning_percentage(tie_win_lose_list), players_histograms]


def find_winner(hands_list_tuple, board_length, deck_list, open_board_deck, tie_win_lose_list,
                card_type_result_histograms):
    ''':param
    hands_list_tuple
    board_length： 3 board
    deck_list: 剩余的卡片list
    board_deck：翻牌区的牌
    '''
    # 总
    result_list = [None] * len(hands_list_tuple)

    for remaining_board in itertools.combinations(deck_list, 5 - board_length):
        if open_board_deck:
            board = open_board_deck[:]
            board.extend(remaining_board)
        else:
            board = remaining_board

        poker_color_res_list, poker_num_res_list, max_poker_color_cnt = (preprocess_board(board))

        for mark, hold_cards in enumerate(hands_list_tuple):
            result_list[mark] = detect_hand(hold_cards, board, poker_color_res_list,
                                            poker_num_res_list, max_poker_color_cnt)

        winner_index = compare_hands(result_list)
        tie_win_lose_list[winner_index] += 1
        # Increment what hand each player made
        for index, result in enumerate(result_list):
            card_type_result_histograms[index][result[0]] += 1


def poker_run(my_hold_cards: list, enemy_hold_cards: list, open_board_deck: list):
    """:param
    hold_cards : 玩家手上的牌
    enemy_hold_Cards : 对方手上的牌
    board_deck : 已翻开的牌,翻牌flop，转牌turn 河牌river
    :return
    win_rate:打平、赢、输的百分比概率
    what_to_win：赢的每种牌型的可能性、需要的卡
    what_to_lose：输的每种牌型的可能性、输掉的卡
    """
    begin_time = time.time()
    build_new_poker_bag()
    logger.info('构建新卡包，洗牌完成，共有{}张牌'.format(len(full_cards_list)))

    generate_rest_of_poker_bag(hands_list=[my_hold_cards, enemy_hold_cards], open_board_deck=open_board_deck)
    logger.info('剔除已知的牌，牌堆还剩{}张牌'.format(len(full_cards_list)))

    final_result = run_simulation([my_hold_cards, enemy_hold_cards], open_board_deck)

    logger.info('总程序运行完毕，共耗时{}'.format(time.time() - begin_time))

    return [logger.warning(i) for i in final_result]


if __name__ == '__main__':
    my_hold_cards = ['AS', '2S']
    enemy_hold_cards = [None, None]
    open_board_deck = ["KS", "QS", "TS"]
    poker_run(my_hold_cards=my_hold_cards, enemy_hold_cards=enemy_hold_cards, open_board_deck=open_board_deck)
