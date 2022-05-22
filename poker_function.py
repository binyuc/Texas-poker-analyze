# 首先定义常量
# 我们把牌面成为 poker_face，把牌力称为 poker_num
import time
import itertools
from loguru import logger
from multiprocessing import Pool

# import pandas as pd 别导入，慢的一批
# from pathos.multiprocessing import ProcessingPool as Pool
# from poker_tools import *
# import dill
# import pickle
# pd.set_option('display.max_rows', 1000)

poker_num_reference = "0123456789TJQKA"
value_string = "AKQJT98765432"
poker_color_reference = {"S": 0, "C": 1, "H": 2, "D": 3}
# 黑桃 S - Spade，梅花C - Club，方块D - Diamonds，红桃H - Hearts
poker_face_to_num_reference = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5,
                               "4": 4, "3": 3, "2": 2}
poker_num_to_face_reference = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'J', 9: '9', 8: '8', 7: '7', 6: '6', 5: '5',
                               4: '4', 3: '3', 2: '2'}
hand_type_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                      "Straight", "Flush", "Full House", "Four of a Kind",
                      "Straight Flush", "Royal Flush")


class Poker:
    def __init__(self):
        self.card_type_result_histograms = None
        self.tie_win_lose_list = None
        self.result_list = []
        self.full_cards_list = []

    def build_new_poker_bag(self):
        """:return 构建新的卡包，52张牌"""
        poker_number_list = '2,3,4,5,6,7,8,9,T,J,Q,K,A'.split(',')
        poker_color_list = 'H,C,D,S'.split(',')

        for color in poker_color_list:
            for number in poker_number_list:
                one_card = number + color
                self.full_cards_list.append(one_card)
        return self.full_cards_list

    def generate_rest_of_poker_bag(self, hands_list, open_board_deck):
        """读取已有的牌，并剔除现有的牌，返回剩余的牌"""
        for each_hands in hands_list:
            for card in each_hands:
                if card is not None:
                    self.full_cards_list.remove(card)
        for card in open_board_deck:
            self.full_cards_list.remove(card)

    def run_simulation(self, hands_list, open_board_deck):
        """:param
        hands_list: [我的手牌，对方手上的牌],[list, list]
        open_board_deck : 翻牌的牌
        """
        num_players = len(hands_list)

        self.card_type_result_histograms, self.tie_win_lose_list = [], [0] * (num_players + 1)
        for _ in range(num_players):
            self.card_type_result_histograms.append([0] * len(hand_type_rankings))

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

            for filler_hole_cards in itertools.combinations(self.full_cards_list, 2):
                hands_list_tuple[unknown_index] = filler_hole_cards

                deck_list = self.full_cards_list.copy()
                deck_list.remove(filler_hole_cards[0])
                deck_list.remove(filler_hole_cards[1])

                # todo 现在迭代一次是0.02s，标准为0.008，一半
                self.find_winner(hands_list_tuple, board_length, deck_list, open_board_deck, self.tie_win_lose_list,
                                 self.card_type_result_histograms)



        else:
            self.find_winner(hands_list_tuple, board_length, self.full_cards_list, open_board_deck,
                             self.tie_win_lose_list,
                             self.card_type_result_histograms)
        # print(self.tie_win_lose_list, self.card_type_result_histograms)
        players_histograms = calc_histogram(self.card_type_result_histograms, self.tie_win_lose_list)
        return [find_winning_percentage(self.tie_win_lose_list), players_histograms]

    def run_simulation_mult_run(self, hands_list, open_board_deck):
        """多进程版
        """
        num_players = len(hands_list)

        self.card_type_result_histograms, self.tie_win_lose_list = [], [0] * (num_players + 1)
        for _ in range(num_players):
            self.card_type_result_histograms.append([0] * len(hand_type_rankings))

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
            pool = Pool(8)
            final = []

            for filler_hole_cards in itertools.combinations(self.full_cards_list, 2):
                hands_list_tuple[unknown_index] = filler_hole_cards

                deck_list = self.full_cards_list.copy()
                deck_list.remove(filler_hole_cards[0])
                deck_list.remove(filler_hole_cards[1])

                final.append(pool.apply(self.find_winner, args=(
                    hands_list_tuple, board_length, deck_list, open_board_deck, self.tie_win_lose_list,
                    self.card_type_result_histograms,)))
            pool.close()
            pool.join()

            # for i in final:
            #     self.tie_win_lose_list[0] += i.get()[0][0]
            #     self.tie_win_lose_list[1] += i.get()[0][1]
            #     self.tie_win_lose_list[2] += i.get()[0][2]
            for i in final:
                self.tie_win_lose_list[0] += i[0][0]
                self.tie_win_lose_list[1] += i[0][1]
                self.tie_win_lose_list[2] += i[0][2]
            print(self.tie_win_lose_list)

        else:
            self.find_winner(hands_list_tuple, board_length, self.full_cards_list, open_board_deck,
                             self.tie_win_lose_list,
                             self.card_type_result_histograms)
        print(self.tie_win_lose_list, self.card_type_result_histograms)
        players_histograms = calc_histogram(self.card_type_result_histograms, self.tie_win_lose_list)
        return [find_winning_percentage(self.tie_win_lose_list), players_histograms]

    def detect_hand(self, hold_cards, given_board, poker_color_res_list,
                    poker_num_res_list, max_poker_color_cnt):
        """:param
        hold_cards:[('7S', '2S'), ('2H', '3H')]
        given_board:['KS', 'QS', 'TS']
        poker_color_res_list [3, 0, 2, 0]
        poker_num_res_list [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        max_poker_color_cnt 3
        """
        if max_poker_color_cnt >= 3:
            flush_index = poker_color_res_list.index(max_poker_color_cnt)
            for card in hold_cards:
                # if Card(card).color_index == flush_index:
                if poker_color_reference[card[1]] == flush_index:
                    max_poker_color_cnt += 1
            if max_poker_color_cnt >= 5:
                given_board.extend(hold_cards)
                suit_board = generate_suit_board(given_board, flush_index)
                result = detect_straight_flush(suit_board)
                if result[0]:
                    return (8, result[1]) if result[1] != 14 else (9,)
                return 5, get_high_cards(suit_board)

        # Add hole cards to histogram data structure and process it
        full_histogram = poker_num_res_list[:]
        for card in hold_cards:
            # full_histogram[14 - Card(card).value] += 1
            full_histogram[14 - poker_face_to_num_reference[card[0]]] += 1
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

    def find_winner(self, hands_list_tuple, board_length, deck_list, open_board_deck, tie_win_lose_list,
                    card_type_result_histograms):
        """:param
        hands_list_tuple
        board_length： 3 board
        deck_list: 剩余的卡片list
        board_deck：翻牌区的牌
        """
        # 总
        result_list = [None] * len(hands_list_tuple)
        for remaining_board in itertools.combinations(deck_list, 5 - board_length):

            if open_board_deck:
                board = open_board_deck[:]
                board.extend(remaining_board)
            else:
                board = remaining_board

            poker_color_res_list, poker_num_res_list, max_poker_color_cnt = preprocess_board(board)

            for mark, hold_cards in enumerate(hands_list_tuple):
                result_list[mark] = self.detect_hand(hold_cards, board, poker_color_res_list,
                                                     poker_num_res_list, max_poker_color_cnt)

            winner_index = compare_hands(result_list)
            tie_win_lose_list[winner_index] += 1
            # Increment what hand each player made
            for index, result in enumerate(result_list):
                card_type_result_histograms[index][result[0]] += 1
        return tie_win_lose_list, card_type_result_histograms

    def after_process_result(self, final_result, time_diff):
        """ 结构化final _result """
        temp = final_result.copy()
        final_result = {
            # overall
            'my_tie_rate': temp[0]['tie'], 'my_win_rate': temp[0]['win'], 'my_lose_rate': temp[0]['lose'],
            # my_card
            'my_high_card_prob': temp[1][0]['High Card'], 'my_pair_prob': temp[1][0]['Pair'],
            'my_two_pair_prob': temp[1][0]['Two Pair'],
            'my_three_kind_prob': temp[1][0]['Three of a Kind'],
            'my_straight_prob': temp[1][0]['Straight'], 'my_flush_prob': temp[1][0]['Flush'],
            'my_full_house_prob': temp[1][0]['Full House'],
            'my_four_kind_prob': temp[1][0]['Four of a Kind'], 'my_straight_flush_prob': temp[1][0]['Straight Flush'],
            'my_royal_flush_prob': temp[1][0]['Royal Flush'],
            # enemy_card
            'enemy_high_card_prob': temp[1][1]['High Card'], 'enemy_pair_prob': temp[1][1]['Pair'],
            'enemy_two_pair_prob': temp[1][1]['Two Pair'],
            'enemy_three_kind_prob': temp[1][1]['Three of a Kind'],
            'enemy_straight_prob': temp[1][1]['Straight'], 'enemy_flush_prob': temp[1][1]['Flush'],
            'enemy_full_house_prob': temp[1][1]['Full House'],
            'enemy_four_kind_prob': temp[1][1]['Four of a Kind'],
            'enemy_straight_flush_prob': temp[1][1]['Straight Flush'],
            'enemy_royal_flush_prob': temp[1][1]['Royal Flush'],
            #     time
            # 'time_diff': round(time_diff, 5)
        }
        for k, v in final_result.items():
            final_result[k] = str(round((v * 100), 2)) + '%'
        final_result['time_diff'] = round(time_diff, 5)
        logger.debug('最终结果{}'.format(final_result))
        return final_result

    def poker_run(self, my_hold_cards: list, enemy_hold_cards: list, open_board_deck: list):
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

        self.build_new_poker_bag()
        logger.info('构建新卡包，洗牌完成，共有{}张牌'.format(len(self.full_cards_list)))

        self.generate_rest_of_poker_bag(hands_list=[my_hold_cards, enemy_hold_cards], open_board_deck=open_board_deck)
        logger.info('剔除已知的牌，牌堆还剩{}张牌'.format(len(self.full_cards_list)))
        logger.info('开始模拟情景')

        # final_result = self.run_simulation_mult_run([my_hold_cards, enemy_hold_cards], open_board_deck)
        final_result = self.run_simulation([my_hold_cards, enemy_hold_cards], open_board_deck)
        logger.info('总程序运行完毕，共耗时{}'.format(time.time() - begin_time))
        [logger.warning(i) for i in final_result]
        return self.after_process_result(final_result, time.time() - begin_time)


def generate_suit_board(flat_board, flush_index):
    """:param，将同花的卡牌组合起来，返回一个由高到底的列表"""
    # histogram = [Card(card).value for card in flat_board
    #              if Card(card).color_index == flush_index]
    histogram = sorted([poker_face_to_num_reference[card[0]] for card in flat_board
                        if poker_color_reference[card[1]] == flush_index], reverse=True)
    # histogram.sort(reverse=True)
    return histogram


def detect_straight_flush(suit_board):
    contiguous_length, fail_index = 1, len(suit_board) - 5
    for mark, elem in enumerate(suit_board):
        current_val, next_val = elem, suit_board[mark + 1]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if mark >= fail_index:
                if (mark == fail_index and next_val == 5 and
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


def preprocess_hand_card_to_tuple(hand_card):
    """处理手牌,返回元组"""
    if hand_card is None:
        return (None, None)
    return tuple(hand_card)


def preprocess_board(flat_board):
    """:return
    poker_face_res_list:
    poker_num_res_list
    max(poker_face_res_list):
    """
    poker_color_res_list, poker_num_res_list = [0] * 4, [0] * 13
    for card in flat_board:
        # poker_num_res_list[14 - Card(card).value] += 1
        # poker_color_res_list[Card(card).color_index] += 1
        poker_num_res_list[14 - poker_face_to_num_reference[card[0]]] += 1
        poker_color_res_list[poker_color_reference[card[1]]] += 1
    return poker_color_res_list, poker_num_res_list, max(poker_color_res_list)


def compare_hands(result_list):
    """比较两个手牌的大小"""
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


# class Card:
#     """用于声明牌的类，返回牌色和牌面"""
#     def __init__(self, card_string):
#         value, self.color = card_string[0], card_string[1]
#         self.value = poker_face_to_num_reference[value]
#         self.color_index = poker_color_reference[self.color]

# def __repr__(self):
#     return value_string[14 - self.value] + self.color
#
# def __eq__(self, other):
#     if self is None:
#         return other is None
#     elif other is None:
#         return False
#     return self.value == other.value and self.color == other.color


def calc_histogram(result_histograms, winner_list):
    """计算牌型的成功率和对应关系"""
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
    """计算tie win lose百分比"""
    float_iterations = float(sum(winner_list))
    percentages = []
    for num_wins in winner_list:
        winning_percentage = float(num_wins) / float_iterations
        percentages.append(winning_percentage)
    return {'tie': percentages[0],
            'win': percentages[1],
            'lose': percentages[2]}


if __name__ == '__main__':
    my_hold_cards = ['7S', '2S']
    enemy_hold_cards = [None, None]
    open_board_deck = ["KS", "QS", "TS"]
    Poker().poker_run(my_hold_cards=my_hold_cards, enemy_hold_cards=enemy_hold_cards, open_board_deck=open_board_deck)
