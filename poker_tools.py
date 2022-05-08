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
