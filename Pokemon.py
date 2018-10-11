class Pokemon(object):

    # The class "constructor" - It's actually an initializer
    def __init__(self, number, name, type1, type2, total, hp, attack, defense, special_attack, special_defense, speed,
                 generation, is_legendary, color, has_mega_evolution, height, weight, body_style):
        self.number = number
        self.name = name
        self.type1 = type1
        self.type2 = type2
        self.total = total
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.special_attack = special_attack
        self.special_defense = special_defense
        self.speed = speed
        self.generation = generation
        self.is_legendary = is_legendary
        self.color = color
        self.has_mega_evolution = has_mega_evolution
        self.height = height
        self.weight = weight
        self.body_style = body_style

# def make_student(number, name, type1, type2, total, hp, attack, defense, special_attack, special_defense, speed,
#                  generation, is_legendary, color, has_mega_evolution, height, weight, body_style):
#     pokemon = Pokemon(number, name, type1, type2, total, hp, attack, defense, special_attack, special_defense, speed,
#                       generation, is_legendary, color, has_mega_evolution, height, weight, body_style)
#     return pokemon
