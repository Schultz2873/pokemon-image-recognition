import csv
from Pokemon import Pokemon

with open('../cleaned/pokemon-stats-clean.csv') as file:
    pokemon_list = []

    reader = csv.reader(file)
    count = 0

    for row in reader:
        # set a count to skip the first line of test data
        if count != 0:
            pokemon = Pokemon(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10],
                              row[11], row[12], row[13], row[14], row[15], row[16], row[17])
            pokemon_list.append(pokemon)

        count = count + 1
    #    Output of list
    print()
    print("%s %6s %13s %10s %10s %13s %8s %8s %16s %18s %7s %11s %11s %9s %19s %9s %9s %13s" % (
        "Number", "Name", "Type1", "Type2", "Total", "Hit Points", "Attack", "Defense", "Special Attack",
        "Special Defense",
        "Speed", "Generation", "Legendary", "Color", "Mega Evolution", "Height", "Weight", "Body Type"))
    print()
    for pokemon in pokemon_list:
        print(
            f'{pokemon.number:8} {pokemon.name:12} {pokemon.type1:10} {pokemon.type2:10} {pokemon.total:8} {pokemon.hp:12} {pokemon.attack:7} {pokemon.defense:10} {pokemon.special_attack:16} {pokemon.special_defense:17} {pokemon.speed:7} {pokemon.generation:12} {pokemon.isLegendary:12} {pokemon.color:10} {pokemon.has_mega_evolution:19}{pokemon.height:9} {pokemon.weight:9} {pokemon.body_style:7}')
