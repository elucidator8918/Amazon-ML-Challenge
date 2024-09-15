entity_unit_map = {
    "width": {
        "centimetre",
        "foot",
        "millimetre",
        "metre",
        "inch",
        "yard"
    },
    "depth": {
        "centimetre",
        "foot",
        "millimetre",
        "metre",
        "inch",
        "yard"
    },
    "height": {
        "centimetre",
        "foot",
        "millimetre",
        "metre",
        "inch",
        "yard"
    },
    "item_weight": {
        "milligram",
        "kilogram",
        "microgram",
        "gram",
        "ounce",
        "ton",
        "pound"
    },
    "maximum_weight_recommendation": {
        "milligram",
        "kilogram",
        "microgram",
        "gram",
        "ounce",
        "ton",
        "pound"
    },
    "voltage": {
        "millivolt",
        "kilovolt",
        "volt"
    },
    "wattage": {
        "kilowatt",
        "watt"
    },
    "item_volume": {
        "cubic foot",
        "microlitre",
        "cup",
        "fluid ounce",
        "centilitre",
        "imperial gallon",
        "pint",
        "decilitre",
        "litre",
        "millilitre",
        "quart",
        "cubic inch",
        "gallon"
    }
}

# Variations for the given mapping
unit_variations = {
    'centimetre': ['cm', 'centimetre', 'centimeter', 'centimeters', 'centimetres'],
    'millimetre': ['mm', 'millimetre', 'millimeter', 'millimeters', 'millimetres'],
    'metre': ['m', 'metre', 'meter', 'meters'],
    'inch': ['"', 'in', 'inch', 'inches'],
    'foot': ["'", 'ft', 'foot', 'feet'],
    'yard': ['yd', 'yard', 'yards'],
    'milligram': ['mg', 'milligram', 'milligrams'],
    'gram': ['g', 'gram', 'grams'],
    'kilogram': ['kg', 'kilogram', 'kilograms'],
    'microgram': ['μg', 'mcg', 'microgram', 'micrograms'],
    'ounce': ['oz', 'ounce', 'ounces'],
    'pound': ['lb', 'lbs', 'pound', 'pounds'],
    'ton': ['ton', 'tons', 'tonne', 'tonnes'],
    'millilitre': ['ml', 'millilitre', 'milliliter', 'milliliters', 'millilitres'],
    'litre': ['l', 'liter', 'litre', 'liters', 'litres'],
    'centilitre': ['cl', 'centilitre', 'centiliter', 'centiliters', 'centilitres'],
    'decilitre': ['dl', 'decilitre', 'deciliter', 'deciliters', 'decilitres'],
    'microlitre': ['μl', 'microlitre', 'microliter', 'microliters', 'microlitres'],
    'gallon': ['gal', 'gallon', 'gallons'],
    'imperial gallon': ['imperial gallon', 'imperial gallons'],
    'quart': ['qt', 'quart', 'quarts'],
    'pint': ['pt', 'pint', 'pints'],
    'cup': ['cup', 'cups'],
    'fluid ounce': ['fl oz', 'fluid ounce', 'fluid ounces'],
    'cubic foot': ['cu ft', 'cubic foot', 'cubic feet'],
    'cubic inch': ['cu in', 'cubic inch', 'cubic inches'],
    'volt': ['v', 'volt', 'volts'],
    'millivolt': ['mv', 'millivolt', 'millivolts'],
    'kilovolt': ['kv', 'kilovolt', 'kilovolts'],
    'watt': ['w', 'watt', 'watts'],
    'kilowatt': ['kw', 'kilowatt', 'kilowatts'],
    'ampere': ['a', 'amp', 'ampere', 'amperes'],

}

# Extract all unique units
# all_units = set()
# for units in entity_unit_map.values():
#     all_units.update(units)

# print("All Units:", all_units)
