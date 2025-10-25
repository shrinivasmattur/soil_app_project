# 1_app_info.py

def get_fruit_info():
    fruit_data = {
        "Ramphala": {
            "description": "Ramphala, also known as the 'bullock's heart' or 'wild sweetsop', is a tropical fruit from the Annonaceae family. It has a heart-shaped, bumpy, reddish-brown skin and a sweet, creamy, and aromatic pulp with a slightly grainy texture.",
            "cultivation": "It thrives in warm, humid climates and well-drained soil. It is relatively drought-tolerant once established. It requires moderate watering and protection from frost, especially when young.",
            "harvest_season": "Generally ripens in the spring and early summer months.",
            "benefits_and_uses": """
* **Nutritional Value:** Rich in Vitamin C, antioxidants, potassium, and magnesium.
* **Culinary Uses:** Primarily eaten fresh. The creamy pulp is also excellent for making smoothies, milkshakes, and ice creams.
* **Traditional Medicine:** In some cultures, parts of the plant are used to treat digestive issues or boils.
            """,
            "image_url": "https://example.com/ramphala.jpg"
        },
        "Lakshmanaphala": {
            "description": "Lakshmanaphala, commonly known as Soursop or Graviola, is a prickly, green, heart-shaped fruit. Its flesh is juicy, white, and fibrous, with a unique flavor that is a combination of strawberry, pineapple, and citrus.",
            "cultivation": "It prefers tropical climates with high humidity and well-drained, slightly acidic soil. It is very sensitive to frost and low temperatures. Requires consistent moisture.",
            "harvest_season": "Can fruit year-round in optimal tropical climates, often with a peak season.",
            "benefits_and_uses": """
* **Nutritional Value:** Excellent source of Vitamin C, B vitamins (B1, B2), and dietary fiber.
* **Culinary Uses:** Extremely popular for making fresh juices, smoothies, sorbets, and dessert flavorings.
* **Traditional Medicine:** The leaves are famously brewed into a tea. The fruit and leaves have been studied for various potential health benefits.
            """,
            "image_url": "https://example.com/lakshmanaphala.jpg"
        },
        "Wood Apple": {
            "description": "The Wood Apple, or Bael, has a very hard, woody shell and a sticky, brown, aromatic pulp with small white seeds. The pulp has a unique sour, tangy, and slightly sweet flavor. It is an acquired taste for many.",
            "cultivation": "This is an extremely hardy fruit that can grow in a wide range of soil types, including poor soil. It is highly tolerant of drought, high temperatures, and is generally low-maintenance.",
            "harvest_season": "The fruits typically mature and are harvested in late spring to early summer (May-July).",
            "benefits_and_uses": """
* **Nutritional Value:** A good source of fiber, Vitamin C, beta-carotene, thiamine, and riboflavin.
* **Culinary Uses:** The pulp is rarely eaten fresh. It's famously made into a refreshing juice (sherbet) by adding water, sugar, and spices. Also used to make jams (murabba) and chutneys.
* **Traditional Medicine:** Highly valued in Ayurveda. The unripe fruit is used to treat digestive issues, while the ripe fruit pulp is known to be a good laxative.
            """,
            "image_url": "https://example.com/wood_apple.jpg"
        }
    }
    return fruit_data

if __name__ == '__main__':
    fruits = get_fruit_info()
    for fruit, info in fruits.items():
        print(f"--- {fruit} ---")
        print(f"Description: {info['description']}")
        print(f"Cultivation: {info['cultivation']}")
        print(f"Season: {info['harvest_season']}")
        print("Benefits & Uses:\n" + info['benefits_and_uses'])