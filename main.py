import sys
import random
import os
import argparse
from PIL import Image
from PIL import ImageStat
import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_evolution(fitness_history, output_dir, base_name):
    if not fitness_history:
        print("⚠️ Nu exista date de fitness pentru a genera graficul.")
        return

    # grupare pe generatie
    from collections import defaultdict
    gen_dict = defaultdict(list)
    for entry in fitness_history:
        gen_dict[entry['generation']].append(entry)

    generations = sorted(gen_dict.keys())
    best_avg = []
    avg_avg = []
    worst_avg = []

    for gen in generations:
        best_avg.append(np.mean([e['best'] for e in gen_dict[gen]]))
        avg_avg.append(np.mean([e['avg'] for e in gen_dict[gen]]))
        worst_avg.append(np.mean([e['worst'] for e in gen_dict[gen]]))

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_avg, label='Best Fitness (medie)', color='green')
    plt.plot(generations, avg_avg, label='Average Fitness (medie)', color='blue')
    plt.plot(generations, worst_avg, label='Worst Fitness (medie)', color='red')
    plt.xlabel('Generatie')
    plt.ylabel('Fitness')
    plt.title('Evolutia Fitness-ului Algoritmului Genetic (medie pe generatie)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f"{base_name}_fitness_evolution.png")
    plt.savefig(plot_path)
    print(f"📊 Grafic fitness salvat in: {plot_path}")
    plt.close()

w = 2
h = 4

grayscale_ramp = r"""@%#*+=-:. """ [::-1]

#grayscale_ramp = r"""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. """ [::-1]

# > 100 - bright
brightness_threshold = 100 # 0-255

default_population_size = len(grayscale_ramp)
default_generations = 5

# Fitness tracking
fitness_history = []
generation_stats = []

# analiza threshold 
def analyze_threshold_impact(image, threshold_values=[50, 80, 100, 128, 150, 180]):
    print(f"\n📊 ANALIZA THRESHOLD BRIGHTNESS:")
    print("="*60)
    
    width, height = image.size
    total_tiles = ((width + w - 1) // w) * ((height + h - 1) // h)
    
    for threshold in threshold_values:
        bright_tiles = 0
        dark_tiles = 0
        
        for y in range(0, height, h):
            for x in range(0, width, w):
                tile = image.crop((x, y, min(x + w, width), min(y + h, height)))
                brightness = getBrightness(tile)
                
                if brightness > threshold:
                    bright_tiles += 1
                else:
                    dark_tiles += 1
        
        bright_percent = (bright_tiles / total_tiles) * 100
        dark_percent = (dark_tiles / total_tiles) * 100
        
        print(f"Threshold {threshold:3d}: 🔆{bright_tiles:3d}({bright_percent:5.1f}%) | 🌑{dark_tiles:3d}({dark_percent:5.1f}%)")

def brightness2char(brightness, x, y, image):
    if brightness > brightness_threshold:
        return genetic_algorithm_for_bright_parts(x, y, image)
    else:
        return grayscale_ramp[int(brightness / 255 * (len(grayscale_ramp) - 1))]


# brightness de la un tile
def getBrightness(img):
    stat = ImageStat.Stat(img)
    return stat.mean[0]


def generate_random_ascii_char():
    return random.choice(grayscale_ramp)

# verifica daca se potrivesc
def fitness_char(char, x, y, image):
    tile = image.crop((x, y, x + w, y + h))
    b = getBrightness(tile)
    c = grayscale_ramp.index(char)
    diff = abs(b - (c * (255 / len(grayscale_ramp))))
    return -diff

def mutate_char(char):
    return random.choice(grayscale_ramp)

def crossover_char(parent1, parent2):
    return parent1 if random.random() < 0.5 else parent2


def genetic_algorithm_for_bright_parts(x, y, image, population_size=None, generations=None, mutation_rate=0.1):
    if population_size is None:
        population_size = default_population_size
    if generations is None:
        generations = default_generations
    
    population = [generate_random_ascii_char() for _ in range(population_size)]
    local_fitness_history = []
    
    for generation in range(generations):
        scored = [(fitness_char(ind, x, y, image), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Calculeaza statistici fitness pentru generatia curenta
        fitness_values = [score for score, _ in scored]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        worst_fitness = min(fitness_values)
        
        # Salveaza statisticile
        gen_stats = {
            'generation': generation + 1,
            'best': best_fitness,
            'avg': avg_fitness,
            'worst': worst_fitness
        }
        local_fitness_history.append(gen_stats)
        
        print(f"Generation {generation + 1}, Best: {best_fitness:.2f}, Avg: {avg_fitness:.2f}", end='\r')

        progress = (generation + 1) / generations
        bar_length = 20
        block = int(round(bar_length * progress))
        text = f"[{'█' * block}{'░' * (bar_length - block)}] {int(progress * 100)}% Gen:{generation + 1}/{generations}"
        print(text, end='\r')

        population = []
        for i in range(population_size // 2):
            population.append(scored[i][1])
            population.append(crossover_char(scored[i][1], scored[i+1][1]))
        
        for i in range(len(population)):
            if random.random() < mutation_rate:
                population[i] = mutate_char(population[i])
    
    # Adauga datele locale la istoricul global
    fitness_history.extend(local_fitness_history)
                
    best = sorted([(fitness_char(ind, x, y, image), ind) for ind in population], key=lambda x: x[0], reverse=True)[0][1]
    return best


def create_grayscale_image(pixel_values, output_path=None, scale=10):
    height = len(pixel_values)
    width = len(pixel_values[0]) if height > 0 else 0
    
    img = Image.new('L', (width * scale, height * scale))
    pixels = []
    
    for y in range(height):
        for _ in range(scale):
            row_pixels = []
            for x in range(width):
                for _ in range(scale):
                    row_pixels.append(pixel_values[y][x])
            pixels.extend(row_pixels)
    
    img.putdata(pixels)
    
    if output_path:
        img.save(output_path)
        print(f"📸 Imagine grayscale salvata: {output_path}")
    
    return img

def create_binary_image(pixel_values, threshold=128, output_path=None, scale=10):
    height = len(pixel_values)
    width = len(pixel_values[0]) if height > 0 else 0
    
    img = Image.new('1', (width * scale, height * scale))
    pixels = []
    
    for y in range(height):
        for _ in range(scale):
            row_pixels = []
            for x in range(width):
                binary_value = 1 if pixel_values[y][x] >= threshold else 0
                for _ in range(scale):
                    row_pixels.append(binary_value)
            pixels.extend(row_pixels)
    
    img.putdata(pixels)
    
    if output_path:
        img.save(output_path)
        print(f"⚫ Imagine binara salvata: {output_path}")
    
    return img

def image_to_pixel_matrix(image):
    import numpy as np
    width, height = image.size
    img_array = np.array(image)
    
    pixel_values = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(int(img_array[y, x]))
        pixel_values.append(row)
    
    return pixel_values

def save_processed_images(image, base_name, output_dir="output", threshold=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pixel_matrix = image_to_pixel_matrix(image)
    
    processed_path = os.path.join(output_dir, f"{base_name}_processed.png")
    image.save(processed_path)
    print(f"🖼️  Imagine procesata salvata: {processed_path}")
    
    grayscale_path = os.path.join(output_dir, f"{base_name}_grayscale.png")
    create_grayscale_image(pixel_matrix, grayscale_path, scale=10)
    
    binary_path = os.path.join(output_dir, f"{base_name}_binary.png")
    create_binary_image(pixel_matrix, threshold, binary_path, scale=10)
    
    total_pixels = len(pixel_matrix) * len(pixel_matrix[0])
    white_pixels = sum(1 for row in pixel_matrix for pixel in row if pixel >= threshold)
    black_pixels = total_pixels - white_pixels
    
    print(f"\n📊 STATISTICI IMAGINE:")
    print(f"  Total pixeli: {total_pixels}")
    print(f"  Pixeli albi (>={threshold}): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
    print(f"  Pixeli negri (<{threshold}): {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")

# argumente
def parse_arguments():
    parser = argparse.ArgumentParser(description='ASCII Art Generator cu Algoritm Genetic')
    parser.add_argument('image', help='Calea catre imaginea de procesat')
    parser.add_argument('--analyze', action='store_true', help='Analizeaza impactul threshold-ului')
    parser.add_argument('--save', action='store_true', help='Salveaza imaginile procesate')
    parser.add_argument('--fitness', action='store_true', help='Analizeaza evolutia fitness-ului')
    parser.add_argument('--population', type=int, default=default_population_size, 
                       help=f'Dimensiunea populatiei pentru algoritmul genetic (default: {default_population_size})')
    parser.add_argument('--generations', type=int, default=default_generations, 
                       help=f'Numarul de generatii pentru algoritmul genetic (default: {default_generations})')
    
    return parser.parse_args()

if len(sys.argv) < 2:
    print("ASCII Art Generator cu Algoritm Genetic")
    print("Utilizare: python main.py <imagine> [--analyze] [--save] [--fitness] [--population N] [--generations N]")
    print("\nExemple:")
    print("  python main.py photo.jpg")
    print("  python main.py portrait.png --analyze")
    print("  python main.py image.jpg --save")
    print("  python main.py image.jpg --fitness")
    print("  python main.py image.jpg --analyze --save --fitness")
    print("  python main.py image.jpg --population 20 --generations 10 --fitness")
    print("\nPentru analiza fitness detaliata, foloseste --fitness")
    sys.exit(1)

args = parse_arguments()

default_population_size = args.population
default_generations = args.generations

save_images = args.save
analyze_threshold = args.analyze
analyze_fitness = args.fitness

image = Image.open(args.image)
original_image_path = args.image
base_name = os.path.splitext(os.path.basename(original_image_path))[0]

# convert grayscale
image = image.convert("L")
width, height = image.size

# nr de tiles
num_tiles_x = (width + w - 1) // w
num_tiles_y = (height + h - 1) // h
total_tiles = num_tiles_x * num_tiles_y

# parametrii algoritm genetic
print(f"The image will be split into {total_tiles} tiles.")
print(f"Parametri Algoritm Genetic: Populatie={default_population_size}, Generatii={default_generations}")


x = 0
y = 0
outputPath = "output.txt"
outputFile = open(outputPath, "w")

# reset la isoricul fitness pentru fiecare rulare
fitness_history = []

while (y < height):
    while (x < width):
        tile = image.crop((x, y, x + w, y + h))
        brightness = getBrightness(tile)
        outputFile.write(brightness2char(brightness, x, y, image))
        x += w
    x = 0
    y += h
    outputFile.write("\n")
outputFile.close()

if analyze_threshold:
    analyze_threshold_impact(image)
    print(f"\nThreshold curent: {brightness_threshold}")
    print("Pentru a schimba threshold-ul, editeaza brightness_threshold in cod")

if analyze_fitness and fitness_history:
    print(f"\nGenerez graficul evolutiei fitness-ului...")
    plot_fitness_evolution(fitness_history, "output", base_name)
elif analyze_fitness and not fitness_history:
    print(f"\nNu exista date de fitness (toate tile-urile au fost sub threshold)")

if save_images:
    print(f"\nSalvez imaginile procesate...")
    save_processed_images(image, base_name, "output", brightness_threshold)

print(f"\nGata! ASCII art-ul tau este salvat in {outputPath}")
print("Poti sa-l vezi si in browser deschizand view_ascii.html")
print(f"Procesat {total_tiles} tile-uri cu succes!")
print(f"Folosit: Populatie={default_population_size}, Generatii={default_generations}")

if analyze_fitness and fitness_history:
    bright_tiles_count = len(fitness_history) // default_generations if fitness_history else 0
    print(f"Tile-uri procesate cu algoritm genetic: {bright_tiles_count}")
    print(f"Grafic fitness salvat in directorul 'output/'")
elif analyze_fitness:
    print(f"Pentru a genera graficul fitness, foloseste --fitness")
if save_images:
    print(f"Imaginile procesate sunt in directorul 'output/'")

