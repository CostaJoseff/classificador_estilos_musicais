from youtubesearchpython import VideosSearch 
from all_functions import search_and_download
from threading import Thread
from logger import log
import os

global stop_thread
stop_thread = False

def stop_thread_flag():
    return stop_thread

generos = {
    'rock': [
        'The Beatles', 'Led Zeppelin', 'Pink Floyd', 'Queen', 'Nirvana', 
        'The Rolling Stones', 'AC/DC', 'Foo Fighters', 'U2', 'Metallica',
        'The Clash', 'David Bowie', 'The Ramones', 'Deep Purple', 'Aerosmith', 
        'Van Halen', 'The Kinks', 'Fleetwood Mac', 'The Velvet Underground', 'Rush', 
        'The Eagles', 'Lynyrd Skynyrd', 'Yes', 'Rage Against The Machine', 'Pearl Jam', 
        'Green Day', 'The Black Keys', 'Red Hot Chili Peppers', 'Soundgarden', 'The White Stripes'
    ],
    'jazz': [
        'Miles Davis', 'John Coltrane', 'Louis Armstrong', 'Billie Holiday', 'Duke Ellington', 
        'Charlie Parker', 'Thelonious Monk', 'Chet Baker', 'Ella Fitzgerald', 'Stan Getz',
        'Wynton Marsalis', 'Herbie Hancock', 'Bill Evans', 'Max Roach', 'Art Blakey', 
        'Sonny Rollins', 'Count Basie', 'Oscar Peterson', 'Dexter Gordon', 'Wayne Shorter', 
        'Chick Corea', 'Cannonball Adderley', 'Sarah Vaughan', 'Norah Jones', 'Keith Jarrett', 
        'Esperanza Spalding', 'Pat Metheny', 'Brad Mehldau', 'Tom Harrell', 'Joshua Redman'
    ],
    'reggae': [
        'Bob Marley', 'Damian Marley', 'Sean Paul', 'Shaggy', 'Ziggy Marley', 
        'Toots and the Maytals', 'Burning Spear', 'Steel Pulse', 'Jimmy Cliff', 'Protoje',
        'Peter Tosh', 'Bunny Wailer', 'Black Uhuru', 'Gregory Isaacs', 'The Gladiators', 
        'Alton Ellis', 'Third World', 'The Mighty Diamonds', 'Chronixx', 'Morgan Heritage', 
        'Capleton', 'Sizzla', 'Luciano', 'Tarrus Riley', 'Jah Cure', 'Busy Signal', 
        'Vybz Kartel', 'Skip Marley', 'Koffee'
    ],
    'funk': [
        'James Brown', 'George Clinton', 'Prince', 'Sly and the Family Stone', 
        'Kool & The Gang', 'Earth, Wind & Fire', 'Chic', 'Parliament', 'The Meters', 'Rick James',
        'Bootsy Collins', 'Maceo Parker', 'Tower of Power', 'Ohio Players', 'Curtis Mayfield', 
        'The Isley Brothers', 'The Commodores', 'Barry White', 'Marvin Gaye', 'The Spinners', 
        'War', 'Average White Band', 'Prince and the Revolution', 'The Temptations', 'Stevie Wonder', 
        'Lenny Kravitz', 'Anderson .Paak', 'Vulfpeck', 'Mark Ronson', 'Janelle Monáe'
    ],
    'hip hop': [
        'Drake', 'Kendrick Lamar', 'J. Cole', 'Cardi B', 'Travis Scott', 
        'Nicki Minaj', 'Lil Nas X', 'Post Malone', 'Migos', 'Lil Wayne',
        'Jay-Z', 'Nas', 'Eminem', 'Snoop Dogg', 'Tupac Shakur', 
        'The Notorious B.I.G.', 'Missy Elliott', 'Lil Kim', 'A$AP Rocky', 'Future', 
        'Dr. Dre', 'Tyler, The Creator', 'Kanye West', 'Logic', 'Childish Gambino', 
        'Travis Barker', 'Run-D.M.C.', 'Lil Uzi Vert', 'Chance The Rapper', '21 Savage'
    ],
    'pagode': [
        'Thiaguinho', 'Sorriso Maroto', 'Péricles', 'Dilsinho', 'Ferrugem', 
        'Xande de Pilares', 'Revelação', 'Grupo Pixote', 'Turma do Pagode',
        'Zeca Pagodinho', 'Raça Negra', 'Samba de Saia', 'Fundo de Quintal', 'Catinguelê', 
        'Imaginasamba', 'Grupo Exalta Samba', 'Harmonia do Samba', 'Jorge Aragão', 'Chorão', 
        'Os Travessos', 'Jeito Moleque', 'Dilema', 'Pique Novo', 'Aviões do Forró', 
        'Patusco', 'Tropa de Elite', 'Cavaco e Viola', 'Os Kachorros', 'Dudu Nobre'
    ],
    'dubstep': [
        'Skrillex', 'Excision', 'Zeds Dead', 'Flux Pavilion', 'Knife Party', 
        'Virtual Riot', 'Borgore', 'Adventure Club', 'Doctor P', 'Rusko',
        'Bassnectar', 'Datsik', '12th Planet', 'Downlink', 'Chasing Status', 
        'The Glitch Mob', 'Savoy', 'Caspa', 'Skism', 'Jauz', 
        'Zomboy', 'Eptic', 'Pegboard Nerds', 'Ganja White Night', 'Muzzy', 
        'Snails', 'PhaseOne', 'Must Die!', 'Kill The Noise', 'Trampa'
    ],
    'house': [
        'Daft Punk', 'Frankie Knuckles', 'Deadmau5', 'Calvin Harris', 'Avicii', 
        'Tiësto', 'David Guetta', 'Eric Prydz', 'Axwell Ingrosso', 'Martin Garrix',
        'Kaskade', 'Solomun', 'John Digweed', 'Carl Cox', 'Steve Aoki', 
        'Sander Van Doorn', 'Hardwell', 'Nicky Romero', 'Oliver Heldens', 'Duke Dumont', 
        'Gorgon City', 'Zedd', 'Jamie Jones', 'Claptone', 'MK', 
        'Loco Dice', 'Tchami', 'Chris Lake', 'Green Velvet', 'Bob Sinclar'
    ],
    'bossa nova': [
        'João Gilberto', 'Antonio Carlos Jobim', 'Astrud Gilberto', 'Stan Getz', 
        'Vinícius de Moraes', 'Elis Regina', 'Nara Leão', 'Tom Jobim', 'Carlos Lyra', 'Baden Powell',
        'Sérgio Mendes', 'Roberto Menescal', 'Luiz Bonfá', 'Edu Lobo', 'Dori Caymmi', 
        'Marco Antônio Guimarães', 'Gal Costa', 'Caetano Veloso', 'Tom Zé', 'Milton Nascimento', 
        'Bebel Gilberto', 'Flora Purim', 'Lana Del Rey', 'Maria Rita', 'Tânia Maria', 
        'Leila Pinheiro', 'Ana Carolina', 'Paula Morelenbaum', 'Sivuca', 'Célia'
    ],
    'merengue': [
        'Juan Luis Guerra', 'Elvis Crespo', 'Wilfrido Vargas', 'Omega', 'Típico Júnior', 
        'Banda Real', 'El Chaval de la Bachata', 'Yoskar Sarante', 'Héctor Acosta', 'Diómedes Díaz',
        'Johnny Ventura', 'Félix Manuel', 'Sergio Vargas', 'Los Hermanos Rosario', 'Milly Quezada', 
        'Margarita La Diosa de la Cumbia', 'La India', 'Tito Rojas', 'Wilfrido Vargas', 'Santo Domingo Merengue All Stars',
        'Carlos Vives', 'Vikina', 'Bony Cepeda', 'El Torito', 'Jhonny Ventura y su Combo', 
        'Ritchie Valens', 'Oscar de León', 'Luis Vargas', 'Banda Gorda'
    ],
    'tango': [
        'Carlos Gardel', 'Astor Piazzolla', 'Aníbal Troilo', 'Osvaldo Pugliese', 
        'Ricardo Tanturi', 'Juan D"Arienzo', 'Horacio Salgán', 'Enrico Macias', 
        'María de Buenos Aires', 'Alfredo De Angelis',
        'Roberto Goyeneche', 'Luis Mariano', 'Carlos Di Sarli', 'Juan Carlos Cobián', 
        'Osvaldo Fresedo', 'Raúl Garello', 'Francisco Canaro', 'Aníbal Arias', 'Néstor Marconi', 
        'Ricardo Vilca', 'Orquesta Típica Fernández Fierro', 'Banda Sinfónica Municipal de Buenos Aires', 
        'Tango Connection', 'Pedro Laurenz', 'María Volonté', 'Atilio Stampone', 'Luis Salinas', 
        'Rodrigo y Gabriela', 'Elbio Fernández', 'Chango Spasiuk'
    ],
}

musicas_por_artista = 5
saida_base = 'musicas'

ignore_list = ["rock"]
try:
    for genero, artistas in generos.items():
        if genero in ignore_list:
            input(f"Ignorando {genero}")
            continue
        print(f"Baixando musica do genero {genero}")
        threads = []
        for i, artista in enumerate(artistas):
            
            pasta = os.path.join(saida_base, genero)
            os.makedirs(pasta, exist_ok=True)

            search_query = f"{artista} official music"
            resultados = VideosSearch(search_query, limit=musicas_por_artista * 5).result()

            # search_and_download(resultados, pasta)
            thread = Thread(target=search_and_download, args=[resultados, pasta, stop_thread_flag])
            thread.start()
            threads.append(thread)
        

        print(f"Aguardando conclusão de theads")
        for i, thread in enumerate(threads):
            thread.join()
            i = i+1
            log(f"{genero} ", f"{i}/{len(artistas)} concluídos", i, len(artistas), bar_size=30)
            
        print("[ V ]")
except KeyboardInterrupt:
    stop_thread = True
    print(f"Finalizando {len(threads)} threads")
    for i, thread in enumerate(threads):
        thread.join()
        i = i+1
        log(f"Finalizando threads", f"{i}/{len(threads)} concluídos", i, len(threads), bar_size=30)
