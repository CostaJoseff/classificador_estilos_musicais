from youtubesearchpython import VideosSearch
from all_functions import search_and_download
from threading import Thread
from logger import log
import os, subprocess

global stop_thread
stop_thread = False


generos = {
    'rock': ['The Beatles', 'Led Zeppelin', 'Pink Floyd', 'Queen', 'Nirvana', 'The Rolling Stones', 'AC/DC', 'Foo Fighters', 'U2', 'Metallica'],
    'jazz': ['Miles Davis', 'John Coltrane', 'Louis Armstrong', 'Billie Holiday', 'Duke Ellington', 'Charlie Parker', 'Thelonious Monk', 'Chet Baker', 'Ella Fitzgerald', 'Stan Getz'],
    'reggae': ['Bob Marley', 'Damian Marley', 'Sean Paul', 'Shaggy', 'Ziggy Marley', 'Toots and the Maytals', 'Burning Spear', 'Steel Pulse', 'Jimmy Cliff', 'Protoje'],
    'funk': ['James Brown', 'George Clinton', 'Prince', 'Sly and the Family Stone', 'Kool & The Gang', 'Earth, Wind & Fire', 'Chic', 'Parliament', 'The Meters', 'Rick James'],
    'hip hop': ['Drake', 'Kendrick Lamar', 'J. Cole', 'Cardi B', 'Travis Scott', 'Nicki Minaj', 'Lil Nas X', 'Post Malone', 'Migos', 'Lil Wayne'],
    'pagode': ['Thiaguinho', 'Sorriso Maroto', 'Péricles', 'Dilsinho', 'Ludmilla', 'Ferrugem', 'Xande de Pilares', 'Revelação', 'Grupo Pixote', 'Turma do Pagode'],
    'dubstep': ['Skrillex', 'Excision', 'Zeds Dead', 'Flux Pavilion', 'Knife Party', 'Virtual Riot', 'Borgore', 'Adventure Club', 'Doctor P', 'Rusko'],
    'house': ['Daft Punk', 'Frankie Knuckles', 'Deadmau5', 'Calvin Harris', 'Avicii', 'Tiësto', 'David Guetta', 'Eric Prydz', 'Axwell Ingrosso', 'Martin Garrix'],
    'bossa nova': ['João Gilberto', 'Antonio Carlos Jobim', 'Astrud Gilberto', 'Stan Getz', 'Vinícius de Moraes', 'Elis Regina', 'Nara Leão', 'Tom Jobim', 'Carlos Lyra', 'Baden Powell'],
    'merengue': ['Juan Luis Guerra', 'Elvis Crespo', 'Wilfrido Vargas', 'Omega', 'Típico Júnior', 'Banda Real', 'El Chaval de la Bachata', 'Yoskar Sarante', 'Héctor Acosta', 'Diómedes Díaz'],
    'tango': ['Carlos Gardel', 'Astor Piazzolla', 'Aníbal Troilo', 'Osvaldo Pugliese', 'Ricardo Tanturi', 'Juan DArienzo', 'Horacio Salgán', 'Enrico Macias', 'María de Buenos Aires', 'Alfredo De Angelis'],
}

musicas_por_artista = 5
saida_base = 'musicas'

ignore_list = []
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
            thread = Thread(target=search_and_download, args=[resultados, pasta])
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
