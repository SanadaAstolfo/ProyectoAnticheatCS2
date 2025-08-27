import psutil
import socket
import time

def encontrar_servidor_cs2():
    """
    Busca el proceso cs2.exe y devuelve la tupla (IP, Puerto) del servidor.
    Devuelve (None, None) si no lo encuentra.
    """
    for proc in psutil.process_iter(['pid', 'name']):
        if 'cs2.exe' in proc.info['name']:
            try:
                conexiones = proc.net_connections(kind='inet')
                for conn in conexiones:
                    if conn.status == psutil.CONN_ESTABLISHED and conn.raddr:
                        ip_servidor = conn.raddr.ip
                        puerto_servidor = conn.raddr.port
                        return ip_servidor, puerto_servidor
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return None, None

def medir_latencia_tcp(destino, puerto, timeout=1):
    """
    Mide la latencia intentando una conexión TCP a un destino y puerto específicos.
    """
    if not destino or not puerto:
        return None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        tiempo_inicio = time.perf_counter()
        resultado = sock.connect_ex((destino, puerto))
        tiempo_fin = time.perf_counter()
        
        sock.close()

        if resultado == 0:
            latencia_ms = (tiempo_fin - tiempo_inicio) * 1000
            return latencia_ms
        else:
            return None
    except socket.gaierror:
        return None

print("Iniciando monitoreo de latencia de CS2... Presiona Ctrl+C para detener.")

while True:
    ip_servidor, puerto_servidor = encontrar_servidor_cs2()
    
    if ip_servidor and puerto_servidor:
        print(f"Partida activa detectada. Servidor en: {ip_servidor}:{puerto_servidor}")
        latencia = medir_latencia_tcp(ip_servidor, puerto=puerto_servidor)
        
        if latencia is not None:
            print(f"Latencia TCP estimada: {latencia:.2f} ms")
        else:
            print("No se pudo medir la latencia TCP (el puerto del juego podría no responder).")
            
    else:
        print("Esperando a que inicie una partida de CS2...")
    
    time.sleep(5)