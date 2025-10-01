from fastapi import FastAPI, Request
import uvicorn
import json

gsi_test_app = FastAPI()

@gsi_test_app.post("/")
async def recibir_datos_gsi(request: Request):
    """
    Endpoint que escucha y simplemente imprime CUALQUIER dato que reciba de CS2.
    """
    try:
        data = await request.json()
        print("--- ¡DATOS RECIBIDOS DE CS2! ---")
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"Error al procesar la petición: {e}")
    
    return {"status": "ok"}

if __name__ == "__main__":
    print("Iniciando servidor de prueba GSI en http://127.0.0.1:3000...")
    print("Por favor, inicia CS2 y entra a una partida (con bots o real).")
    print("Si todo está bien, deberías ver datos JSON impresos en esta ventana.")
    print("Presiona Ctrl+C para detener.")
    uvicorn.run(gsi_test_app, host="127.0.0.1", port=3000, log_level="info")