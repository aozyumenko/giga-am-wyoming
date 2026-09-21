#!python3

import asyncio
import logging
import numpy as np
from onnx_asr import load_model
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk
from wyoming.info import AsrModel, AsrProgram, Info, Attribution
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("WyomingGigaAM")


logger.info("Загрузка модели GigaAM v3 ONNX...")
model = load_model("gigaam-v3-e2e-ctc")
logger.info("Модель успешно загружена!")


# Заранее подготавливаем объект информации о нашем сервере для Home Assistant
SERVER_INFO = Info(
    asr=[AsrProgram(
        name="GigaAM-v3",
        description="Salute Developers GigaAM-v3 Speech-to-Text via ONNX",
        attribution=Attribution(name="Salute Developers", url="https://github.com/salute-developers/GigaAM"),
        installed=True,
        version="3.0",
        models=[
            AsrModel(
                name="gigaam-v3-e2e-ctc",
                description="GigaAM v3 End-to-End CTC Model",
                attribution=Attribution(name="Salute Developers", url="https://github.com/salute-developers/GigaAM"),
                installed=True,
                languages=["ru"],
                version="3.0",
            )
        ],
    )]
)


class GigaAMSpeechHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_buffer = bytearray()
        try:
            peer = self.writer.get_extra_info('peername')
            logger.info(f"[СЕТЬ] Новый клиент подключился: {peer}")
        except Exception:
            logger.info("[СЕТЬ] Новый клиент подключился (не удалось определить IP)")

    async def handle_event(self, event: Event) -> bool:
        logger.debug(f"[СОБЫТИЕ] Получено событие типа: {event.type}")

        if event.type == "describe":
            logger.info("[ОТВЕТ] Отправляем информацию о сервере (SERVER_INFO) в Home Assistant...")
            await self.write_event(SERVER_INFO.event())
            return False

        if event.type == "audio-start":
            logger.info("[ПРОЦЕСС] Начался новый аудиопоток. Очистка буфера.")
            self.audio_buffer = bytearray()
            return True

        # --- НАДЕЖНЫЙ СБОР АУДИОДАННЫХ ЧЕРЕЗ ИНСТРУМЕНТЫ WYOMING ---
        if event.type == "audio-chunk":
            try:
                # Преобразуем базовое событие в объект AudioChunk
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.extend(chunk.audio)
            except Exception as e:
                logger.error(f"[ОШИБКА] Не удалось извлечь аудио из чанка: {e}")
            return True

        if event.type == "audio-stop":
            logger.info(f"[ПРОЦЕСС] Аудиопоток завершен. Буфер: {len(self.audio_buffer)} байт. Запуск GigaAM...")

            if len(self.audio_buffer) == 0:
                logger.warning("[ПРОЦЕСС] Буфер пуст, отправляем пустой текст.")
                await self.write_event(Transcript(text="").event())
                return False

            # Превращаем PCM в float32
            audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                logger.info("[ИНФЕРЕНС] Передаем данные в GigaAM ONNX...")
                text = model.recognize(audio_data)
                logger.info(f"[УСПЕХ] Результат распознавания: '{text}'")
            except Exception as e:
                logger.error(f"[ОШИБКА] Ошибка внутри GigaAM: {e}", exc_info=True)
                text = ""

            await self.write_event(Transcript(text=text).event())
            logger.info("[СЕТЬ] Ответ отправлен клиенту. Закрываем сессию.")
            return False

        return True

    async def disconnect(self):
        logger.info("[СЕТЬ] Клиент отключился от сервера.")
        await super().disconnect()


async def main():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10500")
    logger.info("Wyoming GigaAM сервер готов к работе на порту 10500...")
    await server.run(lambda *args, **kwargs: GigaAMSpeechHandler(*args, **kwargs))


if __name__ == "__main__":
    asyncio.run(main())
