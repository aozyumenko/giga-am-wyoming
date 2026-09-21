# giga-am-wyoming

Добавляем распознавание речи GigaAM-v3 в Home Assistant.

Используется сконвертированная в ONNX модель, которая скачивается с huggingface. Все красиво оформлено в Docker Compose. Можно запускать и напрямую в Linux, но это в сервисы оформляйте уже сами.

Как ставить.
1. sudo mkdir /opt/huggingface-data
2. sudo chown root:root /opt/huggingface-data
3. git clone https://github.com/aozyumenko/giga-am-wyoming.git
4. cd giga-am-wyoming
5. docker-compose up -d --build (или docker compose up -d --build)
6. смотрим, что на порту 10500 кто-то висит: sudo netstat -anp | grep 10500
7. настраиваем Home Assistant Wyoming Protocol (интеграция такая) на localhost:10500

Все, можно наслаждаться прекрасным распознаванием речи. Намного лучше Whishper'а, особенно для русского языка.
