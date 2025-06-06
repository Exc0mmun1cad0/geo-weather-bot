## Geo-weather-bot (наш проект по предмету "Теоретическая информатика")

#### Цель:
Написать телегам-бота, который определяет температуру по фотографии с использованием технологий ML

#### Стек
Язык: 
- Python

ML framework: 
- PyTorch
 
Backend: 
- Aiogram - тг-бот
- S3 - для хранения картинок
- RabbitMQ - для отправки и получения задач на предсказание
- Redis - FSM для тг-бота 

#### ML-модели
- Классификатор (MobileNet_v3_large) для определения температурного диапазона. Их список:
    - [-30, -10]
    - [-10, 0]
    - [0, 10]
    - [10, 30]
- 4 регрессора (ResNet_18) для каждого диапазона

#### Схема работы:
1. Телеграм-бот получает изображение, генерирует для него id и кладет его в S3
2. ID изображения вместе с тг-ником пользователя, от которого оно пришло, кладется в очередь
3. С другого конца очереди сервис предсказаний читает сообщение, берёт картинку из S3 и даёт её ML-моделям для анализа
4. Результат отправляется пользователю в тг 



