from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)

# инициализация устройства и загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "model"
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)


def detoxify_text(text):
    """
    Перефразирование текста с использованием T5 модели
    Args:
        text: Исходный текст
    Returns:
        str: Перефразированный текст
    """
    # Токенизация входного текста
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(device)

    # Генерация нового текста
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Обработка GET и POST запросов:
    GET - возвращает HTML страницу
    POST - принимает JSON с текстом и возвращает перефразированную версию
    """
    if request.method == 'GET':
        return render_template('index.html')

    # Получение и обработка текста из POST запроса
    text = request.get_json().get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        return jsonify({'paraphrases': [detoxify_text(text)]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)


