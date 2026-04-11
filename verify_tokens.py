import os
from pathlib import Path
from transformers import AutoTokenizer

# Настройки
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR = Path("data")

def main():
    print(f"Загрузка токенизатора: {MODEL_NAME}...")
    # Загружаем токенизатор. Скачивание займет пару секунд.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. Считаем токены оригинального ядра (Condition A)
    path_a = DATA_DIR / "condition_A.txt"
    if not path_a.exists():
        print(f"Ошибка: Файл {path_a} не найден!")
        return
        
    text_a = path_a.read_text(encoding="utf-8")
    tokens_a = len(tokenizer.encode(text_a))
    
    # Вычисляем границы +/- 15%
    min_tokens = int(tokens_a * 0.85)
    max_tokens = int(tokens_a * 1.15)
    
    print("-" * 50)
    print(f"Condition A (Оригинал): {tokens_a} токенов")
    print(f"Допустимый диапазон (±15%): от {min_tokens} до {max_tokens} токенов")
    print("-" * 50)
    
    # Функция для проверки папок
    def check_folder(folder_name):
        folder_path = DATA_DIR / folder_name
        if not folder_path.exists():
            print(f"Папка {folder_path} не найдена.")
            return
            
        print(f"\nПроверка {folder_name}:")
        all_passed = True
        
        for file_path in sorted(folder_path.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            token_count = len(tokenizer.encode(text))
            
            if min_tokens <= token_count <= max_tokens:
                status = "✅ OK"
            else:
                status = "❌ FAIL"
                all_passed = False
                
            print(f"  {file_path.name}: {token_count} токенов -> {status}")
            
        return all_passed

    # Проверяем парафразы (B) и контрольные тексты (C)
    b_passed = check_folder("condition_B")
    c_passed = check_folder("condition_C")
    
    print("\n" + "=" * 50)
    if b_passed and c_passed:
        print("🎉 ВСЕ ТЕКСТЫ В ПРЕДЕЛАХ НОРМЫ! Можно запускать run.py")
    else:
        print("⚠️ ЕСТЬ ОТКЛОНЕНИЯ ПО ДЛИНЕ. Отредактируйте файлы с пометкой FAIL.")

if __name__ == "__main__":
    main()