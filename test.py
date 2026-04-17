# test.py - テスト用サンプルコード

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("0で割ることはできません")
    return a / b


# テスト実行
if __name__ == "__main__":
    print("=== 計算テスト ===")
    print(f"add(3, 5) = {add(3, 5)}")
    print(f"subtract(10, 4) = {subtract(10, 4)}")
    print(f"multiply(6, 7) = {multiply(6, 7)}")
    print(f"divide(20, 4) = {divide(20, 4)}")

    try:
        divide(10, 0)
    except ValueError as e:
        print(f"エラー確認: {e}")