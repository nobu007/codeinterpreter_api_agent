from codeinterpreterapi import CodeInterpreterSession

print("AI: Hello, I am the " "code interpreter agent.\n" "Ask me todo something and " "I will use python to do it!\n")

with CodeInterpreterSession() as session:
    while True:
        session.generate_response_sync(input("\nUser: ")).show()
