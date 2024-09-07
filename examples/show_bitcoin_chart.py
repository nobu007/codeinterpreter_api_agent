from datetime import datetime

from codeinterpreterapi.session import CodeInterpreterSession


def main() -> None:
    with CodeInterpreterSession(is_local=True) as session:
        session.start_local()
        currentdate = datetime.now().strftime("%Y-%m-%d")

        response = session.generate_response(f"Plot the bitcoin chart of 2023 YTD (today is {currentdate})")

        # prints the text and shows the image
        response.show()


if __name__ == "__main__":
    main()
