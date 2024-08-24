import asyncio
from typing import Any, List, Optional

from codeboxapi.schema import CodeBoxStatus
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field


class File(BaseModel):
    name: str
    content: bytes

    @classmethod
    def from_path(cls, path: str) -> "File":
        if not path.startswith("/"):
            path = f"./{path}"
        with open(path, "rb") as f:
            path = path.split("/")[-1]
            return cls(name=path, content=f.read())

    @classmethod
    async def afrom_path(cls, path: str) -> "File":
        return await asyncio.to_thread(cls.from_path, path)

    @classmethod
    def from_url(cls, url: str) -> "File":
        import requests  # type: ignore

        r = requests.get(url)
        return cls(name=url.split("/")[-1], content=r.content)

    @classmethod
    async def afrom_url(cls, url: str) -> "File":
        import aiohttp  # type: ignore

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                return cls(name=url.split("/")[-1], content=await r.read())

    def save(self, path: str) -> None:
        if not path.startswith("/"):
            path = f"./{path}"
        with open(path, "wb") as f:
            f.write(self.content)

    async def asave(self, path: str) -> None:
        await asyncio.to_thread(self.save, path)

    def get_image(self) -> Any:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            print("Please install it with " "`pip install 'codeinterpreterapi[image_support]'`" " to display images.")
            exit(1)

        from io import BytesIO

        img_io = BytesIO(self.content)
        img = Image.open(img_io)

        # Convert image to RGB if it's not
        if img.mode not in ("RGB", "L"):  # L is for grayscale images
            img = img.convert("RGB")

        return img

    def show_image(self) -> None:
        img = self.get_image()
        # Display the image
        try:
            # Try to get the IPython shell if available.
            shell = get_ipython().__class__.__name__  # type: ignore
            # If the shell is in a Jupyter notebook or similar.
            if shell == "ZMQInteractiveShell" or shell == "Shell":
                from IPython.display import display  # type: ignore

                display(img)  # type: ignore
            else:
                img.show()
        except NameError:
            img.show()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"File(name={self.name})"


class CodeInput(BaseModel):
    filename: str
    code: str


class FileInput(BaseModel):
    filename: str


class UserRequest(HumanMessage):
    files: list[File] = []

    def __str__(self) -> str:
        return str(self.content)

    def __repr__(self) -> str:
        return f"UserRequest(content={self.content}, files={self.files})"


class CodeInterpreterResponse(AIMessage):
    """
    Response from the code interpreter agent.

    files: list of files to be sent to the user (File )
    code_log: list[tuple[str, str]] = []
    """

    files: Optional[list[File]] = []
    code_log: Optional[dict[str, str]] = []
    agent_name: Optional[str] = ""
    thought: Optional[str] = ""  # 中間的な思考

    def show(self) -> None:
        print("AI: ", self.content)
        for file in self.files:
            print("File: ", file.name)
            file.show_image()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"CodeInterpreterResponse(files={self.files}, agent_name={self.agent_name}, content={self.content})"


class SessionStatus(CodeBoxStatus):
    @classmethod
    def from_codebox_status(cls, cbs: CodeBoxStatus) -> "SessionStatus":
        return cls(status=cbs.status)

    def __repr__(self) -> str:
        return f"<SessionStatus status={self.status}>"


class CodeInterpreterPlan(BaseModel):
    '''単一PlanのAgentとTaskの説明です。
    PlanとAgentとTaskは常に1:1:1の関係です。
    '''

    agent_name: str = Field(
        description="Agentの名前です。タスクの名前も、このagent_nameと常に同じになり、primary keyとして使われます。利用可能な文字は[a-Z_]です。task likeな名前にしてください。"
    )
    task_description: str = Field(
        description="タスクの説明です。可能な範囲でpurpose, execution plan, input, outputの詳細を含めます。"
    )
    expected_output: str = Field(
        description="タスクの最終的な出力形式を明確に定義します。例えばjson/csvというフォーマットや、カラム名やサイズ情報です。"
    )


class CodeInterpreterPlanList(BaseModel):
    '''CodeInterpreterPlanの配列をもつ計画全体です。'''

    reliability: int = Field(
        description="計画の信頼度[0-100]です。100が完全な計画を意味します。50未満だと不完全計画でオリジナルの問題を直接llmに渡した方が良い結果になります。"
    )
    agent_task_list: List[CodeInterpreterPlan] = Field(description="CodeInterpreterPlanの配列です。")
