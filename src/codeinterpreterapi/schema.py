import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from codeboxapi.schema import CodeBoxStatus
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

ToolsRenderer = Callable[[List[BaseTool]], str]


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
    code: str


class FileInput(BaseModel):
    filename: str


class CodeAndFileInput(BaseModel):
    filename: str
    code: str


class BashCommand(BaseModel):
    command: str


class ZoltraakInput(BaseModel):
    prompt: str = Field(
        default="このシステムを改善してください。",
        description="やりたいこと。曖昧な目標でも動作するし、具体的に指定すればピンポイントで編集や改善もできる。",
    )
    name: str = Field(
        default="codeinterpreter",
        description="処理対象の名前。対象がシステムの場合はディレクトリ名に使われる。対象がpythonファイルの場合はpythonファイル名に使われる。",
    )


class UserRequest(HumanMessage):
    files: list[File] = []

    def __str__(self) -> str:
        return str(self.content)

    def __repr__(self) -> str:
        return f"UserRequest(content={self.content}, files={self.files})"


CodeInterpreterRequest = Union[str, List[Union[str, Dict]]]


class CodeInterpreterIntermediateResult(BaseModel):
    thoughts: List[str] = Field(
        default_factory=list,
        description="エージェントの思考プロセスを表す文字列のリスト(最新の思考および根拠を理解するために必要な情報のみが入っている)",
    )
    context: str = Field(default="", description="llmやagentからの回答本文")
    code: str = Field(
        default="", description="プログラムのソースコード(コード自体への説明や不要な改行などを入れないこと)"
    )
    log: str = Field(default="", description="コードの実行結果やテスト結果などのlog")
    language: str = Field(default="", description="pythonやjavaなどのプログラム言語")
    confidence: float = Field(default=0.95, description="現在の回答の信頼度[0.0～1.0], 1.0が最も信頼できる")
    target_confidence: float = Field(default=0.95, description="処理完了の目標信頼度、これ以上なら完了する")
    # metadata: Dict[str, Any] = Field(default_factory=dict, description="追加のメタデータを格納する辞書")
    iteration_count: int = Field(default=0, description="現在の反復回数")
    max_iterations: int = Field(default=10, description="最大反復回数")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"""CodeInterpreterIntermediateResult(
            thoughts={self.thoughts}
            context={self.context}
            code={self.code}
            log={self.log}
            confidence={self.confidence})"""


class CodeInterpreterResponse(AIMessage):
    """
    Response from the code interpreter agent.
    """

    files: Optional[list[File]] = []
    code: str = ""  # final code
    log: str = ""  # final log
    language: str = ""  # ex: python, java
    start: bool = False
    end: bool = False
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

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"<SessionStatus status={self.status}>"


class CodeInterpreterPlan(BaseModel):
    '''単一PlanのAgentとTaskの説明です。
    PlanとAgentとTaskは常に1:1:1の関係です。
    '''

    agent_name: str = Field(
        description="Agentの名前(=タスクの名前)です。システム内のprimary keyとして使われます。利用可能な文字は[a-Z_]です。task likeな名前にしてください。"
    )
    task_description: str = Field(
        description="タスクの説明です。可能な範囲でpurpose, execution plan, input, outputの詳細を含めます。"
    )
    expected_output: str = Field(
        description="タスクの最終的な出力形式を明確に定義します。例えばjson/csvというフォーマットや、カラム名やサイズ情報です。"
    )

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        ret_str = ""
        ret_str += f"<agent_name={self.agent_name}>\n"
        ret_str += f"<task_description={self.task_description}>\n"
        ret_str += f"<expected_output={self.expected_output}>\n"
        return ret_str


class CodeInterpreterPlanList(BaseModel):
    '''CodeInterpreterPlanの配列をもつ計画全体です。'''

    reliability: int = Field(
        description="計画の信頼度[0-100]です。100が完全な計画を意味します。50未満だと不完全計画でオリジナルの問題を直接llmに渡した方が良い結果になります。"
    )
    agent_task_list: List[CodeInterpreterPlan] = Field(description="CodeInterpreterPlanの配列です。")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        ret_str = ""
        ret_str += f"<reliability={self.reliability}>\n"
        for agent_task in self.agent_task_list:
            ret_str += f"{agent_task}>\n"
        return ret_str
