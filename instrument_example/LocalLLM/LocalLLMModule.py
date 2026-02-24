import asyncio
import json
import threading
import time, datetime
from .client import ChatClient
from .text2command_parser import text2command_ver3
import re
import queue
import io
import base64
import inspect
from functools import wraps
from PIL import Image
from enum import Enum
import numpy as np
import SPMUtil as spmu
from SPMUtil import ScanDataHeader, StageConfigure, PythonScanParam

import LabviewRemoteHelper
from SystemLoader import SystemLoader
from Util.EventHandler import FinishEventType
from Framework.LabviewRemoteManager import LabviewRemoteManager
from Framework.ScanEventManager import ScanEventManager, ScanEventState
from Framework.ScanFileManager import ScanFileManager
from Framework.ScanAreaManager import ScanAreaManager
from BaseClass.BaseModule import BaseModule
import matplotlib.pyplot as plt
import datetime

from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.JAPANESE, Language.CHINESE]
lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()

__last_language_code__ = "en"
def detect_language(text):
    global __last_language_code__
    try:
        lang = lang_detector.detect_language_of(text)
        if lang is None:
            __last_language_code__ = "xx"
        elif lang == Language.CHINESE:
            __last_language_code__ = "zh"
        elif lang == Language.ENGLISH:
            __last_language_code__ = "en"
        elif lang == Language.JAPANESE:
            __last_language_code__ = "ja"
    finally:
        return __last_language_code__



def image_to_base64(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(img, str):
        img = Image.open(img)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return "data:image/png;base64," + img_base64


def check_unlock_label(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        if func_name == self.can_process_cmd_unlock_label:
            self.can_process_cmd = True
        return func(self, *args, **kwargs)
    return wrapper



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class BotCommandType(Enum):
    chat = "chat"
    get = "get"
    set = "set"
    file = "file"
    help = "help"
    misc = "misc"



class AgentState(Enum):
    idle = "idle"
    processing = "processing"



class LocalLLMModule(BaseModule):
    client: ChatClient = None
    cmd_queue = queue.Queue()
    can_process_cmd = True
    def __init__(self, url="http://133.1.195.184:7860", session_id="spm-agent", block=False):
        super().__init__()
        self.client = ChatClient(base_url=url)
        self.client.create_session(session_id)
        LocalLLMModule.client = self.client
        ScanEventManager.instance.ScanUpdateEvent += self._OnScanEventUpdate
        ScanFileManager.instance.ScanFileAddEvent += self._OnSaveFileAdd
        ScanAreaManager.instance.SwitchScanAreaEvent += self._OnScanAreaSwitch
        SystemLoader.MODULE_DICT["DriftCorrectionModule"].OnDriftCorrectionEvent += self._OnDriftCorrectionFinish
        SystemLoader.MODULE_DICT["TipFixModule"].OnFinishedEvent += self._OnTipFixFinish

        self.agentState = AgentState.idle
        self.broadcast_scan = False
        # self.can_process_cmd = True
        self.can_process_cmd_unlock_label = "_OnSaveFileAdd"


        if block:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.main())
        else:
            loop = asyncio.new_event_loop()
            def start_loop():
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.main())

            t = threading.Thread(target=start_loop, daemon=True)
            t.daemon = True
            t.start()


            # thread = threading.Thread(self.main())
            # thread.daemon = True
            # thread.start()



            # loop.create_task(self.main())

    def TestDriftModule(self, data):
        SystemLoader.MODULE_DICT["DriftCorrectionModule"].RunDriftCorrection(data, init=True)

    def TestScanAreaSwitch(self):
        ScanAreaManager.instance.SwitchScanArea(SystemLoader.MODULE_DICT["ArrayBuilderManager"].PythonScanParam,
                                                wait_until_finish=False)

    def TestTipFixModule(self, data):
        SystemLoader.MODULE_DICT["TipFixModule"].Run(data, protect_current_scan_area=False)


    """
    Region: Callback function
    """

    @check_unlock_label
    def _OnDriftCorrectionFinish(self, *args):
        func_name = inspect.currentframe().f_code.co_name
        if func_name == self.can_process_cmd_unlock_label:
            self.can_process_cmd = True
        self.print_info("drift correction finish")

    @check_unlock_label
    def _OnScanAreaSwitch(self, *args):
        func_name = inspect.currentframe().f_code.co_name
        if func_name == self.can_process_cmd_unlock_label:
            self.can_process_cmd = True
        self.print_info("scan area switch finish")


    @check_unlock_label
    def _OnTipFixFinish(self, sender, run_event: FinishEventType):
        func_name = inspect.currentframe().f_code.co_name
        if func_name == self.can_process_cmd_unlock_label:
            self.can_process_cmd = True

        if run_event == FinishEventType.Finish:
            self.print_info("tip fix finish")

        elif run_event == FinishEventType.Failed:
            self.print_info("tip fix finish but failed...Module stopped")

    @check_unlock_label
    def _OnSaveFileAdd(self, sender, data):
        func_name = inspect.currentframe().f_code.co_name
        if func_name == self.can_process_cmd_unlock_label:
            self.can_process_cmd = True


        if self._enabled:
            length = len(ScanFileManager.instance.scanHistoryFileDict.values())
            dataSerializer = list(ScanFileManager.instance.scanHistoryFileDict.values())[length - 1]
            if "FWFW_ZMap" in dataSerializer.data_dict.keys():
                self._save_fig(dataSerializer.data_dict["FWFW_ZMap"], "./slack_temp.png")
                self.send_file(file_path="./slack_temp.png", title=dataSerializer.path)




    @staticmethod
    async def send_message_async(msg):
        LocalLLMModule.client.add_conversation(msg)


    @staticmethod
    def send_message(msg):
        LocalLLMModule.client.add_conversation(msg)

    async def send_file_async(self, title, file_path):
        pass

    def send_file(self, title, file_path):
        LocalLLMModule.client.add_conversation(title)
        LocalLLMModule.client.add_conversation(f'<img src="{image_to_base64(file_path)}">')

    async def main(self):
        await asyncio.sleep(1)

        self.send_message(f"[{str(datetime.datetime.now())}] SPM instrument online.")

        task1 = asyncio.create_task(
            self.client.listen_for_messages(LocalLLMModule.task_on_message_received)
        )
        task2 = asyncio.create_task(self.process_cmd_task_queue())

        try:
            await asyncio.gather(task1, task2)
        except asyncio.CancelledError:
            print("Tasks cancelled.")
        finally:
            print("Shutdown cleanly.")


        # await asyncio.sleep(1)
        # self.send_message(f"[{str(datetime.datetime.now())}] SPM instrument online.")
        # loop = asyncio.new_event_loop()
        # loop.create_task(self.client.listen_for_messages(LocalLLMModule.task_on_message_received))
        # loop.create_task(self.process_cmd_task_queue())



    @staticmethod
    def task_on_message_received(msg: str):
        msg_dict = json.loads(msg)

        received_msg = re.sub(r"</?cmd>", "", msg_dict["response"])
        received_msg_type = msg_dict["type"]
        if received_msg_type == "B":
            #     text to command
            mes, cmd, callback_list = text2command_ver3(LocalLLMModule.client, language_code=detect_language(received_msg))

            # print(mes, cmd, callback_list)
            if mes == "None":
                LocalLLMModule.send_message(mes + ":" + cmd)
                LocalLLMModule.send_message("cmd is not executed.")
            elif mes == "OK":
                for i in range(len(cmd)):
                    remote_type, arg = cmd[i]
                    callback = callback_list[i]

                    if remote_type == "ScanEnabled" and arg == "false":
                        LocalLLMModule.can_process_cmd = True

                    # print((remote_type, arg, callback))
                    LocalLLMModule.cmd_queue.put((remote_type, arg, callback))


    async def process_cmd_task_queue(self):
        previous_can_process_cmd = self.can_process_cmd
        while True:
            await asyncio.sleep(0.5)
            if LocalLLMModule.cmd_queue.qsize() == 0 or not self.can_process_cmd:
                if not previous_can_process_cmd and self.can_process_cmd:
                    self.send_message(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")+"cmd processing finish～(´ε｀；)")

                self.agentState = AgentState.idle
                previous_can_process_cmd = self.can_process_cmd
                continue

            remote_type, arg, callback = LocalLLMModule.cmd_queue.get()
            if not ScanFileManager.instance.scan_file_param.Auto_Save:
                self.send_message(f"Auto_Save is required to be True. {remote_type, arg} is disposed.")
                previous_can_process_cmd = self.can_process_cmd
                continue


            # send command to spm, block not is allowed
            if remote_type == "DriftCompensation":
                self.TestDriftModule(None)
            elif remote_type == "TipFix":
                self.TestTipFixModule(None)
            elif remote_type == "SwitchScanarea":
                self.TestScanAreaSwitch()
            else:
                LabviewRemoteManager.instance.SendLabviewRemoteData_Command(remote_type, arg)
            LocalLLMModule.send_message(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")+remote_type + ":" + arg)

            logger.info(callback)
            if callback != "nan":

                self.can_process_cmd = False
                self.can_process_cmd_unlock_label = callback

                self.agentState = AgentState.processing
            logger.info(f"@@@{self.broadcast_scan}", )
            previous_can_process_cmd = self.can_process_cmd


    def _OnScanEventUpdate(self, sender, data):
        if self._enabled:
            if self.broadcast_scan:
                if data == ScanEventState.Start:
                    self.send_message(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")+"scan start!～(*ﾟ▽ﾟ)ﾉ")
                if data == ScanEventState.Finish:
                    self.send_message(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")+"scan finished!～(✿˘艸˘✿)")
                if data == ScanEventState.Stopped:
                    self.send_message(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")+"scan is stopped..(ó﹏ò。)")



    @staticmethod
    def _save_fig(map, figName):
        plt.clf()
        map = spmu.flatten_map(np.asarray(map).copy(), spmu.FlattenMode.Average)
        map = spmu.filter_2d.gaussian_filter(map, 1)
        map = spmu.formula.topo_map_correction(map)
        plt.imshow(map, cmap="afmhot")
        plt.axis('off')
        plt.savefig(figName, transparent=True)









if __name__ == '__main__':
    slack = LocalLLMModule(block=True)






