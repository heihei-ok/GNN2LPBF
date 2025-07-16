#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging


def setlogger(path):
    logger = logging.getLogger()  # 获取根日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")  # 定义日志消息的格式和时间格式

    fileHandler = logging.FileHandler(path)  # 创建一个文件处理器，指定日志文件的路径
    fileHandler.setFormatter(logFormatter)  # 将格式化器应用于文件处理器
    logger.addHandler(fileHandler)  # 将文件处理器添加到日志记录器

    consoleHandler = logging.StreamHandler()  # 创建一个控制台处理器
    consoleHandler.setFormatter(logFormatter)  # 将格式化器应用于控制台处理器
    logger.addHandler(consoleHandler)  # 将控制台处理器添加到日志记录器
    logger.addHandler(consoleHandler)
