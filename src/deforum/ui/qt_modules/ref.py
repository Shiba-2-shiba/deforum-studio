# import math
# import sys
# from datetime import datetime
# from uuid import uuid4
#
# from qtpy.QtCore import Qt, Signal, QLine, QPoint, QRectF, QRect, QPropertyAnimation, QEasingCurve
# from qtpy.QtGui import QColor, QFont, QPainter, QPen, QPolygon, QBrush, QPainterPath, QAction, QCursor
# from qtpy.QtWidgets import (QApplication)
# from qtpy.QtWidgets import QHBoxLayout
# from qtpy.QtWidgets import QLineEdit, QPushButton
# from qtpy.QtWidgets import QVBoxLayout, QWidget, QSlider, QDockWidget, QMenu
#
#
# class KeyFrame:
#     def __init__(self, uid, valueType, position, value, color=Qt.GlobalColor.darkYellow):
#         self.uid = uid
#         self.valueType = valueType
#         self.position = position
#         self.value = value
#         self.color = color
#
#
# __textColor__ = QColor(255, 255, 255)  # White
# __backgroudColor__ = QColor(40, 40, 40)  # Dark Grey
# __font__ = QFont('Decorative', 10)
# __gridColor__ = QColor(211, 211, 211)  # Light Grey
#
#
# class TimeLine(QWidget):
#     keyFramesUpdated = Signal()
#     selectionChanged = Signal(object)
#     keyframeValuesChanged = Signal(str)
#
#     def __init__(self, duration, length):
#         super().__init__()
#         self.duration = duration
#         self.length = length
#
#         # Set variables
#         self.backgroundColor = __backgroudColor__
#         self.textColor = __textColor__
#         self.font = __font__
#         self.pos = None
#         self.oldPos = None
#         self.pointerPos = None
#         self.pointerValue = None
#         self.pointerTimePos = 0
#         self.selectedSample = None
#         self.clicking = False  # Check if mouse left button is being pressed
#         self.is_in = False  # check if user is in the widget
#         self.videoSamples = []  # List of videos samples
#         self.middleHover = False
#         self.setMouseTracking(True)  # Mouse events
#         self.setAutoFillBackground(True)  # background
#         self.edgeGrab = False
#         self.scale = None
#         self.middleHoverActive = False
#         self.selectedValueType = "strength"
#         self.keyHover = False
#         self.hoverKey = None
#         self.selectedKey = None
#         self.moveSelectedKey = False
#         self.posy = 50
#         self.yMiddlePoint = 200
#         self.verticalScale = 10
#         self.keyFrameList = []
#         self.initUI()
#
#     def initUI(self):
#
#         self.setGeometry(300, 300, self.length, 200)
#         self.setWindowTitle("TESTE")
#
#         # Set Background
#         # pal = QPalette()
#         # pal.setColor(QPalette.ColorGroup.Base, self.backgroundColor)
#         # self.setPalette(pal)
#
#     def mixed_order(self, a):
#         return (a.valueType, a.position)
#
#     def paintEvent(self, event):
#         self.keyFrameList.sort(key=self.mixed_order)
#         self.yMiddlePoint = self.height() / 2
#         qp = QPainter(self)
#         qp.setRenderHint(QPainter.RenderHint.Antialiasing)
#         # Fill the entire background first
#         qp.fillRect(self.rect(), __backgroudColor__)
#
#         # Draw time
#         qp.setPen(self.textColor)
#         qp.setFont(self.font)
#         scale = self.getScale()
#         w = 0
#         while w <= self.width():
#             qp.drawText(w - 50, 0, 100, 100, Qt.AlignmentFlag.AlignHCenter, self.get_time_string(w * scale))
#             w += 100
#         # Draw numeric values on the vertical edge
#         qp.setPen(self.textColor)
#         max_value = int(self.height() / 2 / self.verticalScale)
#         for val in range(-max_value, max_value + 1, 5):  # Adjust the range and step for desired density
#             y_pos = int(self.yMiddlePoint - val * self.verticalScale)
#             qp.drawText(0, y_pos, 40, 20, Qt.AlignmentFlag.AlignRight, str(val))
#
#
#         # Draw down line
#         qp.setPen(QPen(Qt.GlobalColor.darkCyan, 5, Qt.PenStyle.SolidLine))
#         qp.drawLine(0, 40, self.width(), 40)
#
#         # Draw Middle Line for 0 Value of Keyframes
#         qp.setPen(QPen(Qt.GlobalColor.darkGreen, 2, Qt.PenStyle.SolidLine))
#         qp.drawLine(0, int(self.yMiddlePoint), int(self.width()), int(self.yMiddlePoint))
#
#         # Draw dash lines
#         qp.setPen(QPen(self.textColor))
#         point = 0
#         while point <= self.width():
#             if point % 30 != 0:
#                 qp.drawLine(3 * point, 40, 3 * point, 30)
#             else:
#                 qp.drawLine(3 * point, 40, 3 * point, 20)
#             point += 10
#
#         # Draw keyframes
#         self.oldY = None
#         self.oldX = None
#         if self.selectedValueType is not None:
#             for i in self.keyFrameList:
#                 if i and i.valueType == self.selectedValueType:
#                     kfStartPoint = int(i.position / self.getScale())
#                     kfYPos = int(self.yMiddlePoint - i.value * self.verticalScale)
#                     if self.oldY is not None:
#                         qp.setPen(QPen(Qt.GlobalColor.darkMagenta, 2, Qt.PenStyle.SolidLine))
#                         qp.drawLine(self.oldX, self.oldY, kfStartPoint, kfYPos)
#                     if self.selectedKey == i.uid:
#                         kfbrush = QBrush(Qt.GlobalColor.blue)  # Color for selected keyframe
#                     else:
#                         kfbrush = QBrush(Qt.GlobalColor.darkRed)
#                     scaleMod = 5
#                     kfPoly = QPolygon([
#                         QPoint(kfStartPoint - scaleMod, kfYPos),
#                         QPoint(kfStartPoint, kfYPos - scaleMod),
#                         QPoint(kfStartPoint + scaleMod, kfYPos),
#                         QPoint(kfStartPoint, kfYPos + scaleMod)
#                     ])
#                     qp.setPen(Qt.GlobalColor.darkRed)
#                     qp.setBrush(kfbrush)
#                     qp.drawPolygon(kfPoly)
#
#                     self.oldY = kfYPos
#                     self.oldX = kfStartPoint
#
#         # Draw samples
#         for sample in self.videoSamples:
#             path = QPainterPath()
#             path.addRoundedRect(QRectF(sample.startPos / scale, 50, sample.duration / scale, 200), 10, 10)
#             qp.setClipPath(path)
#
#             # Draw sample
#             path = QPainterPath()
#             qp.setPen(sample.color)
#             qp.setBrush(sample.color)
#             path.addRect(sample.startPos / scale, 50, sample.duration / scale, 50)
#             qp.fillPath(path, sample.color)
#             qp.drawPath(path)
#
#             # Draw preview pictures
#             if sample.picture:
#                 if sample.picture.size().width() < sample.duration / scale:
#                     path = QPainterPath()
#                     path.addRoundedRect(QRectF(sample.startPos / scale, 52.5, sample.picture.size().width(), 45), 10,
#                                         10)
#                     qp.setClipPath(path)
#                     qp.drawPixmap(QRect(int(sample.startPos / scale), 52.5, sample.picture.size().width(), 45),
#                                   sample.picture)
#                 else:
#                     path = QPainterPath()
#                     path.addRoundedRect(QRectF(sample.startPos / scale, 52.5, sample.duration / scale, 45), 10, 10)
#                     qp.setClipPath(path)
#                     pic = sample.picture.copy(0, 0, sample.duration / scale, 45)
#                     qp.drawPixmap(QRect(int(sample.startPos / scale), 52.5, sample.duration / scale, 45), pic)
#
#             # Clear clip path after each sample
#             qp.setClipPath(QPainterPath())
#
#         # Draw pointer
#         if self.pointerPos is not None:
#             self.pointerTimePos = int(self.pointerTimePos)
#             line = QLine(QPoint(int(self.pointerTimePos / self.getScale()), 40),
#                          QPoint(int(self.pointerTimePos / self.getScale()), self.height()))
#             poly = QPolygon([QPoint(int(self.pointerTimePos / self.getScale() - 10), 20),
#                              QPoint(int(self.pointerTimePos / self.getScale() + 10), 20),
#                              QPoint(int(self.pointerTimePos / self.getScale()), 40)])
#             # Draw pointer
#             qp.setPen(Qt.GlobalColor.darkCyan)
#             qp.setBrush(QBrush(Qt.GlobalColor.darkCyan))
#
#             qp.drawPolygon(poly)
#             qp.drawLine(line)
#         #qp.end()
#
#     # Mouse movement
#     def mouseMoveEvent(self, e):
#
#         self.pos = e.position().x()
#         self.posy = e.position().y()
#         self.pointerValue = self.posy  # if mouse is being pressed, update pointer
#
#         self.checkKeyframeHover(self.pos)
#
#         if self.clicking:
#
#             self.oldPos = self.pointerPos
#             self.oldValue = self.pointerValue
#             x = self.pos
#             y = self.posy
#             self.pointerPos = x
#
#             self.pointerTimePos = self.pointerPos * self.getScale()
#
#             if self.keyHover == True:
#                 for item in self.keyFrameList:
#                     if self.selectedKey is item.uid:
#                         item.position = int(self.pointerPos * self.scale)
#                         if item.position <= 0:
#                             item.position = 0
#                         value = (self.pointerValue - self.yMiddlePoint) / self.verticalScale
#                         item.value = -value
#                         self.keyFramesUpdated.emit()
#                         # print(item.value)
#                         # print(self.posy)
#                         # print(self.yMiddlePoint)
#             if self.edgeGrabActive == True:
#                 for sample in self.videoSamples:
#                     sample.duration = sample.duration + ((self.pointerPos - self.oldPos) * self.scale)
#             elif self.middleHoverActive == True:
#                 self.scale = self.getScale()
#                 for sample in self.videoSamples:
#                     change = (x - self.oldPos)
#                     change = (change * self.scale)
#                     ##print(change)
#                     sample.startPos = sample.startPos + change
#                     sample.endPos = sample.endPos + change
#         self.update()
#
#     # Mouse pressed
#     def checkKeyframeHover(self, x):
#         for item in self.keyFrameList:
#             kfStartPoint = int(int(item.position) / self.getScale())
#             kfYPos = int(self.yMiddlePoint - item.value * self.verticalScale)
#
#             if kfStartPoint - 5 < x < kfStartPoint + 5 and kfYPos + 5 > self.posy > kfYPos - 5:
#                 self.keyHover = True
#                 ##print(item.uid)
#                 self.hoverKey = item.uid
#         self.update()
#
#     def checkKeyClicked(self):
#         for item in self.keyFrameList:
#             if self.hoverKey is item.uid:
#                 self.selectedKey = self.hoverKey
#                 self.keyHover = True
#         self.update()
#
#     def mousePressEvent(self, e):
#         self.selectedKey = None
#         self.update()
#         self.scale = self.getScale()
#         if e.button() == Qt.MouseButton.LeftButton:
#
#             x = e.pos().x()
#             self.checkKeyClicked()
#
#             ##print(self.keyClicked)
#             self.pointerPos = x
#             self.pointerTimePos = self.pointerPos * self.getScale()
#
#             self.clicking = True  # Set clicking check to true
#             if self.edgeGrab == True:
#                 self.edgeGrabActive = True
#             else:
#                 self.edgeGrabActive = False
#             if self.middleHover == True:
#                 self.middleHoverActive = True
#             else:
#                 self.middleHoverActive = False
#         elif e.button() == Qt.MouseButton.RightButton:
#             self.popMenu = QMenu()
#             menuPosition = QCursor.pos()
#             x = self.pos
#             self.checkKeyframeHover(x)
#             self.checkKeyClicked()
#             ##print(self.hoverKey)
#             ##print(self.keyHover)
#             ##print(self.selectedKey)
#             self.popMenu.clear()
#             # populate
#             self.populateBtnContext()
#
#             if self.selectedKey is None:
#                 self.popMenu.delete_action.setEnabled(False)
#
#             # show
#             self.popMenu.move(menuPosition)
#             self.popMenu.show()
#             self.pointerPos = e.pos().x()
#             self.popMenu.delete_action.triggered.connect(self.delete_action)
#             self.popMenu.add_action.triggered.connect(self.add_action)
#         self.update()
#
#     def populateBtnContext(self):
#
#         # Do some if here :
#         self.popMenu.add_action = QAction('add keyframe', self)
#         self.popMenu.delete_action = QAction('delete keyframe', self)
#         self.popMenu.addAction(self.popMenu.delete_action)
#         self.popMenu.addAction(self.popMenu.add_action)
#     def emit_current_values(self):
#         tempString = ""
#         for item in self.keyFrameList:
#             if tempString == "":
#                 tempString = f'{item.position}:({item.value})'
#             else:
#                 tempString = f'{tempString}, {item.position}:({item.value})'
#         return tempString
#     # Mouse release
#     def add_action(self):
#         ##print(self.keyClicked)
#         # self.pointerPos
#         self.pointerTimePos = self.pointerPos * self.getScale()
#
#         matchFound = False
#         value = (self.pointerValue - self.yMiddlePoint) / self.verticalScale
#         value = -value
#         valueType = self.selectedValueType
#         position = int(self.pointerTimePos)
#         keyframe = {}
#         uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
#         keyframe[position] = KeyFrame(uid, valueType, position, value)
#         for items in self.keyFrameList:
#             if items.valueType == valueType:
#                 if items.position == position:
#                     items.value = value
#                     matchFound = True
#         if matchFound == False:
#             self.keyFrameList.append(keyframe[position])
#         self.update()
#         self.keyframeValuesChanged.emit(self.emit_current_values())
#
#         # print(self.keyFrameList)
#         # self.updateAnimKeys()
#
#     def delete_action(self):
#         for idx, item in enumerate(self.keyFrameList):
#             # print(idx)
#             # print(item)
#             if self.hoverKey is item.uid:
#                 self.keyFrameList.pop(idx)
#         self.update()
#         self.keyframeValuesChanged.emit(self.emit_current_values())
#
#         # item.remove()
#         # return
#
#     def mouseReleaseEvent(self, e):
#         self.scale = self.getScale()
#         if e.button() == Qt.MouseButton.LeftButton:
#             self.clicking = False  # Set clicking check to false
#             #self.selectedKey = None
#             self.keyHover = False
#             self.hoverKey = None
#         self.update()
#         self.keyframeValuesChanged.emit(self.emit_current_values())
#
#
#     # Enter
#     def enterEvent(self, e):
#         self.is_in = True
#         self.update()
#
#     # Leave
#     def leaveEvent(self, e):
#         self.is_in = False
#         self.update()
#
#     # check selection
#     def checkSelection(self, x):
#         # Check if user clicked in video sample
#         for sample in self.videoSamples:
#             if sample.startPos + 25 < x < sample.endPos - 25:
#                 sample.color = Qt.GlobalColor.darkCyan
#                 self.middleHover = True
#                 if self.selectedSample is not sample:
#                     self.selectedSample = sample
#                     # self.selectionChanged.emit(sample)
#             else:
#                 sample.color = sample.defColor
#                 self.middleHover = False
#         self.update()
#
#     def checkEdges(self, x, y=50):
#
#         for sample in self.videoSamples:
#             if sample.startPos < x < sample.startPos + 24:
#                 sample.color = Qt.GlobalColor.darkMagenta
#                 if self.selectedSample is not sample:
#                     self.selectedSample = sample
#                     # self.selectionChanged.emit(sample)
#             elif sample.endPos - 24 < x < sample.endPos:
#                 sample.color = Qt.GlobalColor.darkGreen
#                 self.edgeGrab = True
#                 if self.selectedSample is not sample:
#                     self.selectedSample = sample
#                     # self.selectionChanged.emit(sample)
#             else:
#                 sample.color = sample.defColor
#                 self.edgeGrab = False
#         self.update()
#
#     # Get time string from seconds
#     def get_time_string(self, seconds):
#         m, s = divmod(seconds, 60)
#         h, m = divmod(m, 60)
#         # return "%02d:%02d:%02d" % (h, m, s)
#         return "%05d" % (seconds)
#
#     # Get scale from length
#     def getScale(self):
#         return float(self.duration) / float(self.width())
#
#     # Get duration
#     def getDuration(self):
#         return self.duration
#
#     # Get selected sample
#     def getSelectedSample(self):
#         return self.selectedSample
#
#     # Set background color
#     def setBackgroundColor(self, color):
#         self.backgroundColor = color
#
#     # Set text color
#     def setTextColor(self, color):
#         self.textColor = color
#
#     # Set Font
#     def setTextFont(self, font):
#         self.font = font
#     def clearKeyframes(self):
#         self.keyFrameList.clear()
#         self.update()
#
#
#
# class TimeLineQDockWidget(QDockWidget):
#     backgroundColor = __backgroudColor__
#     textColor = __textColor__
#     font = __font__
#     def __init__(self, parent=None):
#
#         super(TimeLineQDockWidget, self).__init__(parent)
#
#         self.timeline = TimeLine(1000, 1000)
#
#         self.horizontalZoomSlider = QSlider(Qt.Orientation.Horizontal)
#         self.verticalZoomSlider = QSlider(Qt.Orientation.Vertical)
#         # Initialize the sliders' range and default value
#         self.horizontalZoomSlider.setRange(5, 5000)  # example range
#         self.horizontalZoomSlider.setValue(1000)  # default value
#
#         self.verticalZoomSlider.setRange(1, 50)  # example range
#         self.verticalZoomSlider.setValue(10)  # default value
#
#         # Connect sliders to zoom functions
#         self.horizontalZoomSlider.valueChanged.connect(self.onHorizontalZoomChanged)
#         self.verticalZoomSlider.valueChanged.connect(self.onVerticalZoomChanged)
#
#         h_layout = QHBoxLayout()
#         layout = QVBoxLayout()
#         self.keyframeEdit = QLineEdit()  # Add a QLineEdit to edit keyframes
#         self.updateButton = QPushButton("Eval")
#         layout.addWidget(self.keyframeEdit)
#         layout.addWidget(self.updateButton)
#         layout.addWidget(self.timeline)
#         layout.addWidget(self.horizontalZoomSlider)
#         h_layout.addLayout(layout)
#         h_layout.addWidget(self.verticalZoomSlider)
#
#         container = QWidget()
#         container.setLayout(h_layout)
#         self.setWidget(container)
#         self.updateButton.clicked.connect(self.updateKeyframesFromText)
#         #self.initAnimations()
#     def onHorizontalZoomChanged(self, value):
#         # Adjust the duration of the timeline based on the slider value
#         self.timeline.duration = value
#         self.timeline.update()
#
#     def onVerticalZoomChanged(self, value):
#         # Adjust the vertical scale of the keyframes based on the slider value
#         self.timeline.verticalScale = value
#         self.timeline.update()
#     def onZoomChanged(self, value):
#         # Handle zoom logic
#         pass
#
#     def initAnimations(self):
#         self.hideAnimation = QPropertyAnimation(self, b"maximumHeight")
#         self.hideAnimation.setDuration(500)
#         self.hideAnimation.setStartValue(self.height())
#         self.hideAnimation.setEndValue(0)
#         self.hideAnimation.setEasingCurve(QEasingCurve.Linear)
#
#         self.showAnimation = QPropertyAnimation(self, b"maximumHeight")
#         self.showAnimation.setDuration(500)
#         self.showAnimation.setStartValue(0)
#         self.showAnimation.setEndValue(self.parent().height())
#         self.showAnimation.setEasingCurve(QEasingCurve.Linear)
#
#     def showWithAnimation(self):
#         self.showAnimation.start()
#
#     def hideWithAnimation(self):
#         self.hideAnimation.start()
#
#     def handle_connection(self, widget):
#
#         #Add logic to disconnect any previous QLineEdit widget,
#         #and connect the one passed to this function, any added/deleted/moved
#         #keyframe should result in changing the LineEdit to the string made by
#         #emit_current_values
#
#         # Disconnect previous connections
#         try:
#             self.timeline.keyframeValuesChanged.disconnect()
#         except TypeError:
#             # No previous connections, so ignore
#             pass
#         # Clear the existing keyframes
#         self.timeline.clearKeyframes()
#
#         # Update the timeline's keyframes based on the text from the new widget
#         is_math = self.updateKeyframesFromText(widget.text())
#         if not is_math:
#             # Connect the signal to a new slot method
#             self.timeline.keyframeValuesChanged.connect(widget.setText)
#
#
#     def updateKeyframesFromText(self):
#         text = self.keyframeEdit.text()
#         self.timeline.clearKeyframes()
#         segments = text.split(", ")
#         is_math = True
#         try:
#             for segment in segments:
#                 # Extract position (time)
#                 position_str, value_segment = segment.split(":")
#                 position = int(position_str.strip())
#
#                 # Extract raw value by removing the first and the last characters
#                 value_segment = value_segment.strip(" ")
#
#                 raw_value_str = value_segment[1:-1]
#
#                 if is_equation(raw_value_str):
#                     # If it's an equation, generate values over the timeline's duration
#                     for t in range(0, self.timeline.getDuration() + 1):
#                         value = eval_equation(raw_value_str, t)
#                         uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
#                         valueType = self.timeline.selectedValueType
#                         keyframe = KeyFrame(uid, valueType, t, value)
#                         self.timeline.keyFrameList.append(keyframe)
#                     is_math = True
#                 else:
#                     # Otherwise, it's a simple value
#                     value = float(raw_value_str)
#                     uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
#                     valueType = self.timeline.selectedValueType
#                     keyframe = KeyFrame(uid, valueType, position, value)
#                     self.timeline.keyFrameList.append(keyframe)
#                     is_math = False
#             self.timeline.update()
#
#         except:
#             pass
#         finally:
#             return is_math
#
#
# def is_equation(s):
#     # A very basic check for now.
#     # A string that starts with '-' and contains no other operators is not an equation
#     if s.startswith('-') and not any(op in s[1:] for op in ['+', '-', '*', '/', 'sin', 'cos', 'tan']):
#         return False
#
#     return '+' in s or '-' in s or '*' in s or '/' in s or 'sin' in s or 'cos' in s or 'tan' in s
#
#
# def eval_equation(eq, t):
#     # Safely evaluate the equation
#     return eval(eq, {"__builtins__": None, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 't': t})
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     timeline = TimeLineQDockWidget()
#     timeline.show()
#     sys.exit(app.exec())
import math
import sys
from datetime import datetime
from uuid import uuid4

from qtpy.QtCore import Qt, Signal, QLine, QPoint, QRectF, QRect, QPropertyAnimation, QEasingCurve
from qtpy.QtGui import QColor, QFont, QPainter, QPen, QPolygon, QBrush, QPainterPath, QAction, QCursor
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget, QSlider, QDockWidget, QMenu


class KeyFrame:
    def __init__(self, uid, valueType, position, value, color=Qt.GlobalColor.darkYellow):
        self.uid = uid
        self.valueType = valueType
        self.position = position
        self.value = value
        self.color = color


__textColor__ = QColor(255, 255, 255)  # White
__backgroudColor__ = QColor(40, 40, 40)  # Dark Grey
__font__ = QFont('Decorative', 10)
__gridColor__ = QColor(211, 211, 211)  # Light Grey


class TimeLine(QWidget):
    keyFramesUpdated = Signal()
    selectionChanged = Signal(object)
    keyframeValuesChanged = Signal(str)

    def __init__(self, duration, length):
        super().__init__()
        self.duration = duration
        self.length = length
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.pos = None
        self.oldPos = None
        self.pointerPos = None
        self.pointerValue = None
        self.pointerTimePos = 0
        self.selectedSample = None
        self.clicking = False
        self.is_in = False
        self.videoSamples = []
        self.middleHover = False
        self.setAutoFillBackground(True)
        self.edgeGrab = False
        self.scale = None
        self.middleHoverActive = False
        self.selectedValueType = "strength"
        self.keyHover = False
        self.hoverKey = None
        self.selectedKey = None
        self.moveSelectedKey = False
        self.posy = 50
        self.yMiddlePoint = 200
        self.verticalScale = 10
        self.keyFrameList = []
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, self.length, 200)
        self.setWindowTitle("Timeline Visualizer")

    def mixed_order(self, a):
        return (a.valueType, a.position)

    def paintEvent(self, event):
        self.keyFrameList.sort(key=self.mixed_order)
        self.yMiddlePoint = self.height() / 2
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), __backgroudColor__)

        qp.setPen(self.textColor)
        qp.setFont(self.font)
        scale = self.getScale()
        self.drawGrid(qp)

        w = 0
        while w <= self.width():
            qp.drawText(w - 50, 0, 100, 100, Qt.AlignmentFlag.AlignHCenter, self.get_time_string(w * scale))
            w += 100
        qp.setPen(self.textColor)
        max_value = int(self.height() / 2 / self.verticalScale)
        for val in range(-max_value, max_value + 1, 1):
            y_pos = int(self.yMiddlePoint - val * self.verticalScale)
            qp.drawText(0, y_pos, 40, 20, Qt.AlignmentFlag.AlignRight, f"{val:.2f}" if val % 1 != 0 else str(val))
        qp.setPen(QPen(Qt.GlobalColor.darkCyan, 5, Qt.PenStyle.SolidLine))
        qp.drawLine(0, 40, self.width(), 40)
        qp.setPen(QPen(Qt.GlobalColor.darkGreen, 2, Qt.PenStyle.SolidLine))
        qp.drawLine(0, int(self.yMiddlePoint), int(self.width()), int(self.yMiddlePoint))
        qp.setPen(QPen(self.textColor))
        point = 0
        while point <= self.width():
            qp.drawLine(3 * point, 40, 3 * point, 30 if point % 30 != 0 else 20)
            point += 10
        self.oldY = None
        self.oldX = None
        if self.selectedValueType is not None:
            for i in self.keyFrameList:
                if i and i.valueType == self.selectedValueType:
                    kfStartPoint = int(i.position / self.getScale())
                    kfYPos = int(self.yMiddlePoint - i.value * self.verticalScale)
                    if self.oldY is not None:
                        qp.setPen(QPen(Qt.GlobalColor.darkMagenta, 2, Qt.PenStyle.SolidLine))
                        qp.drawLine(self.oldX, self.oldY, kfStartPoint, kfYPos)
                    kfbrush = QBrush(Qt.GlobalColor.blue if self.selectedKey == i.uid else Qt.GlobalColor.darkRed)
                    scaleMod = 5
                    kfPoly = QPolygon([
                        QPoint(kfStartPoint - scaleMod, kfYPos),
                        QPoint(kfStartPoint, kfYPos - scaleMod),
                        QPoint(kfStartPoint + scaleMod, kfYPos),
                        QPoint(kfStartPoint, kfYPos + scaleMod)
                    ])
                    qp.setPen(Qt.GlobalColor.darkRed)
                    qp.setBrush(kfbrush)
                    qp.drawPolygon(kfPoly)
                    self.oldY = kfYPos
                    self.oldX = kfStartPoint
        for sample in self.videoSamples:
            path = QPainterPath()
            path.addRoundedRect(QRectF(sample.startPos / scale, 50, sample.duration / scale, 200), 10, 10)
            qp.setClipPath(path)
            path = QPainterPath()
            qp.setPen(sample.color)
            qp.setBrush(sample.color)
            path.addRect(sample.startPos / scale, 50, sample.duration / scale, 50)
            qp.fillPath(path, sample.color)
            qp.drawPath(path)
            if sample.picture:
                picRect = QRect(int(sample.startPos / scale), 52.5, sample.duration / scale if sample.picture.size().width() < sample.duration / scale else sample.picture.size().width(), 45)
                qp.drawPixmap(picRect, sample.picture)
            qp.setClipPath(QPainterPath())
        if self.pointerPos is not None:
            line = QLine(QPoint(int(self.pointerTimePos / self.getScale()), 40), QPoint(int(self.pointerTimePos / self.getScale()), self.height()))
            poly = QPolygon([QPoint(int(self.pointerTimePos / self.getScale() - 10), 20), QPoint(int(self.pointerTimePos / self.getScale() + 10), 20), QPoint(int(self.pointerTimePos / self.getScale()), 40)])
            qp.setPen(Qt.GlobalColor.darkCyan)
            qp.setBrush(QBrush(Qt.GlobalColor.darkCyan))
            qp.drawPolygon(poly)
            qp.drawLine(line)

    def drawGrid(self, qp):
        rect = self.rect()
        min_spacing = 1  # Minimum spacing to prevent zero in range step
        spacing = max(self.scale, min_spacing)  # Ensure spacing is never less than min_spacing

        # Draw vertical lines
        for x in range(0, int(rect.width()), int(spacing)):
            if x // spacing % 10 == 0:  # Every 10th line
                qp.setPen(QPen(QColor(80, 80, 80), 2))  # Make line thicker and slightly darker
            else:
                qp.setPen(QPen(QColor(60, 60, 60), 1))  # Regular light gray lines
            qp.drawLine(x, 0, x, int(rect.height()))

        # Draw horizontal lines
        midY = rect.height() / 2
        vertical_spacing = max(20 * (spacing / 100), 1)  # Scale vertical spacing similarly, minimum 1 pixel
        num_lines = int(rect.height() / 2 / vertical_spacing)  # Calculate number of lines based on spacing

        for i in range(-num_lines, num_lines + 1):
            y = midY + i * vertical_spacing
            if i % 10 == 0:
                qp.setPen(QPen(QColor(80, 80, 80), 2))  # Thicker lines at major intervals
            else:
                qp.setPen(QPen(QColor(60, 60, 60), 1))
            qp.drawLine(0, int(y), rect.width(), int(y))

    def mouseMoveEvent(self, e):
        self.pos = e.position().x()
        self.posy = e.position().y()
        self.pointerValue = self.posy
        self.checkKeyframeHover(self.pos)
        if self.clicking:
            self.oldPos = self.pointerPos
            self.oldValue = self.pointerValue
            x = self.pos
            y = self.posy
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()
            if self.keyHover == True:
                for item in self.keyFrameList:
                    if self.selectedKey is item.uid:
                        item.position = int(self.pointerPos * self.scale)
                        if item.position <= 0:
                            item.position = 0
                        value = (self.pointerValue - self.yMiddlePoint) / self.verticalScale
                        item.value = -value
                        self.keyFramesUpdated.emit()
            if self.edgeGrabActive == True:
                for sample in self.videoSamples:
                    sample.duration = sample.duration + ((self.pointerPos - self.oldPos) * self.scale)
            elif self.middleHoverActive == True:
                self.scale = self.getScale()
                for sample in self.videoSamples:
                    change = (x - self.oldPos)
                    change = (change * self.scale)
                    sample.startPos = sample.startPos + change
                    sample.endPos = sample.endPos + change
        self.update()

    def checkKeyframeHover(self, x):
        for item in self.keyFrameList:
            kfStartPoint = int(int(item.position) / self.getScale())
            kfYPos = int(self.yMiddlePoint - item.value * self.verticalScale)
            if kfStartPoint - 5 < x < kfStartPoint + 5 and kfYPos + 5 > self.posy > kfYPos - 5:
                self.keyHover = True
                self.hoverKey = item.uid
        self.update()

    def checkKeyClicked(self):
        for item in self.keyFrameList:
            if self.hoverKey is item.uid:
                self.selectedKey = self.hoverKey
                self.keyHover = True
        self.update()

    def mousePressEvent(self, e):
        self.selectedKey = None
        self.update()
        self.scale = self.getScale()
        if e.button() == Qt.MouseButton.LeftButton:
            x = e.pos().x()
            self.checkKeyClicked()
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()
            self.clicking = True
            if self.edgeGrab == True:
                self.edgeGrabActive = True
            else:
                self.edgeGrabActive = False
            if self.middleHover == True:
                self.middleHoverActive = True
            else:
                self.middleHoverActive = False
        elif e.button() == Qt.MouseButton.RightButton:
            self.popMenu = QMenu()
            menuPosition = QCursor.pos()
            x = self.pos
            self.checkKeyframeHover(x)
            self.checkKeyClicked()
            self.popMenu.clear()
            self.populateBtnContext()
            if self.selectedKey is None:
                self.popMenu.delete_action.setEnabled(False)
            self.popMenu.move(menuPosition)
            self.popMenu.show()
            self.pointerPos = e.pos().x()
            self.popMenu.delete_action.triggered.connect(self.delete_action)
            self.popMenu.add_action.triggered.connect(self.add_action)
        self.update()

    def populateBtnContext(self):
        self.popMenu.add_action = QAction('Add keyframe', self)
        self.popMenu.delete_action = QAction('Delete keyframe', self)
        self.popMenu.addAction(self.popMenu.delete_action)
        self.popMenu.addAction(self.popMenu.add_action)

    def emit_current_values(self):
        tempString = ""
        for item in self.keyFrameList:
            if tempString == "":
                tempString = f'{item.position}:({item.value})'
            else:
                tempString = f'{tempString}, {item.position}:({item.value})'
        return tempString

    def add_action(self):
        self.pointerTimePos = self.pointerPos * self.getScale()
        matchFound = False
        value = (self.pointerValue - self.yMiddlePoint) / self.verticalScale
        value = -value
        valueType = self.selectedValueType
        position = int(self.pointerTimePos)
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        keyframe = KeyFrame(uid, valueType, position, value)
        for items in self.keyFrameList:
            if items.valueType == valueType and items.position == position:
                items.value = value
                matchFound = True
        if not matchFound:
            self.keyFrameList.append(keyframe)
        self.update()
        self.keyframeValuesChanged.emit(self.emit_current_values())

    def delete_action(self):
        for idx, item in enumerate(self.keyFrameList):
            if self.hoverKey is item.uid:
                self.keyFrameList.pop(idx)
        self.update()
        self.keyframeValuesChanged.emit(self.emit_current_values())

    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicking = False
            self.keyHover = False
            self.hoverKey = None
        self.update()
        self.keyframeValuesChanged.emit(self.emit_current_values())

    def enterEvent(self, e):
        self.is_in = True
        self.update()

    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    def checkSelection(self, x):
        for sample in self.videoSamples:
            if sample.startPos + 25 < x < sample.endPos - 25:
                sample.color = Qt.GlobalColor.darkCyan
                self.middleHover = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
            else:
                sample.color = sample.defColor
                self.middleHover = False
        self.update()

    def checkEdges(self, x, y=50):
        for sample in self.videoSamples:
            if sample.startPos < x < sample.startPos + 24:
                sample.color = Qt.GlobalColor.darkMagenta
                if self.selectedSample is not sample:
                    self.selectedSample = sample
            elif sample.endPos - 24 < x < sample.endPos:
                sample.color = Qt.GlobalColor.darkGreen
                self.edgeGrab = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
            else:
                sample.color = sample.defColor
                self.edgeGrab = False
        self.update()

    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%05d" % (seconds)

    def getScale(self):
        self.scale = float(self.duration) / float(self.width())
        return self.scale

    def getDuration(self):
        return self.duration

    def getSelectedSample(self):
        return self.selectedSample

    def setBackgroundColor(self, color):
        self.backgroundColor = color

    def setTextColor(self, color):
        self.textColor = color

    def setTextFont(self, font):
        self.font = font

    def clearKeyframes(self):
        self.keyFrameList.clear()
        self.update()


class TimeLineQDockWidget(QDockWidget):
    backgroundColor = __backgroudColor__
    textColor = __textColor__
    font = __font__

    def __init__(self, parent=None):
        super(TimeLineQDockWidget, self).__init__(parent)
        self.timeline = TimeLine(1000, 1000)
        self.horizontalZoomSlider = QSlider(Qt.Orientation.Horizontal)
        self.verticalZoomSlider = QSlider(Qt.Orientation.Vertical)
        self.horizontalZoomSlider.setRange(5, 5000)
        self.horizontalZoomSlider.setValue(1000)
        self.verticalZoomSlider.setRange(1, 1000)
        self.verticalZoomSlider.setValue(10)
        self.horizontalZoomSlider.valueChanged.connect(self.onHorizontalZoomChanged)
        self.verticalZoomSlider.valueChanged.connect(self.onVerticalZoomChanged)
        h_layout = QHBoxLayout()
        layout = QVBoxLayout()
        self.keyframeEdit = QLineEdit()
        self.updateButton = QPushButton("Eval")
        layout.addWidget(self.keyframeEdit)
        layout.addWidget(self.updateButton)
        layout.addWidget(self.timeline)
        layout.addWidget(self.horizontalZoomSlider)
        h_layout.addLayout(layout)
        h_layout.addWidget(self.verticalZoomSlider)
        container = QWidget()
        container.setLayout(h_layout)
        self.setWidget(container)
        self.updateButton.clicked.connect(self.updateKeyframesFromText)

    def onHorizontalZoomChanged(self, value):
        self.timeline.duration = value
        self.timeline.update()

    def onVerticalZoomChanged(self, value):
        self.timeline.verticalScale = value
        self.timeline.update()

    def initAnimations(self):
        self.hideAnimation = QPropertyAnimation(self, b"maximumHeight")
        self.hideAnimation.setDuration(500)
        self.hideAnimation.setStartValue(self.height())
        self.hideAnimation.setEndValue(0)
        self.hideAnimation.setEasingCurve(QEasingCurve.Linear)
        self.showAnimation = QPropertyAnimation(self, b"maximumHeight")
        self.showAnimation.setDuration(500)
        self.showAnimation.setStartValue(0)
        self.showAnimation.setEndValue(self.parent().height())
        self.showAnimation.setEasingCurve(QEasingCurve.Linear)

    def showWithAnimation(self):
        self.showAnimation.start()

    def hideWithAnimation(self):
        self.hideAnimation.start()

    def handle_connection(self, widget):
        try:
            self.timeline.keyframeValuesChanged.disconnect()
        except TypeError:
            pass
        self.timeline.clearKeyframes()
        is_math = self.updateKeyframesFromText(widget.text())
        if not is_math:
            self.timeline.keyframeValuesChanged.connect(widget.setText)

    def updateKeyframesFromText(self):
        text = self.keyframeEdit.text()
        self.timeline.clearKeyframes()
        segments = text.split(", ")
        is_math = True
        try:
            for segment in segments:
                position_str, value_segment = segment.split(":")
                position = int(position_str.strip())
                value_segment = value_segment.strip(" ")
                raw_value_str = value_segment[1:-1]
                if is_equation(raw_value_str):
                    for t in range(0, self.timeline.getDuration() + 1):
                        value = eval_equation(raw_value_str, t)
                        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
                        valueType = self.timeline.selectedValueType
                        keyframe = KeyFrame(uid, valueType, t, value)
                        self.timeline.keyFrameList.append(keyframe)
                    is_math = True
                else:
                    value = float(raw_value_str)
                    uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
                    valueType = self.timeline.selectedValueType
                    keyframe = KeyFrame(uid, valueType, position, value)
                    self.timeline.keyFrameList.append(keyframe)
                    is_math = False
            self.timeline.update()
        except:
            pass
        finally:
            return is_math

def is_equation(s):
    if s.startswith('-') and not any(op in s[1:] for op in ['+', '-', '*', '/', 'sin', 'cos', 'tan']):
        return False
    return '+' in s or '-' in s or '*' in s or '/' in s or 'sin' in s or 'cos' in s or 'tan' in s

def eval_equation(eq, t):
    return eval(eq, {"__builtins__": None, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 't': t})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    timeline = TimeLineQDockWidget()
    timeline.show()
    sys.exit(app.exec())
