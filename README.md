# Inbedded
This project attempts to achieve screen to camera communication through (somewhat) imperceptible QR codes.

## Contents
`final_demo.ipynb` contains a step by step run of the encoding and decoding pipeline.

`qr_frame.py` contains the frame structure for one packet of a message transmitted through a single QR code.

`qr_solomon.py` contains the Solomon-Reed encoding and decoding of many frames so that they combine in a single final message.

`protocol.py` contains message packets responsible for transmitting subtitle information. This is not used in the final project since subtitle messages are too small and fit into one QR packet.

`subtitle_manager.py` contains a class responsible for retriving subtititles that should be active at a certain time.

`video_processor.py` contains the class encoding a givens sequence of QR codes into a video by modulating the lightness.