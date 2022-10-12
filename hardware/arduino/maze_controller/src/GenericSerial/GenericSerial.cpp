/*
  genericSerial.cpp - Library for simple digital control over serial.
  Created by Phil Dong, July 11, 2021.
*/
#include "Arduino.h"
#include "GenericSerial.h"

GenericSerial::GenericSerial() {}

void GenericSerial::begin(long baud)
{
    // initialize serial
    Serial.begin(baud);
    // perform handshake
    while (true)
    {
        if (Serial.available())
        {
            int data;
            data = Serial.read();
            if (data == CMD_FLAG)
            {
                Serial.write(CMD_FLAG);
                break;
            }
        }
    }
}

void GenericSerial::process()
{
    if (Serial.available() > 2)
    {
        size_t sigbyte = Serial.readBytesUntil(CMD_FLAG, this->_buffer, 3);
        if (sigbyte == 3)
        {
            switch (this->_buffer[1])
            {
            case CMD_MODE_OUT:
                pinMode(this->_buffer[0], OUTPUT);
                break;
            case CMD_WRITE_LOW:
                digitalWrite(this->_buffer[0], LOW);
                break;
            case CMD_WRITE_HIGH:
                digitalWrite(this->_buffer[0], HIGH);
                break;
            default:
                return;
            }
        }
    }
}

void GenericSerial::send(byte buf[])
{
    size_t len = sizeof(buf);
    Serial.write(buf, len);
    Serial.write(CMD_FLAG);
}