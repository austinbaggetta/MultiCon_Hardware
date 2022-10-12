#include "Adafruit_MPR121.h"
#include "src/GenericSerial/GenericSerial.h"

#ifndef _BV
#define _BV(bit) (1 << (bit))
#endif

#define BAUDRATE 115200

Adafruit_MPR121 cap = Adafruit_MPR121();
GenericSerial gs = GenericSerial();
uint16_t lasttouched = 0;
uint16_t currtouched = 0;

void setup()
{
  while (!cap.begin(0x5A))
  {
    ;
  }
  gs.begin(BAUDRATE);
  cap.setThreshholds(5, 3);
}

void loop()
{
  gs.process();
  currtouched = cap.touched();
  for (uint8_t i = 0; i < 12; i++)
  {
    // it if *is* touched and *wasnt* touched before, alert!
    if ((currtouched & _BV(i)) && !(lasttouched & _BV(i)))
    {
      byte buf[2] = {i, 1};
      gs.send(buf);
    }
    // if it *was* touched and now *isnt*, alert!
    if (!(currtouched & _BV(i)) && (lasttouched & _BV(i)))
    {
      byte buf[2] = {i, 0};
      gs.send(buf);
    }
  }

  // reset our state
  lasttouched = currtouched;
}
