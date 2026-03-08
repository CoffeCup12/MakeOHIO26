#include <WiFi.h>
#include <HTTPClient.h>

void sendData(float delta_temp[]) {
  if(WiFi.status() == WL_CONNECTED){
    HTTPClient http;
    http.begin("http://<server_ip>:5000/analyze");
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"delta_temp\": [";
    for(int i=0;i<3;i++){
      payload += String(delta_temp[i]);
      if(i<2) payload += ",";
    }
    payload += "]}";

    int httpResponseCode = http.POST(payload);
    http.end();
  }
}