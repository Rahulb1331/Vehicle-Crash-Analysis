var express = require('express');
var app = express();
const http = require("http");
const host = 'localhost';
var PORT = 3000;
const server = http.createServer(requestListener);


function postmessage(response, code, data) {

    var json= JSON.stringify({
        group_name: "Name",   // Specify the name of the group
        message: "Hi "  // Specify the content of your message
    });
}

var options = {
        hostname: "api.whatsmate.net",
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(json)
        }
    };

var request = new http.ClientRequest(options);

request.end(json);

request.on('response', function (response) {
    console.log('Status code: ' + response.statusCode);
    response.setEncoding('utf8');
    response.on('data', function (chunk) {
        console.log(chunk);
    });
});


server.listen(port, host, () => {
    if (err) console.log(err);
    console.log('Server is running on http://${host}:${port}');
});

