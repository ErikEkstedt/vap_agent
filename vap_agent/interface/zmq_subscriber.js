const zmq = require('zeromq');

const port = 5557;
const topic = 'vap';

async function run() {
  const sock = new zmq.Subscriber();

  sock.connect('tcp://localhost:' + port.toString());
  sock.subscribe(topic);
  console.log('Subscriber connected to port ' + port);
  for await (const [topic, msg] of sock) {
    let j = JSON.parse(msg);
    console.log(j.p);
  }
}

run();
