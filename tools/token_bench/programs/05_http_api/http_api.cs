using System.IO;
using System.Net;

class Program {
    static void Main() {
        var listener = new HttpListener();
        listener.Prefixes.Add("http://localhost:8080/");
        listener.Start();

        while (true) {
            var ctx = listener.GetContext();
            var req = ctx.Request;
            var res = ctx.Response;

            string body;
            int status = 200;

            if (req.HttpMethod == "GET" && req.Url.AbsolutePath == "/") {
                body = "Welcome to API";
            } else if (req.HttpMethod == "GET" && req.Url.AbsolutePath == "/health") {
                body = "ok";
            } else if (req.HttpMethod == "POST" && req.Url.AbsolutePath == "/echo") {
                using var reader = new StreamReader(req.InputStream);
                body = reader.ReadToEnd();
            } else {
                body = "not found";
                status = 404;
            }

            res.StatusCode = status;
            using var writer = new StreamWriter(res.OutputStream);
            writer.Write(body);
        }
    }
}
