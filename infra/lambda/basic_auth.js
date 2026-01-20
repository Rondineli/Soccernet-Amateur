'use strict';

exports.handler = async (event) => {
    const request = event.Records[0].cf.request;
    const headers = request.headers;

    const authUser = "admin";
    const authPass = "4m4t3urS0cc3RN3t";
    const expectedAuth =
        "Basic " + Buffer.from(authUser + ":" + authPass).toString("base64");

    if (
        !headers.authorization ||
        headers.authorization[0].value !== expectedAuth
    ) {
        return {
            status: '401',
            statusDescription: 'Unauthorized',
            headers: {
                'www-authenticate': [{
                    key: 'WWW-Authenticate',
                    value: 'Basic realm="Restricted"',
                }],
            },
        };
    }

    return request;
};

