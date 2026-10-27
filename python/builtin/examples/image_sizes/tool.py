def execute(inputs):
    # Policy for the injected ``upstream_status`` port: a read-only measurement keeps running after
    # an upstream failure and reports its own status; only tools that modify data must stop.
    return {
        "__status_code": 0,
        "__status_message": "运行成功",
        "sizes": [[int(image.shape[0]), int(image.shape[1])] for image in inputs["images"]],
    }
