resource "aws_lambda_function" "basic_auth" {
  filename         = "lambda/basic_auth.zip"
  function_name    = "basic-auth-edge"
  role             = aws_iam_role.lambda_edge.arn
  handler          = "basic_auth.handler"
  runtime          = "nodejs18.x"
  publish          = true

  source_code_hash = filebase64sha256("lambda/basic_auth.zip")
}

