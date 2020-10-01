using System;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using OnnxConsole.Models;

namespace OnnxConsole
{
	class Program
	{
		public static string GetAbsolutePath(string relativePath)
		{
			FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
			string assemblyFolderPath = _dataRoot.Directory.FullName;

			string fullPath = Path.Combine(assemblyFolderPath, relativePath);

			return fullPath;
		}

		static void Main(string[] args)
		{
			var modelsRelativePath = @"../../../OnnxModels";
			string modelsPath = GetAbsolutePath(modelsRelativePath);
			var modelFilePath = Path.Combine(modelsPath, "roberta_github_issues.onnx");

			var mlContext = new MLContext();

			/*var data = mlContext.Data.LoadFromEnumerable(new List<RobertaGithubIssue>());

			Action<RobertaGithubIssue, RobertaGithubIssueEncoded> mapping = (input, output) =>
			{
				output.input_ids = new Int64[]
				{
					0, 19186, 48527,     4, 28512, 42686, 32483, 46159,  1640,
					42686, 32483, 46159,     4, 43815, 42686, 32483,    43,  1364,
					129,   683,     4, 32048,     5, 13619, 46159,    15,     5,
					595,  3166, 48527,     7,  6039, 42686, 32483,  1364,   129,
					13,     5,    78, 15019,   145,   554,     4,  5053,  7757,
					15019,    34,  6032,     4, 42124, 42686, 32483, 15423,     7,
					48955,     4,     2,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1
				};
				output.attention_mask = new Int64[]
				{
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
				};
			};

			var pipeline = mlContext.Transforms.CustomMapping(mapping, "roberta_encoding")
				.Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelFilePath,
						outputColumnNames: RobertaGithubIssuesModelSettings.ModelOutput, 
						inputColumnNames: RobertaGithubIssuesModelSettings.ModelInput)); ;
			*/

			var data = mlContext.Data.LoadFromEnumerable(new List<RobertaGithubIssueEncoded>());

			var pipeline = mlContext.Transforms.ApplyOnnxModel(modelFile: modelFilePath,
						outputColumnNames: RobertaGithubIssuesModelSettings.ModelOutput,
						inputColumnNames: RobertaGithubIssuesModelSettings.ModelInput); 
			var model = pipeline.Fit(data);

			var issue = new RobertaGithubIssueEncoded()
			{
				input_ids = new Int64[]
				{
					0, 19186, 48527,     4, 28512, 42686, 32483, 46159,  1640,
					42686, 32483, 46159,     4, 43815, 42686, 32483,    43,  1364,
					129,   683,     4, 32048,     5, 13619, 46159,    15,     5,
					595,  3166, 48527,     7,  6039, 42686, 32483,  1364,   129,
					13,     5,    78, 15019,   145,   554,     4,  5053,  7757,
					15019,    34,  6032,     4, 42124, 42686, 32483, 15423,     7,
					48955,     4,     2,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1,     1,     1,     1,     1,     1,     1,     1,
					1,     1
				},
				attention_mask = new Int64[]
				{
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
				}
			};

			var predictionEngine = mlContext.Model.CreatePredictionEngine<RobertaGithubIssueEncoded, RobertaGithubIssuesOutput>(model);
			var predictions = predictionEngine.Predict(issue);

			foreach (var prediction in predictions.predictions)
			{
				Console.WriteLine(prediction);
			}

			mlContext.Model.Save(model, data.Schema, Path.Combine(modelsRelativePath, "github_issues_onnx.zip"));

		}
	}
}
