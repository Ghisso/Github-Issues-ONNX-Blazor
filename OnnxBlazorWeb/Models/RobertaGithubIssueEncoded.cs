using Microsoft.ML.Data;
using System;

namespace OnnxBlazorWeb.Models
{
	public class RobertaGithubIssueEncoded
	{
		[VectorType(128)]
		public Int64[] input_ids;

		[VectorType(128)]
		public Int64[] attention_mask;
	}
}
