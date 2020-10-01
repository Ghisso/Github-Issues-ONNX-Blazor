using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace OnnxConsole.Models
{
	class RobertaGithubIssueEncoded
	{

		[ColumnName("input_ids"), VectorType(128)]
		public Int64[] input_ids;

		[ColumnName("attention_mask"), VectorType(128)]
		public Int64[] attention_mask;
	}
}
