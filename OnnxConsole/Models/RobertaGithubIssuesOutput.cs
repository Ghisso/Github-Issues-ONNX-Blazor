using Microsoft.ML;
using Microsoft.ML.Data;

namespace OnnxConsole.Models
{
	class RobertaGithubIssuesOutput
	{
		[ColumnName("output")]
		public float[] predictions;
	}
}
