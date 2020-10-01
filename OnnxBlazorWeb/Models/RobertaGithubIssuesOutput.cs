using Microsoft.ML.Data;

namespace OnnxBlazorWeb.Models
{
	public class RobertaGithubIssuesOutput
	{
		[ColumnName("output")]
		public float[] predictions;
	}
}
