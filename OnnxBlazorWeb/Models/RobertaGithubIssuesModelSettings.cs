
namespace OnnxBlazorWeb.Models
{
	public class RobertaGithubIssuesModelSettings
	{
		// input tensor name
		public static readonly string[] ModelInput = { "input_ids", "attention_mask" };

		// output tensor name
		public static readonly string[] ModelOutput = { "output" };
	}
}
