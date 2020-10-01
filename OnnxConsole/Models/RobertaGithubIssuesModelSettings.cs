using System;
using System.Collections.Generic;
using System.Text;

namespace OnnxConsole.Models
{
    public struct RobertaGithubIssuesModelSettings
    {

        // input tensor name
        public static readonly string[] ModelInput = { "input_ids", "attention_mask" };

        // output tensor name
        public static readonly string[] ModelOutput = { "output" };
    }
}
