using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ResultsParser
{
    class Program
    {
        class Result
        {
            public float[] Generations { get; set; }
            public float[] Times { get; set; }
            public int[] GenerationsHistory { get; set; }
            public float[] TimesHistory { get; set; }
        }

        static async Task<Result> ParseFileAsync(string path)
        {
            float[] ParseStats(string stats)
            {
                stats = stats.Replace("Stats(", "").Replace(")", "");
                var result = new float[5];

                var tokens = stats.Split(new[] { ',', '=' }, StringSplitOptions.RemoveEmptyEntries);

                for (int i = 0; i < 5; i++)
                    result[i] = float.Parse(tokens[1 + i * 2]);

                return result;
            }

            using (var stream = new StreamReader(path))
            {
                await stream.ReadLineAsync(); // Generations:
                string generations = await stream.ReadLineAsync();
                await stream.ReadLineAsync();
                await stream.ReadLineAsync(); // Time:
                string times = await stream.ReadLineAsync();
                await stream.ReadLineAsync();
                await stream.ReadLineAsync(); // Required generations + fitness history:

                var generationsHistory = new int[50];
                for (int i = 0; i < generationsHistory.Length; i++)
                {
                    string line = await stream.ReadLineAsync();
                    var index = line.IndexOf(',');
                    int value = int.Parse(line.Substring(0, index));
                    generationsHistory[i] = value;
                }

                float[] g = ParseStats(generations);
                float[] t = ParseStats(times);

                await stream.ReadLineAsync();
                await stream.ReadLineAsync(); // Times:
                string timesHistoryString = await stream.ReadLineAsync();
                string[] tokens = timesHistoryString.Substring(1, timesHistoryString.Length - 2).Split(',', StringSplitOptions.RemoveEmptyEntries);
                var timesHistory = new float[50];
                for (int i = 0; i < timesHistory.Length; i++)
                    timesHistory[i] = float.Parse(tokens[i]);
                
                return new Result
                {
                    Generations = g,
                    Times = t,
                    GenerationsHistory = generationsHistory,
                    TimesHistory = timesHistory,
                };
            }
        }

        static async Task Main(string[] args)
        {
            const string ResultsPath = @"E:\Programming\School\Masters-Semestral-Project\src\MastersSemestralProject\MastersSemestralProject\results\final";
            CultureInfo.CurrentCulture = new CultureInfo("en-US");

            var data = new List<(Result result, string alg, string seed, int reward)>();
            foreach (var path in Directory.EnumerateFiles(ResultsPath, "max-reward*"))
            {
                var rewardIndex = path.IndexOf("max-reward=") + 11;
                var rewardIndexEnd = path.IndexOf('_', rewardIndex);
                var reward = int.Parse(path.Substring(rewardIndex, rewardIndexEnd - rewardIndex));
                var algIndex = path.IndexOf("alg=") + 4;
                var algIndexEnd = path.IndexOf('_', algIndex);
                var alg = path.Substring(algIndex, algIndexEnd - algIndex);
                var seedIndex = path.IndexOf("seed=") + 5;
                var seed = path.Substring(seedIndex);

                var result = await ParseFileAsync(path);

                data.Add((result, alg, seed, reward));
            }

            foreach (var seedGroup in data.GroupBy(t => t.seed))
            {
                var seed = seedGroup.Key;
                var columns = seedGroup.OrderBy(t => t.reward).ToList();
                using (var swGenerations = new StreamWriter(Path.Combine(ResultsPath, $"resuts_generations_seed={seed}")))
                using (var swTimes = new StreamWriter(Path.Combine(ResultsPath, $"resuts_times_seed={seed}")))
                {
                    await swGenerations.WriteLineAsync(string.Join(";", columns.Select(t => $"{t.alg}{t.reward}")));
                    await swTimes.WriteLineAsync(string.Join(";", columns.Select(t => $"{t.alg}{t.reward}")));

                    for (int i = 0; i < 50; i++)
                    {
                        string genRow = string.Join(";", columns.Select(t => t.result.GenerationsHistory[i]));
                        string timesRow = string.Join(";", columns.Select(t => t.result.TimesHistory[i]));

                        await Task.WhenAll(
                            swGenerations.WriteLineAsync(genRow),
                            swTimes.WriteLineAsync(timesRow)
                            );
                    }
                }
            }
        }
    }
}
