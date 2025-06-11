#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include "Model.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <numeric> 
#include <ctime>

class TokenPredictor : public Model
{
public:
  // Vocabulary mapping
  std::unordered_map<std::string, int> vocab_to_id;
  std::unordered_map<int, std::string> id_to_vocab;
  int vocab_size;
  int embedding_dim;
  int context_length;
  bool use_attention;

  // Embedding layer weights (separate from regular layers)
  double* embedding_weights;
  double* embedding_gradients;

  // Add gradient clipping threshold
  double gradient_clip_threshold;

public:
  TokenPredictor(int vocab_size, int embedding_dim, int context_length, double learning_rate = 0.01, bool use_attention_arch = true)
    : Model(learning_rate), 
    vocab_size(vocab_size), 
    embedding_dim(embedding_dim),
    context_length(context_length), 
    use_attention(use_attention_arch),
    gradient_clip_threshold(5.0)  // Add gradient clipping
  {
    // ALWAYS allocate memory first
    embedding_weights = new double[vocab_size * embedding_dim];
    embedding_gradients = new double[vocab_size * embedding_dim];

    // Try to load existing embeddings
    bool existing_embeddings = LoadEmbeddings("embeddings", embedding_weights, vocab_size, embedding_dim);
    if (!existing_embeddings)
    {
      std::cout << "Could not load embeddings, initializing new ones..." << std::endl;

      srand(42);
      double xavier_std = sqrt(1.0 / embedding_dim);  // Reduced from sqrt(2.0 / (vocab_size + embedding_dim))
      for (int i = 0; i < vocab_size * embedding_dim; i++)
      {
        embedding_weights[i] = xavier_std * nrnd();
        embedding_gradients[i] = 0.0;
      }

      std::cout << "Initialized new embeddings with Xavier initialization" << std::endl;
    }
  }

  ~TokenPredictor()
  {
    delete[] embedding_weights;
    delete[] embedding_gradients;
  }



  bool SaveEmbeddings(const std::string& filename/*, const double* embedding_weights, int vocab_size, int embedding_dim*/)
  {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) 
    {
      std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
      return false;
    }

    // Write header information
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));

    // Write embedding weights
    size_t total_size = static_cast<size_t>(vocab_size) * embedding_dim;
    file.write(reinterpret_cast<const char*>(embedding_weights), total_size * sizeof(double));

    file.close();
    std::cout << "Successfully saved embeddings to: " << filename << std::endl;
    return true;    
  }

  bool SaveEmbeddings(const std::string& filename,
    const double* embedding_weights,
    int vocab_size,
    int embedding_dim)
  {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
      return false;
    }

    try {
      // Write header information
      file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
      file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));

      // Write embedding weights
      size_t total_size = static_cast<size_t>(vocab_size) * embedding_dim;
      file.write(reinterpret_cast<const char*>(embedding_weights),
        total_size * sizeof(double));

      file.close();
      std::cout << "Successfully saved embeddings to: " << filename << std::endl;
      return true;

    }
    catch (const std::exception& e) {
      std::cerr << "Error writing to file: " << e.what() << std::endl;
      return false;
    }
  }

  // Load embedding weights from binary file
  bool LoadEmbeddings(const std::string& filename,
    double* embedding_weights,
    int expected_vocab_size,
    int expected_embedding_dim)
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
      return false;
    }

    try {
      // Check if embedding_weights pointer is valid
      if (embedding_weights == nullptr) {
        std::cerr << "Error: embedding_weights pointer is null" << std::endl;
        return false;
      }

      // Read header information
      int file_vocab_size, file_embedding_dim;
      file.read(reinterpret_cast<char*>(&file_vocab_size), sizeof(file_vocab_size));

      // Check if header read was successful
      if (file.fail() || file.gcount() != sizeof(file_vocab_size)) {
        std::cerr << "Error: Could not read vocab_size from file header" << std::endl;
        return false;
      }

      file.read(reinterpret_cast<char*>(&file_embedding_dim), sizeof(file_embedding_dim));

      // Check if header read was successful
      if (file.fail() || file.gcount() != sizeof(file_embedding_dim)) {
        std::cerr << "Error: Could not read embedding_dim from file header" << std::endl;
        return false;
      }

      std::cout << "File header - vocab_size: " << file_vocab_size << ", embedding_dim: " << file_embedding_dim << std::endl;

      // Validate dimensions are reasonable
      if (file_vocab_size <= 0 || file_embedding_dim <= 0 ||
        file_vocab_size > 1000000 || file_embedding_dim > 10000) {
        std::cerr << "Error: Invalid dimensions in file. vocab_size: " << file_vocab_size
          << ", embedding_dim: " << file_embedding_dim << std::endl;
        return false;
      }

      // Validate dimensions match expected values
      if (file_vocab_size != expected_vocab_size || file_embedding_dim != expected_embedding_dim) {
        std::cerr << "Error: Dimension mismatch. Expected: vocab=" << expected_vocab_size
          << ", embed=" << expected_embedding_dim
          << ". File contains: vocab=" << file_vocab_size
          << ", embed=" << file_embedding_dim << std::endl;
        return false;
      }

      // Calculate total size and check for overflow
      size_t total_size = static_cast<size_t>(file_vocab_size) * file_embedding_dim;
      size_t bytes_to_read = total_size * sizeof(double);

      std::cout << "About to read " << total_size << " doubles (" << bytes_to_read << " bytes)" << std::endl;

      // Check if the remaining file size is sufficient
      std::streampos current_pos = file.tellg();
      file.seekg(0, std::ios::end);
      std::streampos file_size = file.tellg();
      file.seekg(current_pos);

      std::streampos remaining_bytes = file_size - current_pos;
      if (remaining_bytes < static_cast<std::streampos>(bytes_to_read)) {
        std::cerr << "Error: File too small. Expected " << bytes_to_read
          << " bytes, but only " << remaining_bytes << " bytes remaining" << std::endl;
        return false;
      }

      // Read embedding weights in chunks to avoid potential issues with very large reads
      const size_t chunk_size = 1024 * 1024; // 1MB chunks
      size_t bytes_read = 0;
      char* data_ptr = reinterpret_cast<char*>(embedding_weights);

      while (bytes_read < bytes_to_read) {
        size_t bytes_this_chunk = std::min(chunk_size, bytes_to_read - bytes_read);

        file.read(data_ptr + bytes_read, bytes_this_chunk);

        if (file.fail()) {
          std::cerr << "Error: Failed to read chunk at byte offset " << bytes_read << std::endl;
          return false;
        }

        bytes_read += file.gcount();

        if (file.gcount() != static_cast<std::streamsize>(bytes_this_chunk)) {
          std::cerr << "Error: Incomplete read. Expected " << bytes_this_chunk
            << " bytes, got " << file.gcount() << " bytes" << std::endl;
          return false;
        }
      }

      file.close();
      std::cout << "Successfully loaded embeddings from: " << filename
        << " (vocab_size: " << file_vocab_size << ", embedding_dim: " << file_embedding_dim
        << ", total bytes: " << bytes_read << ")" << std::endl;
      return true;

    }
    catch (const std::exception& e) {
      std::cerr << "Error reading from file: " << e.what() << std::endl;
      return false;
    }
  }

  // Build vocabulary from text corpus automatically
  void BuildVocabularyFromCorpus(const std::vector<std::string>& texts, int max_vocab_size = 1000)
  {
    std::unordered_map<std::string, int> word_counts;

    // Count word frequencies
    for (const auto& text : texts)
    {
      std::istringstream iss(text);
      std::string word;
      while (iss >> word)
      {
        // Simple preprocessing: lowercase and remove punctuation
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        if (!word.empty())
        {
          word_counts[word]++;
        }
      }
    }

    // Sort by frequency
    std::vector<std::pair<int, std::string>> freq_words;
    for (const auto& pair : word_counts)
    {
      freq_words.push_back({ pair.second, pair.first });
    }
    std::sort(freq_words.rbegin(), freq_words.rend());
    vocab_size = freq_words.size() + 4;
    // Build vocabulary with special tokens
    vocab_to_id.clear();
    id_to_vocab.clear();

    // Add special tokens first
    vocab_to_id["<PAD>"] = 0;
    vocab_to_id["<UNK>"] = 1;
    vocab_to_id["<START>"] = 2;
    vocab_to_id["<END>"] = 3;

    id_to_vocab[0] = "<PAD>";
    id_to_vocab[1] = "<UNK>";
    id_to_vocab[2] = "<START>";
    id_to_vocab[3] = "<END>";

    // Add most frequent words
    int vocab_idx = 4;
    for (const auto& pair : freq_words)
    {
      if (vocab_idx >= std::min(max_vocab_size, vocab_size)) break;

      vocab_to_id[pair.second] = vocab_idx;
      id_to_vocab[vocab_idx] = pair.second;
      vocab_idx++;
    }

    std::cout << "Built vocabulary with " << vocab_to_id.size() << " tokens" << std::endl;
  }

  // FIXED: Simplified architecture for better learning
  void BuildTokenPredictorArchitecture(int hidden_size = 64, int num_layers = 2)
  {
    // Create input layer for embeddings
    CreateInputLayer(1, context_length * embedding_dim, 1);

    if (use_attention)
    {
      std::cout << "Building SIMPLIFIED attention-based architecture..." << std::endl;

      // FIXED: Simpler attention configuration
      //CreateAttentionLayer(context_length, embedding_dim, 16, 16, 2);  // Reduced complexity

      // FIXED: Smaller hidden layers for stability
      CreateResidualFullLayer(hidden_size, Layer::tanh);
      CreateResidualFullLayer(hidden_size, Layer::tanh);
      CreateResidualFullLayer(hidden_size , Layer::sigmoid);
    }
    else
    {
      std::cout << "Building MLP-only architecture..." << std::endl;

      // FIXED: Fewer, smaller layers
      CreateFullLayer(hidden_size, Layer::tanh);
      CreateFullLayer(hidden_size / 2, Layer::relu);
    }

    // FIXED: Use softmax activation for proper probability distribution
    CreateFullLayer(vocab_size, Layer::sigmoid);
  }

  // Add vocabulary mapping
  void BuildVocabulary(const std::vector<std::string>& vocabulary)
  {
    vocab_to_id.clear();
    id_to_vocab.clear();

    for (int i = 0; i < vocabulary.size() && i < vocab_size; i++)
    {
      vocab_to_id[vocabulary[i]] = i;
      id_to_vocab[i] = vocabulary[i];
    }
  }

  // Convert tokens to IDs
  std::vector<int> TokensToIds(const std::vector<std::string>& tokens)
  {
    std::vector<int> ids;
    for (const auto& token : tokens)
    {
      auto it = vocab_to_id.find(token);
      if (it != vocab_to_id.end())
      {
        ids.push_back(it->second);
      }
      else
      {
        ids.push_back(1); // <UNK> token
      }
    }
    return ids;
  }

  // Embed tokens into dense vectors
  std::vector<double> EmbedTokens(const std::vector<int>& token_ids)
  {
    std::vector<double> embedded(context_length * embedding_dim, 0.0);

    for (int i = 0; i < std::min((int)token_ids.size(), context_length); i++)
    {
      int token_id = token_ids[i];
      if (token_id >= 0 && token_id < vocab_size)
      {
        // Copy embedding for this token
        for (int j = 0; j < embedding_dim; j++)
        {
          embedded[i * embedding_dim + j] = embedding_weights[token_id * embedding_dim + j];
        }
      }
    }

    return embedded;
  }

  // FIXED: Proper softmax implementation (removed - using Layer::softmax activation)
  std::vector<double> ApplySoftmax(const std::vector<double>& logits)
  {
    std::vector<double> probabilities(logits.size());
    double max_logit = *std::max_element(logits.begin(), logits.end());

    // Subtract max for numerical stability
    double sum = 0.0;
    for (int i = 0; i < logits.size(); i++)
    {
      probabilities[i] = exp(logits[i] - max_logit);
      sum += probabilities[i];
    }

    // Normalize
    for (int i = 0; i < probabilities.size(); i++)
    {
      probabilities[i] /= sum;
    }

    return probabilities;
  }

  // Predict next token given context
  std::vector<double> PredictNextToken(const std::vector<std::string>& context_tokens)
  {
    // Convert tokens to IDs
    std::vector<int> token_ids = TokensToIds(context_tokens);
    // Embed tokens
    std::vector<double> embedded = EmbedTokens(token_ids);
    // Forward pass through the network
    InputData(embedded);
    // Get output (already softmax from Layer::softmax activation)
    std::vector<double> output = GetOutput();

    // Create indices and sort them by corresponding output values
    std::vector<int> indices(output.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by output values (descending - highest probabilities first)
    std::sort(indices.begin(), indices.end(), [&output](int a, int b) 
      {
      return output[a] > output[b];
      });

    // Create ordered predictions using sorted indices
    std::vector<std::string> orderedPredictions;
    for (int i = 0; i < indices.size(); i++)
    {
      int token_id = indices[i];
      orderedPredictions.push_back(id_to_vocab[token_id]);
    }

    // FIXED: Ensure probabilities are valid
    double sum = 0.0;
    for (double prob : output) {
      sum += prob;
    }

    if (sum > 0.0) {
      for (double& prob : output) {
        prob /= sum;
      }
    }

    return output;
  }

  // Helper function for proper sampling
  int SampleFromDistribution(const std::vector<double>& probabilities)
  {
    double random_val = (double)rand() / RAND_MAX;
    double cumulative = 0.0;

    for (int i = 0; i < probabilities.size(); i++)
    {
      cumulative += probabilities[i];
      if (random_val <= cumulative)
      {
        return i;
      }
    }

    return probabilities.size() - 1; // Fallback
  }

  // Generate text with temperature sampling
  std::string GenerateText(const std::vector<std::string>& seed_tokens, int max_length = 50, double temperature = 0.8)
  {
    std::vector<std::string> generated = seed_tokens;
    srand(time(nullptr)); // Better random seed

    for (int i = 0; i < max_length; i++)
    {
      // Get context
      std::vector<std::string> context;
      int start = std::max(0, (int)generated.size() - context_length);
      for (int j = start; j < generated.size(); j++)
      {
        context.push_back(generated[j]);
      }

      // Pad if needed
      while (context.size() < context_length)
      {
        context.insert(context.begin(), "<PAD>");
      }

      // Predict next token
      std::vector<double> probabilities = PredictNextToken(context);

      // Apply temperature scaling for better diversity
      std::vector<double> scaled_probs(probabilities.size());

      double sum = 0.0;
      for (int k = 0; k < probabilities.size(); k++)
      {
        scaled_probs[k] = pow(probabilities[k], 1.0 / temperature);
        sum += scaled_probs[k];
      }

      // Normalize
      for (int k = 0; k < scaled_probs.size(); k++)
      {
        scaled_probs[k] /= sum;
      }

      // Sample from distribution instead of argmax
      int next_token_id = SampleFromDistribution(scaled_probs);

      // Convert back to token and filter special tokens
      auto it = id_to_vocab.find(next_token_id);
      if (it != id_to_vocab.end())
      {
        if (it->second == "<END>") break;
        if (it->second != "<PAD>" && it->second != "<UNK>") // Skip special tokens
        {
          generated.push_back(it->second);
        }
        else
        {
          // If we get a special token, try next best option
          continue;
        }
      }
      else
      {
        break;
      }
    }

    // Convert to string
    std::string result;
    for (int i = seed_tokens.size(); i < generated.size(); i++)
    {
      if (i > seed_tokens.size()) result += " ";
      result += generated[i];
    }

    return result;
  }

  // FIXED: Improved training sequence with better error handling
  void TrainSequence(const std::vector<std::string>& tokens)
  {
    if (tokens.size() < 2) return;

    // Create training pairs: context -> next_token
    for (int i = 0; i < tokens.size() - 1; i++)
    {
      // Get context (up to context_length tokens before current position)
      std::vector<std::string> context;
      int start = std::max(0, i - context_length + 1);
      for (int j = start; j <= i; j++)
      {
        context.push_back(tokens[j]);
      }

      // Pad context if needed
      while (context.size() < context_length)
      {
        context.insert(context.begin(), "<PAD>");
      }

      // Target is the next token
      std::string target_token = tokens[i + 1];
      auto target_it = vocab_to_id.find(target_token);
      int target_id = (target_it != vocab_to_id.end()) ? target_it->second : 1; // Use <UNK> if not found

      // FIXED: Create proper one-hot target vector
      std::vector<double> target_vector(vocab_size, 0.0);
      if (target_id >= 0 && target_id < vocab_size)
      {
        target_vector[target_id] = 1.0;
      }

      // Train on this example
      TrainExample(context, target_vector);
    }
  }

  // FIXED: Gradient clipping function
  double ClipGradient(double gradient)
  {
    if (gradient > gradient_clip_threshold) return gradient_clip_threshold;
    if (gradient < -gradient_clip_threshold) return -gradient_clip_threshold;
    return gradient;
  }

  // Train on a single example
  void TrainExample(const std::vector<std::string>& context_tokens, const std::vector<double>& target)
  {
    training_ = true;

    // Convert tokens to IDs and embed
    std::vector<int> token_ids = TokensToIds(context_tokens);
    std::vector<double> embedded = EmbedTokens(token_ids);

    // Forward pass
    InputData(embedded);

    // Backward pass
    TrainData(target);

    // Update embedding weights
    UpdateEmbeddingWeights(token_ids);

    training_ = false;
  }

  // FIXED: Update embedding weights with gradient clipping
  void UpdateEmbeddingWeights(const std::vector<int>& token_ids)
  {
    // Get gradients from input layer
    Layer* input_layer = GetLayer(0);

    // Propagate gradients back to embeddings
    for (int i = 0; i < std::min((int)token_ids.size(), context_length); i++)
    {
      int token_id = token_ids[i];
      if (token_id >= 0 && token_id < vocab_size)
      {
        // Update embedding for this token
        for (int j = 0; j < embedding_dim; j++)
        {
          int embed_idx = token_id * embedding_dim + j;
          int input_idx = i * embedding_dim + j;

          // FIXED: Calculate gradient with clipping
          double gradient = ClipGradient(input_layer->errors[input_idx]);
          embedding_gradients[embed_idx] = gradient;

          // FIXED: Update weight with clipped gradient
          embedding_weights[embed_idx] -= GetLearningRate() * gradient;
        }
      }
    }
  }

  // Parse text into tokens (simple whitespace tokenization)
  std::vector<std::string> TokenizeText(const std::string& text)
  {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word)
    {
      // Simple preprocessing: lowercase and remove punctuation
      std::transform(word.begin(), word.end(), word.begin(), ::tolower);
      word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

      if (!word.empty())
      {
        tokens.push_back(word);
      }
    }

    return tokens;
  }

  // Train from text strings
  void TrainFromTexts(const std::vector<std::string>& texts)
  {
    for (const auto& text : texts)
    {
      std::vector<std::string> tokens = TokenizeText(text);
      if (tokens.size() > 1)
      {
        TrainSequence(tokens);
      }
    }
  }

  // Enhanced evaluation with specific metrics for complex patterns
  double EvaluateComplexPatterns(const std::vector<std::string>& test_texts)
  {
    double long_range_accuracy = 0.0;
    int long_range_count = 0;

    for (const auto& text : test_texts) {
      std::vector<std::string> tokens = TokenizeText(text);
      if (tokens.size() < 8) continue; // Only test long sequences

      // Test predictions at various distances
      for (int i = 4; i < tokens.size(); i++) {
        std::vector<std::string> context;
        int start = std::max(0, i - context_length);
        for (int j = start; j < i; j++) {
          context.push_back(tokens[j]);
        }

        while (context.size() < context_length) {
          context.insert(context.begin(), "<PAD>");
        }

        std::vector<double> probs = PredictNextToken(context);
        auto it = vocab_to_id.find(tokens[i]);
        if (it != vocab_to_id.end()) {
          // Check if correct token is in top 3 predictions
          std::vector<std::pair<double, int>> prob_pairs;
          for (int k = 0; k < probs.size(); k++) {
            prob_pairs.push_back({ probs[k], k });
          }
          std::sort(prob_pairs.rbegin(), prob_pairs.rend());

          for (int k = 0; k < std::min(3, (int)prob_pairs.size()); k++) {
            if (prob_pairs[k].second == it->second) {
              long_range_accuracy += 1.0;
              break;
            }
          }
          long_range_count++;
        }
      }
    }

    return long_range_count > 0 ? long_range_accuracy / long_range_count : 0.0;
  }

  // Get the current vocabulary size
  int GetVocabSize() const { return vocab_size; }
  int GetEmbeddingDim() const { return embedding_dim; }
  int GetContextLength() const { return context_length; }

  // Print vocabulary statistics
  void PrintVocabularyStats()
  {
    std::cout << "Vocabulary Statistics:" << std::endl;
    std::cout << "  Total tokens: " << vocab_to_id.size() << std::endl;
    std::cout << "  Max vocab size: " << vocab_size << std::endl;
    std::cout << "  Embedding dimension: " << embedding_dim << std::endl;
    std::cout << "  Context length: " << context_length << std::endl;
    std::cout << "  Architecture: " << (use_attention ? "ATTENTION" : "MLP-ONLY") << std::endl;
  }
};

// FIXED: Simplified training data for better learning
std::vector<std::string> GetSimplifiedTrainingData()
{
  return {
    // Simple, consistent patterns
    "the cat sits on the mat",
    "the dog runs in the park",
    "the bird flies in the sky",
    "the fish swims in the water",
    "the sun shines in the day",
    //// Repeated patterns for better learning
    "i like to eat apples",
    "i like to eat oranges",
    "i like to eat bananas",
    "we like to eat food",
    "they like to eat dinner",
    // Simple conditionals
    "if it rains we stay home",
    "if it snows we wear coats",
    "if it sunny we go out",
    "when we work we earn money",
    "when we study we learn things",
    // Repeat key patterns multiple times
    "the cat sits on the mat",
    "the dog runs in the park",
    "i like to eat apples",
    "if it rains we stay home",
    "when we work we earn money",
    // More variations
    "the small cat sits quietly",
    "the big dog runs fast",
    "the red bird flies high",
    "the blue fish swims deep",
    "the bright sun shines warm",
    // Simple sequences
    "first we wake up",
    "then we eat breakfast",
    "next we go to work",
    "finally we come home",
    "first we study hard",
    "then we take tests",
    "next we get grades",
    "finally we graduate",

    //// Additional 6-word examples
    "the cat sleeps on my bed",
    "the dog plays in our yard",
    "the bird sings in tall trees",
    "the fish hides under small rocks",
    "the moon glows in dark night",
    "i want to drink cold water",
    "i need to buy new shoes",
    "we have to clean the house",
    "they want to watch good movies",
    "you like to read interesting books",
    "if it gets cold we freeze",
    "if we run fast we win",
    "when it gets dark we sleep",
    "when we eat well we grow",
    "the quick cat jumps over fences",
    "the loud dog barks at strangers",
    "the small bird builds tiny nests",
    "the old fish swims very slowly",
    "the hot sun makes people sweat",
    "first we brush our white teeth",
    "then we put on clean clothes",
    "next we drive to the office",
    "finally we finish all our work",
  };
}

// FIXED: Main function with conservative parameters
int main(int argc, char* argv[])
{

  TokenPredictor predictor(142, 32, 10, 0.2);  // Much smaller and conservative
  //predictor.LoadEmbeddings("embediings", embedding_weights, vocab_size, dim);
  // Get simplified training data
  std::vector<std::string> training_texts = GetSimplifiedTrainingData();

  predictor.BuildVocabularyFromCorpus(training_texts, 300);
  predictor.PrintVocabularyStats();

  if (predictor.LoadModelBinary("languageModel") == false)
  {
    predictor.BuildTokenPredictorArchitecture(32, 2);  // Small hidden layers
  }

  std::vector<std::string> train_texts(training_texts.begin(), training_texts.begin() + training_texts.size());
  std::cout << "Training samples: " << train_texts.size() << std::endl;
  
  // FIXED: Much more conservative training parameters
  //double learning_rate = 0.2;  // Much smaller
  //int total_epochs = 2000;        // Fewer epochs

  //predictor.SetLearningRate(learning_rate);
  //predictor.SetDropRate(0.00);
  //for (int epoch = 0; epoch < total_epochs; epoch++)
  //{
  //  // FIXED: Train only once per epoch, not 3 times
  //  predictor.TrainFromTexts(train_texts);

  //  // Progress reporting every 50 epochs
  //  if (epoch % 50 == 0)
  //  {
  //    double train_error = predictor.GetTotalError();
  //    std::cout << "Epoch " << epoch << ", Train Error: " << train_error << ", LR: " << predictor.GetLearningRate() << std::endl;

  //    // Check prediction quality
  //    //std::vector<double> probs = predictor.PredictNextToken({ "the", "cat", "sits", "<PAD>" });
  //    std::vector<double> probs = predictor.PredictNextToken({ "<PAD>", "<PAD>", "<PAD>" , "<PAD>", "<PAD>", "<PAD>", "<PAD>" , "<PAD>", "the", "bird" });
  //    if (!probs.empty())
  //    {
  //      double max_prob = *std::max_element(probs.begin(), probs.end());
  //      std::cout << "  Max prediction probability: " << (max_prob * 100) << "%" << std::endl;
  //      predictor.SetLearningRate(predictor.GetLearningRate() * .99);
  //    }
  //  }  
  //}
  //
  //predictor.SaveModelBinary("languageModel");
  //predictor.SaveEmbeddings("embeddings"/*, embedding_weights, 142, 32*/);
  
  std::cout << "\nTraining completed!" << std::endl;

  // Simple testing
  std::cout << "\n=== Simple Prediction Testing ===" << std::endl;

  std::vector<std::vector<std::string>> test_contexts = {
      {"<PAD>","<PAD>", "<PAD>", "<PAD>","<PAD>", "<PAD>", "<PAD>", "the", "cat", "sits"},
      {"<PAD>", "<PAD>", "<PAD>","<PAD>", "<PAD>", "<PAD>", "the", "dog", "runs"},
      {"<PAD>","<PAD>", "<PAD>", "<PAD>","<PAD>", "<PAD>", "i", "like", "to", "eat"},
      {"<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "if", "it", "rains", "we"},
      {"<PAD>", "<PAD>", "<PAD>","<PAD>", "<PAD>", "when", "we", "work", "we", "earn" },
      {"<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>","<PAD>","<PAD>", "first", "we", "study"},
      {"<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>","then",  "we",  "take"},
      {"<PAD>","<PAD>", "<PAD>", "<PAD>","<PAD>","<PAD>", "<PAD>", "<PAD>","next", "we"},
      {"<PAD>","<PAD>", "<PAD>", "<PAD>","<PAD>","<PAD>", "we", "like","to"},
  };
  
  for (const auto& context : test_contexts)
  {
    std::cout << "\nContext: ";
    for (const auto& word : context)
    {
      std::cout << word << " ";
    }
    std::cout << std::endl;

    std::vector<double> probs = predictor.PredictNextToken(context);

    // Find top 3 predictions
    std::vector<std::pair<double, int>> prob_pairs;
    for (int i = 0; i < probs.size(); i++)
    {
      prob_pairs.push_back({ probs[i], i });
    }
    std::sort(prob_pairs.rbegin(), prob_pairs.rend());

    std::cout << "Top 3 predictions:" << std::endl;
    for (int i = 0; i < std::min(5, (int)prob_pairs.size()); i++)
    {
      int token_id = prob_pairs[i].second;
      double prob = prob_pairs[i].first;
      if (predictor.id_to_vocab.count(token_id))
      {
        std::cout << "  " << predictor.id_to_vocab[token_id] << ": " << (prob * 100) << "%" << std::endl;
      }
    }
  }

  std::string start_phrase = argv[1];
  std::vector<std::string> seed_tokens;
  std::istringstream iss(start_phrase);
  std::string token;

  while (iss >> token) 
  {
    seed_tokens.push_back(token);
  }
  std::string generatedText = predictor.GenerateText(seed_tokens, 6, 1);
  std::cout << start_phrase << " " << generatedText;
  std::cout << "\n=== FIXED Program Complete ===" << std::endl;
  return 0;
}
