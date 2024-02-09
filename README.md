# mpnet-rs
## What is this?
> This is a translation of MPNet from PyTorch into Rust Candle.
- The trained model I used is PatentSBERTa, which is designed to obtain embeddings optimized for the patent domain. 
- train pipeline is NOT yet prepared.
- If you have your own MPNet weights, they can be loaded using this carte.
## How to use
### get trained model
- download the model from [huggingface](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
- if you want to load model from .safetensors, you have to convert it yourself.
[this implementation](https://gist.github.com/epicfilemcnulty/1f55fd96b08f8d4d6693293e37b4c55e) might be helpful.
### load model and weights
```rust
use patentpick::mpnet::load_model;
let (model, tokenizer, pooler) = load_model("/path/to/model/and/tokenizer").unwrap();
```
### get embeddings(with pooler): see test function below
this is about how to get embeddings adn consine similarity 
```rust
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder,  Module};

use mpnet_rs::mpnet::{MPNetEmbeddings, MPNetConfig, create_position_ids_from_input_ids, cumsum, load_model, get_embeddings, normalize_l2, PoolingConfig, MPNetPooler};


fn test_get_embeddings() ->Result<()>{
    let path_to_checkpoints_folder = "D:/RustWorkspace/checkpoints/AI-Growth-Lab_PatentSBERTa".to_string();

    let (model, mut tokenizer, pooler) = load_model(path_to_checkpoints_folder).unwrap();

    let sentences = vec![
        "an invention that targets GLP-1",
        "new chemical that targets glucagon like peptide-1 ",
        "de novo chemical that targets GLP-1",
        "invention about GLP-1 receptor",
        "new chemical synthesis for glp-1 inhibitors",
        "It feels like I'm in America",
        "It's rainy. all day long.",
    ];
    let n_sentences = sentences.len();
    let embeddings = get_embeddings(&model, &tokenizer, Some(&pooler), &sentences).unwrap();

    let l2norm_embeds = normalize_l2(&embeddings).unwrap();
    println!("pooled embeddings {:?}", l2norm_embeds.shape());

    let mut similarities = vec![];
    for i in 0..n_sentences {
        let e_i = l2norm_embeds.get(i)?;
        for j in (i + 1)..n_sentences {
            let e_j = l2norm_embeds.get(j)?;
            let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())
}
```

## Note
### Pooling layer
- In the original PyTorch implementation in Transformers, the pooling layers are declared in the MPNetModel class
- I have implemented the pooling layer independently, separating it from the MPNetModel class.
### activation
- In the original implementation, tanh is used as the activation function for the pooling layers. 
- However, since it was difficult to find the implementation of tanh in Candle, I have set gelu as the default
## References
- [candle](https://github.com/huggingface/candle)
- [candle-tutorial](https://github.com/ToluClassics/candle-tutorial?tab=readme-ov-file)
- [transformers-mpnet](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/mpnet/modeling_mpnet.py)
- [PatentSBERTa](https://github.com/AI-Growth-Lab/PatentSBERTa)