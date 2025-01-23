from SemEval_Task8.utils import label_answer_type
from SemEval_Task8.sql_manager import (
    generate_sql_prompt,
    load_dataset_into_db,
    preprocess_query_for_postgresql,
    execute_sql_query,
    post_process_result
)

def split_into_batches(data, batch_size):
    """
    Splits data into batches for processing.
    """
    qa_pairs = [
        {"question": question, "answer": answer, "dataset": dataset}
        for question, answer, dataset in zip(data["question"], data["answer"], data["dataset"])
    ]
    return [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]


def process_batch(tokenizer, model, engine, classifier, vectorizer, label_encoder, batch):
    """
    Processes a batch of QA pairs.
    """
    # Load the classifier, vectorizer, and label encoder

    results = []
    questions = [qa['question'] for qa in batch]
    dataset_ids = [qa['dataset'] for qa in batch]
    gts = [qa['answer'] for qa in batch]
    question_vectors = vectorizer.transform(questions)
    predicted_labels = classifier.predict(question_vectors)
    predicted_types = label_encoder.inverse_transform(predicted_labels)

    for idx, (question, dataset_id, gt, predicted_type) in enumerate(zip(questions, dataset_ids, gts, predicted_types)):
        try:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            conn, schema = load_dataset_into_db(dataset_id, engine)
            prompt = generate_sql_prompt(schema, dataset_id, question)

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            generated_ids = model.generate(**inputs, 
                num_return_sequences=1,
                max_new_tokens=400, 
                num_beams=1,
                do_sample=False,
                eos_token_id=100001,
                pad_token_id=100001
            )
            generated_query = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("```sql")[-1].strip()

            preprocessed_query = preprocess_query_for_postgresql(generated_query, schema.columns)
            result = execute_sql_query(conn, preprocessed_query)
            processed_result = post_process_result(result, predicted_type)

            results.append({"question": question, "result": processed_result, "type": predicted_type})
        except Exception as e:
            results.append({"question": question, "error": str(e)})

    return results
