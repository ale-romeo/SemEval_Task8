from SemEval_Task8.sql_manager import (
    generate_sql_prompt,
    load_dataset_into_db,
    preprocess_query_for_postgresql,
    execute_sql_query
)
from SemEval_Task8.utils import post_process_result

def split_into_batches(data, batch_size):
    """
    Splits data into batches for processing.
    """
    qa_pairs = [
        {"question": question, "answer": answer, "dataset": dataset}
        for question, answer, dataset in zip(data["question"], data["answer"], data["dataset"])
    ]
    return [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]


def process_batch(tokenizer, model, engine, predictor, batch):
    """
    Processes a batch of QA pairs.
    """
    results = []
    questions = [qa['question'] for qa in batch]
    dataset_ids = [qa['dataset'] for qa in batch]
    answers = [qa['answer'] for qa in batch]

    # Predict answer types for all questions
    classifier, vectorizer, label_encoder = predictor
    question_vectors = vectorizer.transform(questions)
    predicted_labels = classifier.predict(question_vectors)
    predicted_types = label_encoder.inverse_transform(predicted_labels)

    for idx, (question, dataset_id, predicted_type, answer) in enumerate(zip(questions, dataset_ids, predicted_types, answers)):
        try:
            # Ensure tokenizer has a padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            conn, schema = load_dataset_into_db(dataset_id, engine)

            prompt = generate_sql_prompt(schema, dataset_id, question)

            # Tokenize and generate SQL queries
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            generated_ids = model.generate(
                **inputs,
                num_return_sequences=1,
                max_new_tokens=400,
                do_sample=False,
                num_beams=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_query = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("[SQL]")[-1].strip()
            
            if not generated_query.strip():
                print(f"Empty SQL query generated for: {question}")
                continue

            preprocessed_query = preprocess_query_for_postgresql(generated_query, schema)
            print(question)
            print(preprocessed_query)
            
            try:
                result = execute_sql_query(conn, preprocessed_query)

            except Exception as e:
                result = f"Execution Error: {str(e)}"

            processed_result = post_process_result(result, predicted_type, schema)
            print(result, processed_result, answer)
            results.append({"question": question, "result": processed_result, "type": predicted_type})

        except Exception as e:
            results.append({"question": question, "error": str(e)})

    return results

