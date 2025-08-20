
from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from evaluation.datasets_loader import get_dataset_handler
from mathruler.grader import extract_boxed_content, grade_answer

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8)
args = parser.parse_args()

print('[init] loading model …')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
)

sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10,
)

stop_event = threading.Event()    # 程序整体退出
pause_event = threading.Event()   # 请求期间暂停

def gpu_idle_worker():
    print('[idle_worker] started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] paused.')
                running = False
            time.sleep(0.1)
            continue
        else:
            if not running:
                print('[idle_worker] resumed.')
                running = True
        try:
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] RuntimeError: {e}. Sleeping 1s …')
            time.sleep(1)
    print('[idle_worker] stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_sec: int, *args, **kwargs):
    result_holder = {}
    error_holder = {}

    def _target():
        try:
            result_holder['value'] = func(*args, **kwargs)
        except Exception as e:
            error_holder['error'] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_sec)

    if t.is_alive():
        raise TimeoutException()
    if 'error' in error_holder:
        raise error_holder['error']
    return result_holder.get('value')

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():

    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] received {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': q}
            ])
    print('[server] valid_chats prepared.')

    if valid_chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in valid_chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] generation completed.')

    def process_single(question, golden_answer, response):
        results = [extract_boxed_content(out.text) for out in response.outputs]

        answer_counts = {}
        for res in results:
            matched = False
            for exist_ans in list(answer_counts.keys()):
                try:
                    if (grade_answer(res, exist_ans) or grade_answer(exist_ans, res)
                        or res == exist_ans
                        or ('no ' in res.lower() and 'no ' in exist_ans.lower())):
                        answer_counts[exist_ans] += 1
                        matched = True
                        break
                except Exception:
                    continue
            if not matched:
                answer_counts[res] = 1

        max_count    = max(answer_counts.values()) if answer_counts else 0
        majority_ans = max(answer_counts, key=answer_counts.get) if answer_counts else ''
        score        = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score if majority_ans == golden_answer and score > 0.1 else 0,
            'results':  results
        }

    results_all = []
    response_idx = 0
    for q, a in zip(questions, answers):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                item = run_with_timeout(process_single, 10, q, a, response)
                results_all.append(item)
            else:
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
        except TimeoutException:
            print(f'[server] timeout: {q}')
            print(f'[server] timeout: {a}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    'timeout'
            })
        except Exception as e:
            print(f'[server] error: {e}')
            results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
    print('[server] results_all completed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    time.sleep(10)
    print(f'[server] processed {name}, results saved to {out_path}.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        stop_event.set()
        idle_thread.join()
        print('[main] shutdown completed.')