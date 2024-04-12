import os
import pickle
import difflib
import textwrap

def format_text_state(state_text):
    rows = [' ; '.join(row) for row in state_text]
    text_string = ' ;\n'.join(rows)
    return text_string

def state_diffs(text0, text1):
    diffs = []
    for i in range(text0.shape[0]):
        for j in range(text0.shape[1]):
            if text0[i, j] != text1[i, j]:
                diffs.append((text0[i, j], text1[i, j]))
    return diffs

def transition_string(action, string0, string1, diffs):
    if len(diffs) == 0:
        diff_string = "Nothing happened"
    else:
        diff_strings = []
        for diff in diffs:
            ## Did agent direction change
            if 'Agent' in diff[0] and 'Agent' in diff[1]:
                pos = diff[0].split('(')[1].split(',')[0:2]
                direction0 = diff[0].split('direction')[1].split(')')[0].split('(')[1].split(',')
                direction1 = diff[1].split('direction')[1].split(')')[0].split('(')[1].split(',')
                carrying0 = diff[0].split('carrying=')[1].split(')')[0]
                carrying1 = diff[1].split('carrying=')[1].split(')')[0]
                if direction0 != direction1:
                    diff_strings.append(f'The agent (direction=({direction0[0].strip()}, {direction0[1].strip()}) '
                                        f'at pos ({pos[0].strip()}, {pos[1].strip()}) becomes an agent '
                                        f'(direction=({direction1[0].strip()}, {direction1[1].strip()}))'
                                        )
                elif carrying0 != carrying1:
                    diff_strings.append(f'The agent (carrying={carrying0} '
                                        f'at pos ({pos[0].strip()}, {pos[1].strip()}) becomes an agent '
                                        f'carrying={carrying1})'
                                        )
                else:
                    raise(ValueError('Agent strings do not match but direction and carrying are the same.'))

            ## Did agent carrying change

            ## Did agent position change

            ## Did cow position change

            ## Did cow health change?
        diff_string = str(diff_strings)
        # diff_string = str(diffs)

    prompt = \
f"""
The action "{action}" transforms the state from
```
{string0}
```
to
```
{string1}
```
The difference is
\"\"\"
{diff_string}
\"\"\"
"""
    prompt = textwrap.dedent(prompt).strip()
    return prompt

def transition_prompt(state0, state1):
    action = state1['info']['action']
    text0 = state0['obs']['text']
    text1 = state1['obs']['text']

    diffs = state_diffs(text0, text1)

    string0 = f"""{format_text_state(text0)}"""
    string1 = f"""{format_text_state(text1)}"""

    prompt = transition_string(action, string0, string1, diffs)
    return prompt

def main():
    replay_fname = f'{os.path.expanduser("~")}/logdir/hunter/replay.pkl'
    with open(replay_fname, 'rb') as f:
        replay = pickle.load(f)

    print('')
    # print(transition_prompt(replay[0], replay[1]))
    print('')
    print(transition_prompt(replay[1], replay[2]))

if __name__ == "__main__":
    main()


print('done')