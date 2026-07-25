import tomllib
from pathlib import Path
import sys

try:
    p = Path('.streamlit/secrets.toml')
    # Read with utf-8-sig to strip a possible BOM
    data = tomllib.loads(p.read_text(encoding='utf-8-sig'))
    api_key = data.get('OPENROUTER_API_KEY')
    model = data.get('OPENROUTER_MODEL', 'openai/gpt-3.5-turbo')
    if not api_key:
        print('No OPENROUTER_API_KEY found in .streamlit/secrets.toml')
        sys.exit(2)

    from openai import OpenAI
    client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":"مرحبا، اختبر اتصال OpenRouter وأعطني جملة ترحيب قصيرة."}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    print('---OPENROUTER RESPONSE START---')
    print(content[:1000])
    print('---OPENROUTER RESPONSE END---')
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
