from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 특정질병 | 분류코드 | 항목명\n'
 'GAA002 | 외이도염 (말라세지아)\n'
 'GAA003 | 외이도염 (알러지성)\n'
 'GAA004 | 외이도염 (원인 불명)\n'
 'GAA006 | 외이염\n'
 'GBA001 | 중이염\n'
 'GCA001 | 내이염\n'
 'LAA001 | 농피증 / 세균성 피부염\n'
 'LAA002 | 말라세지아 피부염\n'
 'LAA003 | 피부 사상균증 · 곰팡이성 피부염\n'
 'LAA004 | 모낭염\n'
 'LAA005 | 모낭충증\n'
 'LAA006 | 식이 알러지\n'
 'LAA007 | 알러지 피부염 (항원 특이적)\n'
 'LAA008 | 아토피 (만성 피부염)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 198},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000685',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
