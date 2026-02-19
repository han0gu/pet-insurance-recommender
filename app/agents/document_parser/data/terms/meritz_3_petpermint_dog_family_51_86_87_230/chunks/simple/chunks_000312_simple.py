from langchain_core.documents import Document

chunk = Document(
    page_content=('방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의 특성에 의한 사고\n'
 '⑤ 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '【핵연료물질】\n'
 '사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】\n'
 '원자핵분열 생성물을 포함합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 115},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000312',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
