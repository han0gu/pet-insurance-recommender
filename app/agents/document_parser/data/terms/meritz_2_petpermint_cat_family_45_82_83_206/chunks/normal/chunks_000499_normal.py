from langchain_core.documents import Document

chunk = Document(
    page_content=('【핵연료물질에 의하여 오염된 물질】\n'
 '원자핵분열 생성물을 포함합니다.\n'
 '⑥ 최초 계약의 보험계약일 이전에 이미 감염 또는 발병 한 질병 및 상해'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000499',
              'chunk_char_len': 78,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
