from langchain_core.documents import Document

chunk = Document(
    page_content=('보장계약에 대해서는 갱신일 현재의 제도 또는 보험료 등을\n'
 '적용합니다.189\uf000 회사는 제2조(자동갱신 적용대상 계약의 자동갱신)에서\n'
 '정한 갱신제한 사유 및 제1항의 갱신보장계약 보험료에 대\n'
 '하여 갱신대상 보장계약의 보험기간이 끝나기 15일 전까지\n'
 '그 내용을 계약자에게 서면, 전화 또는 전자문서 등으로 안\n'
 '내하여 드립니다.\uf000 제1항 및 제2항에도 불구하고 법령 및 표준약관 변경으\n'
 '로 보장내용 등이 변경되어 약관이 개정되는 경우 보험기간\n'
 '이 끝나는 날 이전까지 중요사항 변경내역(갱신보험료 변경'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
