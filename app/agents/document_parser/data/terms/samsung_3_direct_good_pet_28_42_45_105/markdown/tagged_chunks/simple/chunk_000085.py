from langchain_core.documents import Document

chunk = Document(
    page_content=('니다.- 1. 제13조(계약 전 알릴 의무)에 따라 계약자 또는 피보험자가 회사에 알린 내용이나\n'
 '- 건강진단 내용이 보험금 지급사유의 발생에 영향을 미쳤음을 회사가 증명하는 경\n'
 '- 우\n'
 '- 2. 제15조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는 경우\n'
 '- 3. 진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우. 다만, 진\n'
 '- 단계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는 경우\n'
 '- 시드 H자O 划厂2U INI'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
