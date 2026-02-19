from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 제2조(특별면책조건의 내용) 제1항 제1호의 특정신체부위에 발생한 질병에 대하여 면책을 조건으로 체결한 후 보장개시일(책임개시일) '
 '이전에 동일한 특정신체부위 에 질병이 발생한 경우 2. 제2조(특별면책조건의 내용) 제1항 제2호의 특정질병에 대하여 면책을 조건으로 '
 '체결한 후 보장개시일(책임개시일) 이전에 동일한 특정질병이 발생한 경우\n'
 '제2조 (특별면책조건의 내용)\n'
 '① 이 특별약관에서 정한 회사가 보험금을 지급하지 않는 기간 중에 다음 각 호의 질병을'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000646',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
