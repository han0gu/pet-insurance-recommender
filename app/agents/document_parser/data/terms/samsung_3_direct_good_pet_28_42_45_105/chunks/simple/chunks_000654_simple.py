from langchain_core.documents import Document

chunk = Document(
    page_content=('의 요구 또는 보험료 납입 연체로 해지된 경우 유사계약 청약일과 계약 청약일 사이\n'
 '에 제1항 제1호 또는 제2호에서 정한 질병으로 재진단 또는 치료를 받지 않았다면 계'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000654',
              'chunk_char_len': 93,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
