from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조 (특별면책조건의 내용)\n'
 '① 이 특별약관에서 정한 회사가 보험금을 지급하지 않는 기간 중에 다음 각 호의 질병을 직접적인 원인으로 보험계약에서 정한 보험금 '
 '지급사유가 발생한 경우에 회사는 보험'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000823',
              'chunk_char_len': 110,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
