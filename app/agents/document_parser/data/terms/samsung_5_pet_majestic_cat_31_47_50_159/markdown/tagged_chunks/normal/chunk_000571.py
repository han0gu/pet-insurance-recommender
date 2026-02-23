from langchain_core.documents import Document

chunk = Document(
    page_content=('② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한\n'
 '지급보험금 결정에는 영향을 미치지 않습니다.# 제6조 (특별약관의 소멸)보험증권에 기재된 반려묘가 보험기간 중에 사망하여 이 '
 '추가특별약관에서 정한 보험금\n'
 '지급사유가 더이상 발생할 수 없는 경우에는 "보험료 및 해약환급금 산출방법서" 에 정\n'
 '하는 바에 따라 회사가 적립한 사망당시 이 추가특별약관의 계약자적립액 및 미경과보험'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000571',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
