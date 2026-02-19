from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 보험수익자가 반려동물 양육자금Ⅰ을 일시에 지급받고자 요청한 때에는 회사는 평균 공시이율을 반영하여 연단위 복리로 할인한 금액과 이 '
 '특별약관의 보장부분 적용이율 을 반영하여 연단위 복리로 할인한 금액 중 큰 금액을 지급합니다.\n'
 '제3조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 86},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000472',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
