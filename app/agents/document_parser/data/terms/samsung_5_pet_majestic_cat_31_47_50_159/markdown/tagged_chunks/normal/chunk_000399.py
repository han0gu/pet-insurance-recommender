from langchain_core.documents import Document

chunk = Document(
    page_content=('공시이율을 반영하여 연단위 복리로 할인한 금액과 이 특별약관의 보장부분 적용이율\n'
 '을 반영하여 연단위 복리로 할인한 금액 중 큰 금액을 지급합니다.# 제3조 (특별약관의 소멸)① 회사가 제1조(보험금의 지급사유)에서 '
 '정한 반려동물 양육자금Ⅰ을 지급한 때에는 그\n'
 '손해보상의 원인이 생긴 때부터 이 특별약관은 소멸되며 그 때부터 효력이 없습니다.\n'
 '이 경우 회사는 이 특별약관의 해약환급금을 지급하지 않습니다.\n'
 '② 피보험자가 보험기간 중에 이 특별약관에서 보장하지 않는 사유로 사망하였을 경우에'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000399',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
