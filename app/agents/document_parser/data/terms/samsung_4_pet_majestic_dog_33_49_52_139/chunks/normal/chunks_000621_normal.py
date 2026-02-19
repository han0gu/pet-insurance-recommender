from langchain_core.documents import Document

chunk = Document(
    page_content=('무 위반의 효과)가 적용됩니다.\n'
 '④ 부활(효력회복)되는 특별약관의 보장개시는 제1항 내지 제3항을 따릅니다. 이 경우 부 활(효력회복)일을 보험계약일로 하여 '
 '제3조(보험금의 지급사유) 제3항을 적용합니다.\n'
 '제23조 (강제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000621',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
