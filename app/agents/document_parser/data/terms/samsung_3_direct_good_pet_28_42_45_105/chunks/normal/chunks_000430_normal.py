from langchain_core.documents import Document

chunk = Document(
    page_content=('74 / 181\n'
 '활(효력회복)일을 보험계약일로 하여 제3조(보험금의 지급사유) 제3항을 적용합니다.\n'
 '제23조 (강제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000430',
              'chunk_char_len': 96,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
