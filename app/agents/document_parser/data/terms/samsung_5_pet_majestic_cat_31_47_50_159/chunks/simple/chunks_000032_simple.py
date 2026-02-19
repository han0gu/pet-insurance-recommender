from langchain_core.documents import Document

chunk = Document(
    page_content=('6. 제4조(보험금 지급에 관한 세부규정) 제4항에 따라 보험금 지급사유에 대해 제3자 의 의견에 따르기로 한 경우\n'
 '<유의사항>\n'
 '분쟁조정은 이 약관의 (분쟁의 조정) 조항에 따라 금융감독원에 신청할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
