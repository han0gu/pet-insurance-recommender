from langchain_core.documents import Document

chunk = Document(
    page_content=('- 45 -\n'
 '에 발생하였음에도 2024년 4월 1일까지 보험금을 청구하지 않는 경우 소멸시효가 완성되어 보험 금 등을 지급받지 못할 수 있습니다.\n'
 '제 42조 (약관의 해석)\n'
 '① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라 다 르게 해석하지 않습니다.\n'
 '<용어풀이>\n'
 '[신의성실의 원칙]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
