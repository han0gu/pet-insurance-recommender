from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[소멸시효]\n'
 '- 47 -\n'
 '소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합니다. 보험금 지급사유가 2021년 4월 1일 에 발생하였음에도 2024년 4월 '
 '1일까지 보험금을 청구하지 않는 경우 소멸시효가 완성되어 보험 금 등을 지급받지 못할 수 있습니다.\n'
 '제 42조 (약관의 해석)\n'
 '① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라 다 르게 해석하지 않습니다.\n'
 '<용어풀이>\n'
 '[신의성실의 원칙]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
