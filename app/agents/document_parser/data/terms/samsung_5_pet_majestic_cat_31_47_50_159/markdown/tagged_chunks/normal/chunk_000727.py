from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 대하여 가산이율을 적용하지 않습니다.\n'
 "- 6. 회사가 해지권을 행사하는 경우 위 표의 '청구일' 은 회사의 해지 의사표시(서면, 전자우편,\n"
 '- 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시 포함)가 보험계약자 또는 그의 대리\n'
 '- 인에게 도달한 날로 봅니다.\n'
 '- 135 -[별 표2] 장해분류표# <총 칙># 1. 장해의 정의- 가. "장해" 라 함은 상해 또는 질병에 대하여 치유된 후 신체에 '
 '남아있는 영구적인\n'
 '- 정신 또는 육체의 훼손상태 및 기능상실 상태를 말한다. 다만, 질병과 부상의 주'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000727',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
