from langchain_core.documents import Document

chunk = Document(
    page_content=('여부를 판단합니다.| 분류항목 | 분류번호 |\n'
 '| --- | --- |\n'
 '| 1 . 머리의 으깸손상 | S07 |\n'
 '| 2 . 목의 골절 | S12 |\n'
 '| 3 . 흉추의 골절 및 흉추의 다발골절 | S22.0 - S22.1 |\n'
 '| 4 . 요추 및 골반의 골절 | S32 |\n'
 '| 5 . 대퇴골의 골절 | S72 |\n'
 '| 6 . 출산손상으로 인한 두개골골절 | P13.0 |\n'
 '| 7 . 두개골의 기타 출산손상 | P13.1 |\n'
 '| 8 . 척추 및 척수의 출산손상 | P11.5 |\n'
 '| 9 . 대퇴골의 출산손상 | P13.2 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000864',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
