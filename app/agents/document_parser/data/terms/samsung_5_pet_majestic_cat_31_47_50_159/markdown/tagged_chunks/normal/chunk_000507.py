from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다.\n'
 '# 제18조 (보험나이 등)- ① 이 특별약관에서의 반려묘의 나이는 만나이를 기준으로 합니다.\n'
 '- ② 제1항의 만나이는 계약일 현재 반려묘의 실제 만나이를 기준으로 하며, 이후 매년 계\n'
 '- 약해당일에 나이가 증가하는 것으로 합니다.\n'
 '- ③ 반려묘의 나이 및 품종에 관한 청약서상 기재사항이 사실과 다른 경우에는 정정된 나\n'
 '- 이 및 품종에 해당하는 보험금 및 보험료로 변경합니다. 다만, 반려동물의 나이 및 품\n'
 '- 종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에 적용된 보험료율」 의 「나'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000507',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
