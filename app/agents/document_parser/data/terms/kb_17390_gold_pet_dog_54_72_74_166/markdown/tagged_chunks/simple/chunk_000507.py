from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항의 만나이는 계약일 현재 반려동물의 실제 만나이를 기준으로 하며, 이후 매\n'
 '- 년 계약 해당일에 나이가 증가하는 것으로 합니다.\n'
 '- \uf000 청약서에 기재된 반려동물의 나이 및 품종에 관한 사항이 사실과 다른 경우에는\n'
 '- 정정된 나이 및 품종에 해당하는 보험금 및 보험료로 변경합니다. 다만, 반려동\n'
 '- 물의 나이 및 품종이 정정되기 이전에는 "나이 및 품종이 정정되기 전에 적용된\n'
 '- 보험요율"의 "나이 및 품종이 정정된 후에 적용해야할 보험요율"에 대한 비율에\n'
 '| 따라 보험금을 | 삭감하여 지급합니다. |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000507',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
