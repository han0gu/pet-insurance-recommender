from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약해당일에 나이가 증가하는 것으로 합니다.\n'
 '- ③ 반려견의 나이 및 품종에 관한 청약서상 기재사항이 사실과 다른 경우에는 정정된 나\n'
 '- 이 및 품종에 해당하는 보험금 및 보험료로 변경합니다. 다만, 반려동물의 나이 및 품\n'
 '- 종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에 적용된 보험료율」의 「나\n'
 '- 이 및 품종이 정정된 후에 적용해야할 보험료율」에 대한 비율에 따라 보험금을 삭감\n'
 '- 하여 지급합니다.\n'
 '<예시안내># [계약해당일 계산]최초계약일과 동일한 월, 일을 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000355',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
