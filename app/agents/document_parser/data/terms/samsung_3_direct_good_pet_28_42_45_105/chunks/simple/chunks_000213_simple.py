from langchain_core.documents import Document

chunk = Document(
    page_content=('[보험금 삭감]\n'
 '일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 보험 가입 후 기간이 경 과함에 따라 위험의 크기 및 정도가 점차 '
 '감소하는 위험에 대해 적용하여 보험 가입 후 일정기간 내에 보험사고가 발생할 경우 미리 정해진 비율로 보험금을 감액하여 지급하는 방법을 '
 '말합니다.\n'
 '[보험료 할증]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 51},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
