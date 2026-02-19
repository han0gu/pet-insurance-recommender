from langchain_core.documents import Document

chunk = Document(
    page_content=('반려동물 사망위로금 특별약관\n'
 '제1조(보상하는 손해)\n'
 '① 회사는 보험증권에 기재된 반려동물이 보험기간 중에 사망한 경우 보험증권에 기재된 보험가입금 액을 보상하여 드립니다. ② 제1항의 '
 '사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경우 동물병원에서 발 급한 소견서를 제출하여야 합니다.\n'
 '제2조(보상하지 않는 손해)\n'
 '회사는 아래의 사유로 인한 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
