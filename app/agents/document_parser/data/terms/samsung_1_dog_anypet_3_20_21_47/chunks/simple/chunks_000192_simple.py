from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자는 정산기간 종료 후 5일 이내에 정산기간 종료 시점의 피보험자 수를 회사에 통지해야 합 니다. ② 회사는 제1항의 통지를 받은 '
 '때로부터 5일 이내에 정산기간에 해당하는 확정보험료를 산출하여 계 약자에게 통지하여야 합니다. ③ 확정보험료가 예치보험료보다 작은 경우에 '
 '회사는 그 차액을 제2항의 통지일로부터 5일 이내에 계 약자에게 돌려 드리며, 반대의 경우에는 계약자는 그 차액을 제2항의 통지를 받은 '
 '날로부터 5일 이내에 회사에 납입하여야 합니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 38},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000192',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
