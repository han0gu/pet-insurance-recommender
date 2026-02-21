from langchain_core.documents import Document

chunk = Document(
    page_content=("- ② 회사는 계약의 청약을 받고 보험료 전액 또는 제1회 보험료(이하 '제1회 보험료 등'이라 합니다)를\n"
 '- 받은 경우에는 청약일부터 30일 이내에 승낙 또는 거절의 통지를 하며 통지가 없으면 승낙한 것으\n'
 '- 로 봅니다.\n'
 '- ③ 회사가 청약을 승낙한 때에는 지체없이 보험증권을 계약자에게 교부하여 드리며, 청약을 거절한 경\n'
 '- 우에는 거절통지와 함께 받은 금액을 계약자에게 돌려드립니다.\n'
 '- ④ 이미 성립한 계약을 연장하거나 변경하는 경우에는 회사는 보험증권에 그 사실을 기재함으로써 보\n'
 '- 험증권의 교부에 대신할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000038',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
