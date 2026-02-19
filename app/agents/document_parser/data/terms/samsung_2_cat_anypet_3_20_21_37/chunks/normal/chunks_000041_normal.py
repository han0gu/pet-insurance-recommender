from langchain_core.documents import Document

chunk = Document(
    page_content=("① 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다. ② 회사는 계약의 청약을 받고 보험료 전액 또는 제1회 보험료(이하 '제1회 "
 "보험료 등'이라 합니다)를 받은 경우에는 청약일부터 30일 이내에 승낙 또는 거절의 통지를 하며 통지가 없으면 승낙한 것으 로 봅니다. "
 '③ 회사가 청약을 승낙한 때에는 지체없이 보험증권을 계약자에게 교부하여 드리며, 청약을 거절한 경 우에는 거절통지와 함께 받은 금액을 '
 '계약자에게 돌려드립니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 10},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
