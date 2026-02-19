from langchain_core.documents import Document

chunk = Document(
    page_content=('제8조(보험금 등의 지급한도)\n'
 '① 회사는 1회의 보험사고에 대하여 다음과 같이 보상합니다. 이 경우 보상한도액과 자기 부담금은 각각 보험증권에 기재된 금액을 '
 '말합니다.\n'
 '1. 제3조(보상하는 손해) 제1호의 손해배상금 : 보상한도액을 한도로 보상하되, 매회의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
