from langchain_core.documents import Document

chunk = Document(
    page_content=('다.2. 제3조(보상하는 손해) 제2호 ‘가’목, ‘나’목 또는 ‘마’목의 비용 : 비용의 전액을 보상\n'
 '합니다.\n'
 '3. 제3조(보상하는 손해) 제2호 ‘다’목 또는 ‘라’목의 비용 : 이 비용과 제1호에 의한\n'
 '보상액의 합계액을 보상한도액내에서 보상합니다.② 보험기간 중 발생하는 사고에 대한 회사의 보상총액은 보험증권에 기재된 총 보상한도\n'
 '액을 한도로 합니다.제9조(의무보험과의 관계)① 회사는 이 특별약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
