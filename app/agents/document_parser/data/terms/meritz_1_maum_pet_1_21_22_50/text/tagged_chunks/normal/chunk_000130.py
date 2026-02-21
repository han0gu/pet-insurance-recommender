from langchain_core.documents import Document

chunk = Document(
    page_content=('에 더하여 지급합니다. 그러나 계약자 또는 피보험자의 책임있는 사유로 지급이 지연된\n'
 '때에는 그 해당기간에 대한 이자는 더하여 지급하지 않습니다.제8조(보험금 등의 지급한도)① 회사는 1회의 보험사고에 대하여 다음과 같이 '
 '보상합니다. 이 경우 보상한도액과 자기\n'
 '부담금은 각각 보험증권에 기재된 금액을 말합니다.1. 제3조(보상하는 손해) 제1호의 손해배상금 : 보상한도액을 한도로 보상하되, '
 '매회의- 24 -사고마다 자기부담금 3만원을 초과하는 경우에 한하여 그 초과한 부분만 보상합니'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
