from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조 (보험금의 지급한도)\n'
 '① 회사는 제3조(보상하는 손해) 제2항의 손해에 대하여 다음과 같이 보상합니다. 이 경 우 보상한도액과 자기부담금은 각각 보험증권에 '
 '기재된 금액을 말합니다.\n'
 '1. 제3조(보상하는 손해) 제2항 제1호의 손해배상금 : 매회의 사고마다 자기부담금 10 만원을 초과하는 경우에 한하여 그 초과하는 '
 '배상책임 손해에 대한 금액을 보상한 -'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000572',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
