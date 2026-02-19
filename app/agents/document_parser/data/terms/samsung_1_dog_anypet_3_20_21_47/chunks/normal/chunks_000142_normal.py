from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(보험금 등의 지급한도)\n'
 '① 회사는 1회의 보험사고에 대하여 다음과 같이 보상합니다. 이 경우 보상한도액과 자기부담금은 각 각 보험증권에 기재된 금액을 '
 '말합니다.\n'
 '1. 제1조(보상하는 손해) 제1항 제1호의 손해배상금: 보상한도액을 한도로 보상합니다. 2. 제1조(보상하는 손해) 제1항 제2호 '
 "'가'목, '나'목 또는 '마'목의 비용: 비용의 전액을 보상합니다. 3. 제1조(보상하는 손해) 제1항 제2호 '다'목 또는 '라'목의 "
 '비용: 이 비용과 제1호에 의한 보상액 의 합계액을 보상한도액 내에서 보상합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
