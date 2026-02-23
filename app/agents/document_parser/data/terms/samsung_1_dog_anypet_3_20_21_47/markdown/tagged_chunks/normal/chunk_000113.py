from langchain_core.documents import Document

chunk = Document(
    page_content=('비용에 대하여 보상한도액을 한도로 보상하여 드립니다.# 제4조(보험금 등의 지급한도)① 회사는 1회의 보험사고에 대하여 다음과 같이 '
 '보상합니다. 이 경우 보상한도액과 자기부담금은 각\n'
 '각 보험증권에 기재된 금액을 말합니다.- 1. 제1조(보상하는 손해) 제1항 제1호의 손해배상금: 보상한도액을 한도로 보상합니다.\n'
 "- 2. 제1조(보상하는 손해) 제1항 제2호 '가'목, '나'목 또는 '마'목의 비용: 비용의 전액을 보상합니다."),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
