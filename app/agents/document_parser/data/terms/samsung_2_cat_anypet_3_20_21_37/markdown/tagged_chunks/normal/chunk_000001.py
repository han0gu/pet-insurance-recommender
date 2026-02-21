from langchain_core.documents import Document

chunk = Document(
    page_content=('니다) 사이에 보험증권에 기재된 반려동물의 상해 또는 질병에 대한 위험을 보장하기 위하여 체결됩니다.# 제2조(용어의 정의)이 계약에서 '
 '사용되는 용어의 정의는 이 계약의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.# 1. 계약 관련 용어- 가. 계약자: 회사와 '
 '계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다.\n'
 '- 나. 피보험자: 보험사고로 인하여 손해를 입은 사람(법인인 경우에는 그 이사 또는 법인의 업무\n'
 '- 를 집행하는 그 밖의 기관)을 말합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000001',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
