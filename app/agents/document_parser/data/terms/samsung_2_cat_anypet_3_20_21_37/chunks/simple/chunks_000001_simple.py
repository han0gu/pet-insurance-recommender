from langchain_core.documents import Document

chunk = Document(
    page_content=('제1관 목적 및 용어의 정의\n'
 '제1조(목적)\n'
 "이 보험계약(이하 '계약'이라 합니다)은 보험계약자(이하 '계약자'라 합니다)와 보험회사(이하 '회사'라 합 니다) 사이에 보험증권에 "
 '기재된 반려동물의 상해 또는 질병에 대한 위험을 보장하기 위하여 체결됩니다.\n'
 '제2조(용어의 정의)\n'
 '이 계약에서 사용되는 용어의 정의는 이 계약의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '1. 계약 관련 용어'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000001',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
