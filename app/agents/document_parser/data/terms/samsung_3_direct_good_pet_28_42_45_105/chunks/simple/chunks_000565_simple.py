from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[핵연료물질]\n'
 '사용된 연료를 포함합니다. [핵연료물질에 의하여 오염된 물질] 원자핵 분열 생성물을 포함합니다.\n'
 '② 회사는 피보험자가 다음에 열거한 배상책임을 부담함으로써 입은 손해를 보상하지 않 습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000565',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
