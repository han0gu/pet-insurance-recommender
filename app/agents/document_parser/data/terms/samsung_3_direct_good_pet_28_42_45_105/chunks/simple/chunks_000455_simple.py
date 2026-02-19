from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합니다) 중 에 보험증권에 기재된 반려견이 국내에서 수의사에게 '
 '이물 섭취 치료를 목적으로 이 물제거(내시경) 또는 이물제거(구토유도약물)를 받은 경우 연간 2회에 한하여 당일 피 보험자가 부담한 '
 '반려견의 치료에 사용된 비용(각종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000455',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
